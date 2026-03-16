#include "index/ivf.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ann {
namespace {

constexpr int kKMeansIterations = 20;
constexpr uint32_t kDefaultSeed = 42;

struct ListEntry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DocId doc_id;
  VersionSet versions;
  Eigen::VectorXf vector;
  float norm{0.0f};
  std::vector<uint8_t> pq_code;
};

struct IndexData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint32_t dim{0};
  uint32_t nlist{0};
  VersionId version{0};
  bool use_pq{false};
  bool pq_residual{true};
  uint32_t M{0};
  uint32_t nbits{8};
  uint32_t Ks{0};
  uint32_t dsub{0};
  MatrixRM routing_centroids;
  std::vector<MatrixRM> pq_codebooks;
  std::vector<AlignedVector<ListEntry>> lists;
  std::unordered_set<DocId> doc_ids;
  uint64_t ntotal{0};
};

struct ListStats {
  uint64_t total{0};
  uint32_t min{0};
  double avg{0.0};
  double p99{0.0};
  uint32_t max{0};
};

ListStats ComputeListStats(const IndexData& data) {
  ListStats stats;
  if (data.lists.empty()) {
    return stats;
  }
  std::vector<uint32_t> sizes;
  sizes.reserve(data.lists.size());
  uint64_t total = 0;
  uint32_t min_sz = std::numeric_limits<uint32_t>::max();
  uint32_t max_sz = 0;
  for (const auto& list : data.lists) {
    const uint32_t sz = static_cast<uint32_t>(list.size());
    sizes.push_back(sz);
    total += sz;
    min_sz = std::min(min_sz, sz);
    max_sz = std::max(max_sz, sz);
  }
  stats.total = total;
  stats.min = min_sz;
  stats.max = max_sz;
  stats.avg = static_cast<double>(total) / static_cast<double>(sizes.size());
  std::sort(sizes.begin(), sizes.end());
  const double rank = 0.99 * static_cast<double>(sizes.size() - 1);
  const size_t lo = static_cast<size_t>(rank);
  const size_t hi = std::min(sizes.size() - 1, lo + 1);
  const double frac = rank - static_cast<double>(lo);
  stats.p99 = static_cast<double>(sizes[lo]) +
              (static_cast<double>(sizes[hi]) - static_cast<double>(sizes[lo])) * frac;
  return stats;
}

void LogListStats(const IndexData& data) {
  ListStats stats = ComputeListStats(data);
  std::cout << "[IVF] lists total=" << stats.total << " (ntotal=" << data.ntotal
            << ") min=" << stats.min << " avg=" << stats.avg << " p99=" << stats.p99
            << " max=" << stats.max << std::endl;
}

int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const IndexData& data) {
  float best = std::numeric_limits<float>::max();
  int best_idx = 0;
  for (int i = 0; i < data.routing_centroids.rows(); ++i) {
    float dist = (data.routing_centroids.row(i).transpose() - vec).squaredNorm();
    if (dist < best) {
      best = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

MatrixRM InitializeCentroids(Eigen::Ref<const MatrixRM> X, uint32_t nlist) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  MatrixRM centroids(nlist, dim);
  std::vector<int64_t> indices(num_vecs);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 gen(kDefaultSeed);
  std::shuffle(indices.begin(), indices.end(), gen);
  for (uint32_t i = 0; i < nlist; ++i) {
    centroids.row(i) = X.row(indices[i % num_vecs]);
  }
  return centroids;
}

void RunKMeans(Eigen::Ref<const MatrixRM> X, MatrixRM* centroids) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  const int64_t k = centroids->rows();
  std::vector<int> assignments(num_vecs, 0);

  for (int iter = 0; iter < kKMeansIterations; ++iter) {
    // Assignment step.
  #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
  #endif
    for (int64_t i = 0; i < num_vecs; ++i) {
      Eigen::VectorXf vec = X.row(i).transpose();
      float best = std::numeric_limits<float>::max();
      int best_idx = 0;
      for (int64_t c = 0; c < k; ++c) {
        float dist = (centroids->row(c).transpose() - vec).squaredNorm();
        if (dist < best) {
          best = dist;
          best_idx = static_cast<int>(c);
        }
      }
      assignments[i] = best_idx;
    }

    // Update step.
    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int64_t> counts(k, 0);
    #ifdef _OPENMP
      int num_threads = 1;
    #pragma omp parallel
      {
    #pragma omp single
        { num_threads = omp_get_num_threads(); }
      }
      std::vector<MatrixRM> partial_sums(
        static_cast<size_t>(num_threads), MatrixRM::Zero(k, dim));
      std::vector<std::vector<int64_t>> partial_counts(
        static_cast<size_t>(num_threads), std::vector<int64_t>(static_cast<size_t>(k), 0));
    
    #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        MatrixRM& thread_sum = partial_sums[static_cast<size_t>(tid)];
        std::vector<int64_t>& thread_counts = partial_counts[static_cast<size_t>(tid)];
    #pragma omp for schedule(static)
        for (int64_t i = 0; i < num_vecs; ++i) {
          const int assign = assignments[i];
          thread_sum.row(assign) += X.row(i);
          thread_counts[static_cast<size_t>(assign)]++;
        }
      }
    
      for (int t = 0; t < num_threads; ++t) {
        new_centroids += partial_sums[static_cast<size_t>(t)];
        const auto& thread_counts = partial_counts[static_cast<size_t>(t)];
        for (int64_t c = 0; c < k; ++c) {
          counts[static_cast<size_t>(c)] += thread_counts[static_cast<size_t>(c)];
        }
      }
    #else
      for (int64_t i = 0; i < num_vecs; ++i) {
        new_centroids.row(assignments[i]) += X.row(i);
        counts[assignments[i]]++;
      }
    #endif
    std::mt19937 gen(kDefaultSeed + iter);
    std::uniform_int_distribution<int64_t> dist_index(0, num_vecs - 1);
    for (int64_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        new_centroids.row(c) /= static_cast<float>(counts[c]);
      } else {
        int64_t repl = dist_index(gen);
        new_centroids.row(c) = X.row(repl);
      }
    }
    *centroids = new_centroids;
  }
}

class KMeansIVFIndex : public IVFIndex {
 public:
  Result<VersionId> Build(Eigen::Ref<const MatrixRM> Xw,
                          const std::vector<DocId>& ids,
                          const IVFParams& p,
                          VersionId index_version) override {
    if (Xw.rows() == 0 || Xw.cols() == 0) {
      return Status::InvalidArgument("Xw is empty");
    }
    if (ids.size() != static_cast<size_t>(Xw.rows())) {
      return Status::InvalidArgument("ids size mismatch");
    }
    if (!p.use_fixed_routing_centroids && p.nlist == 0) {
      return Status::InvalidArgument("nlist must be >0");
    }
    uint32_t dim = static_cast<uint32_t>(Xw.cols());
    if (p.pq.enable) {
      if (p.pq.M == 0) {
        return Status::InvalidArgument("PQ M must be >0");
      }
      if (dim % p.pq.M != 0) {
        return Status::InvalidArgument("dim must be divisible by PQ M");
      }
      if (p.pq.nbits == 0 || p.pq.nbits > 8) {
        return Status::InvalidArgument("PQ nbits must be in [1,8]");
      }
    }
    uint32_t nlist = 0;
    MatrixRM centroids;
    if (p.use_fixed_routing_centroids) {
      if (p.fixed_routing_centroids.rows() == 0 || p.fixed_routing_centroids.cols() == 0) {
        return Status::InvalidArgument("fixed_routing_centroids is empty");
      }
      if (p.fixed_routing_centroids.cols() != Xw.cols()) {
        return Status::InvalidArgument("fixed_routing_centroids dim mismatch");
      }
      centroids = p.fixed_routing_centroids;
      nlist = static_cast<uint32_t>(centroids.rows());
    } else {
      nlist = std::min<uint32_t>(p.nlist, static_cast<uint32_t>(Xw.rows()));
      centroids = InitializeCentroids(Xw, nlist);
      RunKMeans(Xw, &centroids);
    }

    auto data = std::make_unique<IndexData>();
    data->dim = dim;
    data->nlist = nlist;
    data->routing_centroids = std::move(centroids);
    data->use_pq = p.pq.enable;
    data->pq_residual = p.pq.residual;
    if (data->use_pq) {
      data->M = p.pq.M;
      data->nbits = p.pq.nbits;
      data->Ks = 1u << data->nbits;
      data->dsub = dim / data->M;
    } else {
      data->M = 0;
      data->nbits = 8;
      data->Ks = 0;
      data->dsub = 0;
      data->pq_codebooks.clear();
    }
    data->lists.clear();
    data->lists.resize(nlist);
    data->doc_ids.clear();
    data->ntotal = 0;
    if (data->use_pq) {
      data->pq_codebooks.resize(data->M);
      const int64_t num_vecs = Xw.rows();
      MatrixRM residuals(num_vecs, dim);
      #ifdef _OPENMP
      #pragma omp parallel for schedule(static)
      #endif 
      for (int64_t i = 0; i < num_vecs; ++i) {
        Eigen::VectorXf vec = Xw.row(i).transpose();
        int centroid = NearestCentroid(vec, *data);
        residuals.row(i) = Xw.row(i);
        if (data->pq_residual) {
          residuals.row(i) -= data->routing_centroids.row(centroid);
        }
      }
      #ifdef _OPENMP
      #pragma omp parallel for schedule(dynamic)
      #endif
      for (uint32_t m = 0; m < data->M; ++m) {
        const Eigen::Index offset = static_cast<Eigen::Index>(m * data->dsub);
        const Eigen::Index subdim = static_cast<Eigen::Index>(data->dsub);
        MatrixRM sub = residuals.block(0, offset, residuals.rows(), subdim);
        MatrixRM codebook = InitializeCentroids(sub, data->Ks);
        RunKMeans(sub, &codebook);
        data->pq_codebooks[m] = std::move(codebook);
      }
    }

    std::unique_lock lock(mu_);
    VersionId version = (index_version == 0) ? next_version_++ : index_version;
    data->version = version;
    for (auto& list : data->lists) {
      for (auto& entry : list) {
        entry.versions.index_version = version;
      }
    }
    data_map_[version] = std::move(data);
    latest_version_ = version;
    return version;
  }

  Status Add(const AlignedVector<VectorRecord>& recs) override {
    std::unique_lock lock(mu_);
    if (latest_version_ == 0) {
      return Status::InvalidArgument("Index not built");
    }
    auto it = data_map_.find(latest_version_);
    if (it == data_map_.end()) {
      return Status::NotFound("Latest version missing");
    }
    IndexData& data = *it->second;
    
    for (const auto& rec : recs) {
      if (rec.x.size() != data.dim) {
        return Status::InvalidArgument("Record dim mismatch");
      }
      if (!data.doc_ids.insert(rec.doc_id).second) {
        return Status::AlreadyExists("doc_id already present in IVF");
      }
    }
    
    struct PendingEntry {
      int centroid{0};
      ListEntry entry;
    };

    std::vector<PendingEntry> pending(recs.size());
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
        for (int64_t i = 0; i < static_cast<int64_t>(recs.size()); ++i) {
          const auto& rec = recs[static_cast<size_t>(i)];
          PendingEntry local;
          local.centroid = NearestCentroid(rec.x, data);
          local.entry.doc_id = rec.doc_id;
          local.entry.versions = rec.versions;
          local.entry.versions.index_version = data.version;
      if (data.use_pq) {
        local.entry.pq_code.resize(data.M);
        Eigen::VectorXf residual = rec.x;
        if (data.pq_residual) {
          residual -= data.routing_centroids.row(local.centroid).transpose();
        }
        for (uint32_t m = 0; m < data.M; ++m) {
          const MatrixRM& codebook = data.pq_codebooks[m];
          Eigen::Map<const Eigen::VectorXf> sub(residual.data() + m * data.dsub,
                                                static_cast<Eigen::Index>(data.dsub));
          float best = std::numeric_limits<float>::max();
          uint32_t best_idx = 0;
          for (uint32_t k = 0; k < data.Ks; ++k) {
            float dist = (codebook.row(k).transpose() - sub).squaredNorm();
            if (dist < best) {
              best = dist;
              best_idx = k;
            }
          }
          local.entry.pq_code[m] = static_cast<uint8_t>(best_idx);
        }
      } else {
        local.entry.vector = rec.x;
        local.entry.norm = local.entry.vector.squaredNorm();
      }
      pending[static_cast<size_t>(i)] = std::move(local);
    }

    for (auto& item : pending) {
      data.lists[static_cast<size_t>(item.centroid)].push_back(std::move(item.entry));
      ++data.ntotal;
    }
    // LogListStats(data);
    return Status::OK();
  }

  Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                              uint32_t topk,
                              uint32_t nprobe,
                              const VersionSet& route_versions,
                              uint8_t from_new) const override {
    if (topk == 0) {
      return Status::InvalidArgument("topk must be positive");
    }
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    if (static_cast<uint32_t>(qw.size()) != data.dim) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    const uint32_t probes = std::max<uint32_t>(1, std::min<uint32_t>(nprobe, data.nlist));
    const bool use_pq = data.use_pq && data.M > 0 && data.Ks > 0 && data.dsub > 0 &&
                        data.pq_codebooks.size() == data.M;

    std::vector<std::pair<float, uint32_t>> centroid_dists(data.nlist);
    for (uint32_t i = 0; i < data.nlist; ++i) {
      float dist = (data.routing_centroids.row(i).transpose() - qw).squaredNorm();
      centroid_dists[i] = {dist, i};
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + probes, centroid_dists.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    auto heap_cmp = [](const Candidate& a, const Candidate& b) {
      return a.approx_dist < b.approx_dist;
    };
    std::vector<Candidate> heap;
    heap.reserve(topk);
    uint64_t scanned = 0;
    const float qnorm = use_pq ? 0.0f : qw.squaredNorm();
    std::vector<float> distance_table;
    if (use_pq) {
      distance_table.resize(static_cast<size_t>(data.M) * data.Ks);
    }
    for (uint32_t pi = 0; pi < probes; ++pi) {
      uint32_t list_id = centroid_dists[pi].second;
      Eigen::VectorXf qres = qw;
      if (use_pq && data.pq_residual) {
        qres -= data.routing_centroids.row(list_id).transpose();
      }
      if (use_pq) {
        for (uint32_t m = 0; m < data.M; ++m) {
          const MatrixRM& codebook = data.pq_codebooks[m];
          Eigen::Map<const Eigen::VectorXf> qsub(qres.data() + m * data.dsub,
                                                 static_cast<Eigen::Index>(data.dsub));
          for (uint32_t k = 0; k < data.Ks; ++k) {
            float dist = (qsub - codebook.row(k).transpose()).squaredNorm();
            distance_table[static_cast<size_t>(m) * data.Ks + k] = dist;
          }
        }
      }
      for (const auto& entry : data.lists[list_id]) {
        ++scanned;
        Candidate cand;
        cand.doc_id = entry.doc_id;
        if (use_pq) {
          if (entry.pq_code.size() != data.M) {
            continue;
          }
          float approx = 0.0f;
          for (uint32_t m = 0; m < data.M; ++m) {
            uint8_t code = entry.pq_code[m];
            approx += distance_table[static_cast<size_t>(m) * data.Ks + code];
          }
          cand.approx_dist = approx;
        } else {
          const float dot = entry.vector.dot(qw);
          cand.approx_dist = qnorm + entry.norm - 2.0f * dot;
        }
        cand.rerank_dist = cand.approx_dist;
        cand.versions = entry.versions;
        cand.versions.index_version = data.version;
        cand.from_new = from_new;
        if (heap.size() < topk) {
          heap.push_back(std::move(cand));
          std::push_heap(heap.begin(), heap.end(), heap_cmp);
        } else if (!heap.empty() && cand.approx_dist < heap.front().approx_dist) {
          std::pop_heap(heap.begin(), heap.end(), heap_cmp);
          heap.back() = std::move(cand);
          std::push_heap(heap.begin(), heap.end(), heap_cmp);
        }
      }
    }

    if (heap.empty()) {
      SearchResult result;
      result.scanned_candidates = scanned;
      return result;
    }

    std::sort(heap.begin(), heap.end(),
              [](const Candidate& a, const Candidate& b) { return a.approx_dist < b.approx_dist; });
    SearchResult result;
    result.topk = std::move(heap);
    result.scanned_candidates = scanned;
    return result;
  }

  Result<MatrixRM> GetRoutingCentroids(const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    return it->second->routing_centroids;
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }

  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

 private:
  mutable std::shared_mutex mu_;
  std::unordered_map<VersionId, std::unique_ptr<IndexData>> data_map_;
  VersionId next_version_{1};
  VersionId latest_version_{0};
};

}  // namespace

std::shared_ptr<IVFIndex> CreateIVFIndex() { return std::make_shared<KMeansIVFIndex>(); }

}  // namespace ann
