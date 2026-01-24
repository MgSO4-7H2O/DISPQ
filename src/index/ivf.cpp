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
};

struct IndexData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint32_t dim{0};
  uint32_t nlist{0};
  VersionId version{0};
  MatrixRM centroids;
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
  for (int i = 0; i < data.centroids.rows(); ++i) {
    float dist = (data.centroids.row(i).transpose() - vec).squaredNorm();
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
    for (int64_t i = 0; i < num_vecs; ++i) {
      new_centroids.row(assignments[i]) += X.row(i);
      counts[assignments[i]]++;
    }
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
    if (p.nlist == 0) {
      return Status::InvalidArgument("nlist must be >0");
    }
    uint32_t dim = static_cast<uint32_t>(Xw.cols());
    const uint32_t nlist = std::min<uint32_t>(p.nlist, static_cast<uint32_t>(Xw.rows()));

    MatrixRM centroids = InitializeCentroids(Xw, nlist);
    RunKMeans(Xw, &centroids);

    auto data = std::make_unique<IndexData>();
    data->dim = dim;
    data->nlist = nlist;
    data->centroids = std::move(centroids);
    data->lists.clear();
    data->lists.resize(nlist);
    data->doc_ids.clear();
    data->ntotal = 0;

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
      int centroid = NearestCentroid(rec.x, data);
      ListEntry entry;
      entry.doc_id = rec.doc_id;
      entry.versions = rec.versions;
      entry.versions.index_version = data.version;
      entry.vector = rec.x;
      entry.norm = entry.vector.squaredNorm();
      data.lists[centroid].push_back(std::move(entry));
      ++data.ntotal;
    }
    LogListStats(data);
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

    std::vector<std::pair<float, uint32_t>> centroid_dists(data.nlist);
    for (uint32_t i = 0; i < data.nlist; ++i) {
      float dist = (data.centroids.row(i).transpose() - qw).squaredNorm();
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
    const float qnorm = qw.squaredNorm();
    for (uint32_t pi = 0; pi < probes; ++pi) {
      uint32_t list_id = centroid_dists[pi].second;
      for (const auto& entry : data.lists[list_id]) {
        ++scanned;
        Candidate cand;
        cand.doc_id = entry.doc_id;
        const float dot = entry.vector.dot(qw);
        cand.approx_dist = qnorm + entry.norm - 2.0f * dot;
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
