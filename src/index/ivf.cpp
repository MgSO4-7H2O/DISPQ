#include "index/ivf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ann {
namespace {

constexpr uint32_t kDefaultKMeansIterations = 20;
constexpr uint32_t kDefaultSeed = 42;
constexpr double kDefaultNQEEps = 1e-6;

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
  std::vector<std::vector<uint64_t>> pq_counts;
  double nqe_baseline{0.0};
  double nqe_ema{0.0};
  uint64_t online_pq_batch_count{0};
  double warmup_nqe_sum{0.0};
  uint32_t warmup_seen_batches{0};
  std::vector<AlignedVector<ListEntry>> lists;
  std::unordered_set<DocId> doc_ids;
  uint64_t ntotal{0};
};

int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const IndexData& data) {
  float best = std::numeric_limits<float>::max();
  int best_idx = 0;
  for (int i = 0; i < data.routing_centroids.rows(); ++i) {
    const float dist = (data.routing_centroids.row(i).transpose() - vec).squaredNorm();
    if (dist < best) {
      best = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

uint32_t NearestCodeword(Eigen::Ref<const Eigen::VectorXf> sub,
                         const MatrixRM& codebook,
                         float* best_dist_out) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_idx = 0;
  for (uint32_t k = 0; k < static_cast<uint32_t>(codebook.rows()); ++k) {
    const float dist = (codebook.row(static_cast<Eigen::Index>(k)).transpose() - sub).squaredNorm();
    if (dist < best) {
      best = dist;
      best_idx = k;
    }
  }
  if (best_dist_out != nullptr) {
    *best_dist_out = best;
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
    centroids.row(static_cast<Eigen::Index>(i)) = X.row(indices[i % num_vecs]);
  }
  return centroids;
}

void RunKMeans(Eigen::Ref<const MatrixRM> X, MatrixRM* centroids, uint32_t iterations) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  const int64_t k = centroids->rows();
  std::vector<int> assignments(num_vecs, 0);

  const uint32_t kmeans_iters = (iterations == 0) ? kDefaultKMeansIterations : iterations;
  for (uint32_t iter = 0; iter < kmeans_iters; ++iter) {
    // Assignment step.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t i = 0; i < num_vecs; ++i) {
      const Eigen::VectorXf vec = X.row(i).transpose();
      float best = std::numeric_limits<float>::max();
      int best_idx = 0;
      for (int64_t c = 0; c < k; ++c) {
        const float dist = (centroids->row(c).transpose() - vec).squaredNorm();
        if (dist < best) {
          best = dist;
          best_idx = static_cast<int>(c);
        }
      }
      assignments[static_cast<size_t>(i)] = best_idx;
    }

    // Update step.
    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int64_t> counts(static_cast<size_t>(k), 0);
#ifdef _OPENMP
    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
      { num_threads = omp_get_num_threads(); }
    }
    std::vector<MatrixRM> partial_sums(static_cast<size_t>(num_threads), MatrixRM::Zero(k, dim));
    std::vector<std::vector<int64_t>> partial_counts(
        static_cast<size_t>(num_threads), std::vector<int64_t>(static_cast<size_t>(k), 0));

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      MatrixRM& thread_sum = partial_sums[static_cast<size_t>(tid)];
      std::vector<int64_t>& thread_counts = partial_counts[static_cast<size_t>(tid)];
#pragma omp for schedule(static)
      for (int64_t i = 0; i < num_vecs; ++i) {
        const int assign = assignments[static_cast<size_t>(i)];
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
      new_centroids.row(assignments[static_cast<size_t>(i)]) += X.row(i);
      counts[static_cast<size_t>(assignments[static_cast<size_t>(i)])]++;
    }
#endif
    std::mt19937 gen(kDefaultSeed + iter);
    std::uniform_int_distribution<int64_t> dist_index(0, num_vecs - 1);
    for (int64_t c = 0; c < k; ++c) {
      if (counts[static_cast<size_t>(c)] > 0) {
        new_centroids.row(c) /= static_cast<float>(counts[static_cast<size_t>(c)]);
      } else {
        const int64_t repl = dist_index(gen);
        new_centroids.row(c) = X.row(repl);
      }
    }
    *centroids = new_centroids;
  }
}

uint32_t ClampTopCount(double ratio, uint32_t total) {
  if (total == 0) {
    return 0;
  }
  if (!std::isfinite(ratio)) {
    return total;
  }
  const double r = std::clamp(ratio, 0.0, 1.0);
  const uint32_t k = static_cast<uint32_t>(std::ceil(r * static_cast<double>(total)));
  return std::max<uint32_t>(1, std::min<uint32_t>(k, total));
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
    const uint32_t dim = static_cast<uint32_t>(Xw.cols());
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
    if (p.kmeans_iterations == 0) {
      return Status::InvalidArgument("kmeans_iterations must be >0");
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
      RunKMeans(Xw, &centroids, p.kmeans_iterations);
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
      data->pq_counts.clear();
    }
    data->lists.clear();
    data->lists.resize(nlist);
    data->doc_ids.clear();
    data->ntotal = 0;

    if (data->use_pq) {
      data->pq_codebooks.resize(data->M);
      if (p.use_fixed_pq_codebooks) {
        if (p.fixed_pq_codebooks.size() != data->M) {
          return Status::InvalidArgument("fixed_pq_codebooks size mismatch");
        }
        for (uint32_t m = 0; m < data->M; ++m) {
          const MatrixRM& book = p.fixed_pq_codebooks[static_cast<size_t>(m)];
          if (book.rows() != static_cast<Eigen::Index>(data->Ks) ||
              book.cols() != static_cast<Eigen::Index>(data->dsub)) {
            return Status::InvalidArgument("fixed_pq_codebooks shape mismatch");
          }
          data->pq_codebooks[static_cast<size_t>(m)] = book;
        }
      } else {
        const int64_t num_vecs = Xw.rows();
        MatrixRM residuals(num_vecs, dim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < num_vecs; ++i) {
          const Eigen::VectorXf vec = Xw.row(i).transpose();
          const int centroid = NearestCentroid(vec, *data);
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
          RunKMeans(sub, &codebook, p.kmeans_iterations);
          data->pq_codebooks[static_cast<size_t>(m)] = std::move(codebook);
        }
      }

      std::vector<std::vector<uint64_t>> computed_counts(
          data->M, std::vector<uint64_t>(data->Ks, 0));
      double nqe_sum = 0.0;
      for (int64_t i = 0; i < Xw.rows(); ++i) {
        const Eigen::VectorXf vec = Xw.row(i).transpose();
        const int centroid = NearestCentroid(vec, *data);
        Eigen::VectorXf residual = vec;
        if (data->pq_residual) {
          residual -= data->routing_centroids.row(centroid).transpose();
        }
        double err2 = 0.0;
        double r2 = 0.0;
        for (uint32_t m = 0; m < data->M; ++m) {
          Eigen::Map<const Eigen::VectorXf> sub(
              residual.data() + static_cast<Eigen::Index>(m * data->dsub),
              static_cast<Eigen::Index>(data->dsub));
          float best_dist = 0.0f;
          const uint32_t k =
              NearestCodeword(sub, data->pq_codebooks[static_cast<size_t>(m)], &best_dist);
          computed_counts[static_cast<size_t>(m)][static_cast<size_t>(k)]++;
          err2 += static_cast<double>(best_dist);
          r2 += static_cast<double>(sub.squaredNorm());
        }
        nqe_sum += err2 / (r2 + kDefaultNQEEps);
      }
      const double computed_baseline = nqe_sum / static_cast<double>(Xw.rows());

      bool use_fixed_counts = false;
      if (!p.fixed_pq_counts.empty()) {
        if (p.fixed_pq_counts.size() != data->M) {
          return Status::InvalidArgument("fixed_pq_counts size mismatch");
        }
        for (uint32_t m = 0; m < data->M; ++m) {
          if (p.fixed_pq_counts[static_cast<size_t>(m)].size() != data->Ks) {
            return Status::InvalidArgument("fixed_pq_counts shape mismatch");
          }
        }
        use_fixed_counts = true;
      }
      data->pq_counts = use_fixed_counts ? p.fixed_pq_counts : computed_counts;
      data->nqe_baseline =
          (p.fixed_pq_baseline_nqe > 0.0) ? p.fixed_pq_baseline_nqe : computed_baseline;
      data->nqe_ema = (p.fixed_pq_ema_nqe > 0.0) ? p.fixed_pq_ema_nqe : data->nqe_baseline;
    }

    std::unique_lock lock(mu_);
    const VersionId version = (index_version == 0) ? next_version_++ : index_version;
    data->version = version;
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
    Status validate = ValidateRecordsForInsertLocked(data, recs);
    if (!validate.ok()) {
      return validate;
    }

    std::vector<int> centroids(recs.size(), 0);
    AlignedVector<ListEntry> entries(recs.size());

    for (size_t i = 0; i < recs.size(); ++i) {
      const auto& rec = recs[i];
      ListEntry entry;
      const int centroid = NearestCentroid(rec.x, data);
      entry.doc_id = rec.doc_id;
      entry.versions = rec.versions;
      entry.versions.index_version = data.version;
      if (data.use_pq) {
        entry.pq_code.resize(data.M);
        Eigen::VectorXf residual = rec.x;
        if (data.pq_residual) {
          residual -= data.routing_centroids.row(centroid).transpose();
        }
        for (uint32_t m = 0; m < data.M; ++m) {
          const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
          Eigen::Map<const Eigen::VectorXf> sub(
              residual.data() + static_cast<Eigen::Index>(m * data.dsub),
              static_cast<Eigen::Index>(data.dsub));
          entry.pq_code[static_cast<size_t>(m)] =
              static_cast<uint8_t>(NearestCodeword(sub, codebook, nullptr));
        }
      } else {
        entry.vector = rec.x;
        entry.norm = entry.vector.squaredNorm();
      }
      centroids[i] = centroid;
      entries[i] = std::move(entry);
    }

    CommitPendingLocked(&data, centroids, &entries);
    return Status::OK();
  }

  Result<OnlinePQUpdateStats> AddWithOnlinePQ(
      const AlignedVector<VectorRecord>& recs,
      const OnlinePQUpdateOptions& options) override {
    if (options.qe_ratio_threshold <= 0.0) {
      return Status::InvalidArgument("qe_ratio_threshold must be > 0");
    }
    if (options.ema_alpha <= 0.0 || options.ema_alpha > 1.0) {
      return Status::InvalidArgument("ema_alpha must be in (0,1]");
    }
    if (options.nqe_eps <= 0.0) {
      return Status::InvalidArgument("nqe_eps must be > 0");
    }
    if (options.warmup_enable && options.warmup_batches == 0) {
      return Status::InvalidArgument("warmup_batches must be > 0 when warmup_enable=true");
    }
    if (options.partial_top_alpha &&
        (options.partial_alpha <= 0.0 || options.partial_alpha > 1.0)) {
      return Status::InvalidArgument("partial_alpha must be in (0,1]");
    }
    if (options.partial_top_lambda &&
        (options.partial_lambda <= 0.0 || options.partial_lambda > 1.0)) {
      return Status::InvalidArgument("partial_lambda must be in (0,1]");
    }

    OnlinePQUpdateStats stats;
    stats.processed_vectors = static_cast<uint32_t>(recs.size());

    std::unique_lock lock(mu_);
    if (latest_version_ == 0) {
      return Status::InvalidArgument("Index not built");
    }
    auto it = data_map_.find(latest_version_);
    if (it == data_map_.end()) {
      return Status::NotFound("Latest version missing");
    }
    IndexData& data = *it->second;
    Status validate = ValidateRecordsForInsertLocked(data, recs);
    if (!validate.ok()) {
      return validate;
    }
    if (recs.empty()) {
      stats.use_online_pq = data.use_pq && data.pq_residual;
      stats.nqe_baseline = data.nqe_baseline;
      stats.nqe_ema = data.nqe_ema;
      stats.qe_ratio = (data.nqe_baseline > 0.0) ? (data.nqe_ema / data.nqe_baseline) : 1.0;
      return stats;
    }

    const bool can_online = data.use_pq && data.pq_residual && data.M > 0 && data.Ks > 0 &&
                            data.dsub > 0 && data.pq_codebooks.size() == data.M &&
                            data.pq_counts.size() == data.M;
    if (!can_online) {
      std::vector<int> centroids(recs.size(), 0);
      AlignedVector<ListEntry> entries(recs.size());
      for (size_t i = 0; i < recs.size(); ++i) {
        const auto& rec = recs[i];
        ListEntry entry;
        const int centroid = NearestCentroid(rec.x, data);
        entry.doc_id = rec.doc_id;
        entry.versions = rec.versions;
        entry.versions.index_version = data.version;
        if (data.use_pq) {
          entry.pq_code.resize(data.M);
          Eigen::VectorXf residual = rec.x;
          if (data.pq_residual) {
            residual -= data.routing_centroids.row(centroid).transpose();
          }
          for (uint32_t m = 0; m < data.M; ++m) {
            const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
            Eigen::Map<const Eigen::VectorXf> sub(
                residual.data() + static_cast<Eigen::Index>(m * data.dsub),
                static_cast<Eigen::Index>(data.dsub));
            entry.pq_code[static_cast<size_t>(m)] =
                static_cast<uint8_t>(NearestCodeword(sub, codebook, nullptr));
          }
        } else {
          entry.vector = rec.x;
          entry.norm = entry.vector.squaredNorm();
        }
        centroids[i] = centroid;
        entries[i] = std::move(entry);
      }
      CommitPendingLocked(&data, centroids, &entries);
      return stats;
    }

    stats.use_online_pq = true;
    const double eps = std::max(1e-12, options.nqe_eps);
    const double ema_alpha = options.ema_alpha;
    const uint32_t n = static_cast<uint32_t>(recs.size());

    std::vector<int> centroids(recs.size(), 0);
    AlignedVector<Eigen::VectorXf> residuals;
    residuals.reserve(recs.size());
    std::vector<std::vector<uint8_t>> codes_before(
        recs.size(), std::vector<uint8_t>(data.M, 0));
    std::vector<std::vector<uint32_t>> batch_cnt(
        data.M, std::vector<uint32_t>(data.Ks, 0));
    std::vector<MatrixRM> sum_vec;
    std::vector<MatrixRM> sum_err;
    sum_vec.reserve(data.M);
    sum_err.reserve(data.M);
    for (uint32_t m = 0; m < data.M; ++m) {
      sum_vec.emplace_back(MatrixRM::Zero(data.Ks, data.dsub));
      sum_err.emplace_back(MatrixRM::Zero(data.Ks, data.dsub));
    }
    std::vector<double> sub_err_sum(data.M, 0.0);
    std::vector<double> sub_energy_sum(data.M, 0.0);

    double nqe_sum = 0.0;
    for (size_t i = 0; i < recs.size(); ++i) {
      const auto& rec = recs[i];
      const int centroid = NearestCentroid(rec.x, data);
      centroids[i] = centroid;

      Eigen::VectorXf residual = rec.x;
      if (data.pq_residual) {
        residual -= data.routing_centroids.row(centroid).transpose();
      }
      residuals.push_back(residual);

      double err2 = 0.0;
      double r2 = 0.0;
      for (uint32_t m = 0; m < data.M; ++m) {
        const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
        Eigen::Map<const Eigen::VectorXf> sub(
            residual.data() + static_cast<Eigen::Index>(m * data.dsub),
            static_cast<Eigen::Index>(data.dsub));
        float best_dist = 0.0f;
        const uint32_t best_idx = NearestCodeword(sub, codebook, &best_dist);
        codes_before[i][static_cast<size_t>(m)] = static_cast<uint8_t>(best_idx);
        batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(best_idx)]++;
        sum_vec[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(best_idx)) +=
            sub.transpose();
        sum_err[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(best_idx)) +=
            (sub - codebook.row(static_cast<Eigen::Index>(best_idx)).transpose()).transpose();

        const double e2 = static_cast<double>(best_dist);
        const double en = static_cast<double>(sub.squaredNorm());
        err2 += e2;
        r2 += en;
        sub_err_sum[static_cast<size_t>(m)] += e2;
        sub_energy_sum[static_cast<size_t>(m)] += en;
      }
      nqe_sum += err2 / (r2 + eps);
    }

    stats.nqe_batch = nqe_sum / static_cast<double>(n);
    data.online_pq_batch_count++;
    if (options.warmup_enable && data.online_pq_batch_count <= options.warmup_batches) {
      data.warmup_nqe_sum += stats.nqe_batch;
      data.warmup_seen_batches++;
      data.nqe_baseline = data.warmup_nqe_sum / static_cast<double>(data.warmup_seen_batches);
      data.nqe_ema = data.nqe_baseline;
      stats.in_warmup = true;
      stats.warmup_batches_left =
          static_cast<uint32_t>(options.warmup_batches - data.online_pq_batch_count);
      stats.nqe_baseline = data.nqe_baseline;
      stats.nqe_ema = data.nqe_ema;
      stats.qe_ratio = 1.0;
      stats.trigger_update = false;
    } else {
      if (!std::isfinite(data.nqe_baseline) || data.nqe_baseline <= 0.0) {
        data.nqe_baseline = stats.nqe_batch;
      }
      if (!std::isfinite(data.nqe_ema) || data.nqe_ema <= 0.0) {
        data.nqe_ema = stats.nqe_batch;
      } else {
        data.nqe_ema = ema_alpha * stats.nqe_batch + (1.0 - ema_alpha) * data.nqe_ema;
      }
      stats.nqe_baseline = data.nqe_baseline;
      stats.nqe_ema = data.nqe_ema;
      stats.qe_ratio = data.nqe_ema / std::max(eps, data.nqe_baseline);
      stats.trigger_update = options.enable && stats.qe_ratio > options.qe_ratio_threshold;
      stats.in_warmup = false;
      stats.warmup_batches_left = 0;
    }

    std::vector<std::vector<uint8_t>> codes_for_insert = codes_before;
    if (stats.trigger_update) {
      std::vector<uint8_t> subspace_selected(data.M, 1);
      if (options.partial_top_alpha) {
        std::fill(subspace_selected.begin(), subspace_selected.end(), 0);
        const uint32_t keep = ClampTopCount(options.partial_alpha, data.M);
        std::vector<std::pair<double, uint32_t>> scored;
        scored.reserve(data.M);
        for (uint32_t m = 0; m < data.M; ++m) {
          const double score = sub_err_sum[static_cast<size_t>(m)] /
                               (sub_energy_sum[static_cast<size_t>(m)] + eps);
          scored.emplace_back(score, m);
        }
        std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (uint32_t i = 0; i < keep; ++i) {
          subspace_selected[static_cast<size_t>(scored[static_cast<size_t>(i)].second)] = 1;
        }
      }

      std::vector<std::vector<uint8_t>> codeword_selected(
          data.M, std::vector<uint8_t>(data.Ks, 0));
      if (options.partial_top_lambda) {
        struct CodewordScore {
          double score{0.0};
          uint32_t m{0};
          uint32_t k{0};
        };
        std::vector<CodewordScore> scored;
        scored.reserve(static_cast<size_t>(data.M) * data.Ks);
        for (uint32_t m = 0; m < data.M; ++m) {
          if (subspace_selected[static_cast<size_t>(m)] == 0) {
            continue;
          }
          for (uint32_t k = 0; k < data.Ks; ++k) {
            if (batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)] == 0) {
              continue;
            }
            const double score =
                sum_err[static_cast<size_t>(m)]
                    .row(static_cast<Eigen::Index>(k))
                    .squaredNorm();
            scored.push_back(CodewordScore{score, m, k});
          }
        }
        if (!scored.empty()) {
          const uint32_t keep = ClampTopCount(
              options.partial_lambda, static_cast<uint32_t>(data.M * data.Ks));
          const uint32_t cap = std::min<uint32_t>(keep, static_cast<uint32_t>(scored.size()));
          std::partial_sort(scored.begin(), scored.begin() + cap, scored.end(),
                            [](const CodewordScore& a, const CodewordScore& b) {
                              return a.score > b.score;
                            });
          for (uint32_t i = 0; i < cap; ++i) {
            const auto& s = scored[static_cast<size_t>(i)];
            codeword_selected[static_cast<size_t>(s.m)][static_cast<size_t>(s.k)] = 1;
          }
        }
      } else {
        for (uint32_t m = 0; m < data.M; ++m) {
          if (subspace_selected[static_cast<size_t>(m)] == 0) {
            continue;
          }
          for (uint32_t k = 0; k < data.Ks; ++k) {
            if (batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)] > 0) {
              codeword_selected[static_cast<size_t>(m)][static_cast<size_t>(k)] = 1;
            }
          }
        }
      }

      std::vector<uint8_t> subspace_updated(data.M, 0);
      double drift_sq = 0.0;
      for (uint32_t m = 0; m < data.M; ++m) {
        if (subspace_selected[static_cast<size_t>(m)] == 0) {
          continue;
        }
        for (uint32_t k = 0; k < data.Ks; ++k) {
          if (codeword_selected[static_cast<size_t>(m)][static_cast<size_t>(k)] == 0) {
            continue;
          }
          const uint32_t cnt = batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)];
          if (cnt == 0) {
            continue;
          }

          const uint64_t old_n = data.pq_counts[static_cast<size_t>(m)][static_cast<size_t>(k)];
          const uint64_t new_n = old_n + static_cast<uint64_t>(cnt);
          const Eigen::VectorXf z_old =
              data.pq_codebooks[static_cast<size_t>(m)]
                  .row(static_cast<Eigen::Index>(k))
                  .transpose();
          const Eigen::VectorXf mu_batch =
              sum_vec[static_cast<size_t>(m)]
                  .row(static_cast<Eigen::Index>(k))
                  .transpose() /
              static_cast<float>(cnt);
          const Eigen::VectorXf z_new =
              (static_cast<float>(old_n) * z_old + static_cast<float>(cnt) * mu_batch) /
              static_cast<float>(new_n);
          const Eigen::VectorXf diff = z_new - z_old;
          drift_sq += static_cast<double>(diff.squaredNorm());
          data.pq_codebooks[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(k)) =
              z_new.transpose();
          data.pq_counts[static_cast<size_t>(m)][static_cast<size_t>(k)] = new_n;
          stats.updated_codewords++;
          if (subspace_updated[static_cast<size_t>(m)] == 0) {
            subspace_updated[static_cast<size_t>(m)] = 1;
            stats.updated_subspaces++;
          }
        }
      }
      stats.updated_codebook = stats.updated_codewords > 0;
      stats.codebook_drift_l2 = std::sqrt(drift_sq);

      if (stats.updated_codebook && options.reencode_batch_after_update) {
        for (size_t i = 0; i < recs.size(); ++i) {
          const Eigen::VectorXf& residual = residuals[static_cast<size_t>(i)];
          for (uint32_t m = 0; m < data.M; ++m) {
            const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
            Eigen::Map<const Eigen::VectorXf> sub(
                residual.data() + static_cast<Eigen::Index>(m * data.dsub),
                static_cast<Eigen::Index>(data.dsub));
            codes_for_insert[static_cast<size_t>(i)][static_cast<size_t>(m)] =
                static_cast<uint8_t>(NearestCodeword(sub, codebook, nullptr));
          }
        }
        stats.reencoded_batch = true;
      }
    }

    AlignedVector<ListEntry> entries(recs.size());
    for (size_t i = 0; i < recs.size(); ++i) {
      const auto& rec = recs[i];
      ListEntry entry;
      entry.doc_id = rec.doc_id;
      entry.versions = rec.versions;
      entry.versions.index_version = data.version;
      entry.pq_code = std::move(codes_for_insert[static_cast<size_t>(i)]);
      entries[i] = std::move(entry);
    }
    CommitPendingLocked(&data, centroids, &entries);
    return stats;
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
      const float dist = (data.routing_centroids.row(static_cast<Eigen::Index>(i)).transpose() - qw)
                             .squaredNorm();
      centroid_dists[static_cast<size_t>(i)] = {dist, i};
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
      const uint32_t list_id = centroid_dists[static_cast<size_t>(pi)].second;
      Eigen::VectorXf qres = qw;
      if (use_pq && data.pq_residual) {
        qres -= data.routing_centroids.row(static_cast<Eigen::Index>(list_id)).transpose();
      }
      if (use_pq) {
        for (uint32_t m = 0; m < data.M; ++m) {
          const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
          Eigen::Map<const Eigen::VectorXf> qsub(
              qres.data() + static_cast<Eigen::Index>(m * data.dsub),
              static_cast<Eigen::Index>(data.dsub));
          for (uint32_t k = 0; k < data.Ks; ++k) {
            const float dist =
                (qsub - codebook.row(static_cast<Eigen::Index>(k)).transpose()).squaredNorm();
            distance_table[static_cast<size_t>(m) * data.Ks + k] = dist;
          }
        }
      }
      for (const auto& entry : data.lists[static_cast<size_t>(list_id)]) {
        ++scanned;
        Candidate cand;
        cand.doc_id = entry.doc_id;
        if (use_pq) {
          if (entry.pq_code.size() != data.M) {
            continue;
          }
          float approx = 0.0f;
          for (uint32_t m = 0; m < data.M; ++m) {
            const uint8_t code = entry.pq_code[static_cast<size_t>(m)];
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

  Result<PQRuntimeState> GetPQRuntimeState(const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    PQRuntimeState state;
    state.use_pq = data.use_pq;
    state.pq_residual = data.pq_residual;
    state.M = data.M;
    state.Ks = data.Ks;
    state.dsub = data.dsub;
    state.codebooks = data.pq_codebooks;
    state.counts = data.pq_counts;
    state.nqe_baseline = data.nqe_baseline;
    state.nqe_ema = data.nqe_ema;
    state.qe_ratio = (data.nqe_baseline > 0.0) ? (data.nqe_ema / data.nqe_baseline) : 1.0;
    state.ntotal = data.ntotal;
    return state;
  }

  Result<std::vector<uint8_t>> GetDocPQCode(const VersionSet& route_versions,
                                            DocId doc_id) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    for (const auto& list : data.lists) {
      for (const auto& entry : list) {
        if (entry.doc_id == doc_id) {
          if (entry.pq_code.empty()) {
            return Status::NotFound("doc has no pq code");
          }
          return entry.pq_code;
        }
      }
    }
    return Status::NotFound("doc_id not found");
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }

  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

 private:
  Status ValidateRecordsForInsertLocked(const IndexData& data,
                                        const AlignedVector<VectorRecord>& recs) const {
    std::unordered_set<DocId> batch_ids;
    batch_ids.reserve(recs.size());
    for (const auto& rec : recs) {
      if (static_cast<uint32_t>(rec.x.size()) != data.dim) {
        return Status::InvalidArgument("Record dim mismatch");
      }
      if (data.doc_ids.find(rec.doc_id) != data.doc_ids.end()) {
        return Status::AlreadyExists("doc_id already present in IVF");
      }
      if (!batch_ids.insert(rec.doc_id).second) {
        return Status::AlreadyExists("duplicate doc_id in batch");
      }
    }
    return Status::OK();
  }

  void CommitPendingLocked(IndexData* data,
                           const std::vector<int>& centroids,
                           AlignedVector<ListEntry>* entries) {
    for (size_t i = 0; i < entries->size(); ++i) {
      data->doc_ids.insert((*entries)[i].doc_id);
      data->lists[static_cast<size_t>(centroids[i])].push_back(std::move((*entries)[i]));
      ++data->ntotal;
    }
  }

  mutable std::shared_mutex mu_;
  std::unordered_map<VersionId, std::unique_ptr<IndexData>> data_map_;
  VersionId next_version_{1};
  VersionId latest_version_{0};
};

}  // namespace

std::shared_ptr<IVFIndex> CreateIVFIndex() { return std::make_shared<KMeansIVFIndex>(); }

}  // namespace ann
