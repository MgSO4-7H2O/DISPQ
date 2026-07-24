#include "index/ivf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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

#include "common/timer.h"
#include "soa_kernels.h"

namespace ann {
namespace {

constexpr uint32_t kDefaultKMeansIterations = 20;
constexpr uint32_t kDefaultSeed = 42;
constexpr double kDefaultNQEEps = 1e-6;
constexpr size_t kPrecomputedTableMaxBytes = (static_cast<size_t>(2) << 30);  // 2GB
constexpr size_t kKMeansPartialMaxBytes = static_cast<size_t>(256) << 20;
constexpr uint32_t kInvalidListId = std::numeric_limits<uint32_t>::max();

struct DocLocation {
  uint32_t list_id{kInvalidListId};
  uint32_t offset{0};
};

class PagedDocLocationMap {
 public:
  bool Get(DocId doc_id, DocLocation* location) const {
    const auto it = pages_.find(doc_id >> kPageBits);
    if (it == pages_.end()) {
      return false;
    }
    const DocLocation& value = it->second->values[doc_id & kPageMask];
    if (value.list_id == kInvalidListId) {
      return false;
    }
    if (location != nullptr) {
      *location = value;
    }
    return true;
  }

  bool Contains(DocId doc_id) const { return Get(doc_id, nullptr); }

  void Set(DocId doc_id, DocLocation location) {
    const uint32_t page_id = doc_id >> kPageBits;
    auto [it, inserted] = pages_.try_emplace(page_id);
    if (inserted) {
      it->second = std::make_unique<Page>();
    }
    DocLocation& value = it->second->values[doc_id & kPageMask];
    if (value.list_id == kInvalidListId) {
      it->second->live++;
      size_++;
    }
    value = location;
  }

  void Erase(DocId doc_id) {
    const uint32_t page_id = doc_id >> kPageBits;
    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
      return;
    }
    DocLocation& value = it->second->values[doc_id & kPageMask];
    if (value.list_id == kInvalidListId) {
      return;
    }
    value = DocLocation{};
    it->second->live--;
    size_--;
    if (it->second->live == 0) {
      pages_.erase(it);
    }
  }

  void Clear() {
    pages_.clear();
    size_ = 0;
  }

  size_t size() const { return size_; }

 private:
  static constexpr uint32_t kPageBits = 8;
  static constexpr uint32_t kPageSize = 1u << kPageBits;
  static constexpr uint32_t kPageMask = kPageSize - 1;

  struct Page {
    Page() {
      for (DocLocation& value : values) {
        value.list_id = kInvalidListId;
      }
    }
    std::array<DocLocation, kPageSize> values;
    uint32_t live{0};
  };

  std::unordered_map<uint32_t, std::unique_ptr<Page>> pages_;
  size_t size_{0};
};

struct ListEntry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DocId doc_id;
  VersionSet versions;
  Eigen::VectorXf vector;
  float norm{0.0f};
  std::vector<uint8_t> pq_code;
};

struct ListPQCodesSoA {
  size_t rows{0};
  size_t stride{0};
  std::vector<uint8_t> codes;
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
  // PQ layouts are uniform; sub_offsets is retained for the existing hot paths.
  uint32_t dsub{0};
  std::vector<uint32_t> pq_sub_offsets;
  MatrixRM routing_centroids;
  Eigen::VectorXf routing_centroid_norms;
  std::vector<MatrixRM> pq_codebooks;
  std::vector<MatrixRM> pq_codebooks_soa;
  std::vector<std::vector<uint8_t>> pq_codes_by_list;
  std::vector<ListPQCodesSoA> pq_codes_soa_by_list;
  bool use_precomputed_table{false};
  std::vector<float> pq_precomputed_table;
  std::vector<std::vector<uint64_t>> pq_counts;
  double nqe_baseline{0.0};
  double nqe_ema{0.0};
  bool defer_pq_stats_to_add{false};
  double deferred_nqe_sum{0.0};
  uint64_t deferred_nqe_count{0};
  uint64_t online_pq_batch_count{0};
  double warmup_nqe_sum{0.0};
  uint32_t warmup_seen_batches{0};
  std::vector<AlignedVector<ListEntry>> lists;
  PagedDocLocationMap doc_locations;
  uint64_t ntotal{0};
  double last_patch_pq_reencode_ms{0.0};
};

std::vector<uint32_t> MakeUniformPQSubOffsets(uint32_t dim, uint32_t M) {
  std::vector<uint32_t> offsets(static_cast<size_t>(M) + 1, 0);
  const uint32_t dsub = dim / M;
  for (uint32_t m = 0; m < M; ++m) {
    offsets[static_cast<size_t>(m) + 1] = offsets[static_cast<size_t>(m)] + dsub;
  }
  return offsets;
}

bool HasValidPQSubspaces(const IndexData& data) {
  // Layouts are constructed once by MakeBalancedPQSubOffsets and never mutated.
  return data.use_pq && data.M > 0 && data.M <= data.dim &&
         data.pq_sub_offsets.size() == static_cast<size_t>(data.M) + 1 &&
         data.pq_sub_offsets.front() == 0 && data.pq_sub_offsets.back() == data.dim;
}

inline uint32_t PQSubOffset(const IndexData& data, uint32_t m) {
  return data.pq_sub_offsets[static_cast<size_t>(m)];
}

inline uint32_t PQSubDim(const IndexData& data, uint32_t m) {
  return data.pq_sub_offsets[static_cast<size_t>(m) + 1] -
         data.pq_sub_offsets[static_cast<size_t>(m)];
}

void RebuildPQSoACache(IndexData* data) {
  data->pq_codebooks_soa.clear();
  if (!HasValidPQSubspaces(*data) || data->Ks == 0 ||
      data->pq_codebooks.size() != data->M) {
    return;
  }
  data->pq_codebooks_soa.resize(data->M);
  for (uint32_t m = 0; m < data->M; ++m) {
    const MatrixRM& aos = data->pq_codebooks[static_cast<size_t>(m)];
    const uint32_t subdim = PQSubDim(*data, m);
    if (aos.rows() != static_cast<Eigen::Index>(data->Ks) ||
        aos.cols() != static_cast<Eigen::Index>(subdim)) {
      data->pq_codebooks_soa.clear();
      return;
    }
    MatrixRM soa(subdim, data->Ks);
    for (Eigen::Index k = 0; k < aos.rows(); ++k) {
      for (Eigen::Index d = 0; d < aos.cols(); ++d) {
        soa(d, k) = aos(k, d);
      }
    }
    data->pq_codebooks_soa[static_cast<size_t>(m)] = std::move(soa);
  }
}

inline float SquaredL2FromNormDot(float a_norm, float b_norm, float dot) {
  const float dist = a_norm + b_norm - 2.0f * dot;
  return (dist >= 0.0f) ? dist : 0.0f;
}

int NearestCentroid(Eigen::Ref<const Eigen::VectorXf> vec, const IndexData& data) {
  const float vec_norm = vec.squaredNorm();
  const bool has_routing_norms =
      data.routing_centroid_norms.size() == data.routing_centroids.rows();
  float best = std::numeric_limits<float>::max();
  int best_idx = 0;
  for (int i = 0; i < data.routing_centroids.rows(); ++i) {
    const float dot = data.routing_centroids.row(i).dot(vec);
    const float centroid_norm =
        has_routing_norms ? data.routing_centroid_norms(i)
                          : data.routing_centroids.row(i).squaredNorm();
    const float dist = SquaredL2FromNormDot(centroid_norm, vec_norm, dot);
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

uint32_t NearestCodeword(Eigen::Ref<const Eigen::VectorXf> sub,
                         const MatrixRM& codebook,
                         const MatrixRM* codebook_soa,
                         float* best_dist_out) {
  if (codebook_soa != nullptr && codebook.cols() == sub.size() &&
      codebook_soa->rows() == sub.size() &&
      codebook_soa->cols() == codebook.rows()) {
    return internal::AssignPQSoA(
        sub.data(), static_cast<uint32_t>(sub.size()), codebook_soa->data(),
        static_cast<uint32_t>(codebook.rows()), best_dist_out);
  }
  return NearestCodeword(sub, codebook, best_dist_out);
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

void AssignKMeansPQSoA(Eigen::Ref<const MatrixRM> X,
                       const MatrixRM& centroids_soa,
                       std::vector<int>* assignments,
                       bool use_parallel,
                       int num_threads) {
#ifndef _OPENMP
  (void)use_parallel;
  (void)num_threads;
#endif
#ifdef _OPENMP
#pragma omp parallel for if (use_parallel) num_threads(num_threads) schedule(static)
#endif
  for (int64_t i = 0; i < X.rows(); ++i) {
    (*assignments)[static_cast<size_t>(i)] = static_cast<int>(internal::AssignPQSoA(
        X.row(i).data(), static_cast<uint32_t>(X.cols()), centroids_soa.data(),
        static_cast<uint32_t>(centroids_soa.cols()), nullptr));
  }
}

enum class KMeansMode {
  kGeneric,
  kPQ,
};

void RunKMeans(Eigen::Ref<const MatrixRM> X,
               MatrixRM* centroids,
               uint32_t iterations,
               KMeansMode mode = KMeansMode::kGeneric) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  const int64_t k = centroids->rows();
  std::vector<int> assignments(num_vecs, 0);

  const uint32_t kmeans_iters = (iterations == 0) ? kDefaultKMeansIterations : iterations;
  bool use_parallel = false;
  int num_threads = 1;
#ifdef _OPENMP
  use_parallel = !omp_in_parallel();
  const size_t partial_bytes_per_thread =
      static_cast<size_t>(k) * (static_cast<size_t>(dim) * sizeof(float) + sizeof(int64_t));
  const int memory_limited_threads = partial_bytes_per_thread == 0
                                         ? 1
                                         : static_cast<int>(std::max<size_t>(
                                               1, kKMeansPartialMaxBytes /
                                                      partial_bytes_per_thread));
  num_threads = use_parallel
                    ? std::max(1, std::min(omp_get_max_threads(), memory_limited_threads))
                    : 1;
#endif
  for (uint32_t iter = 0; iter < kmeans_iters; ++iter) {
#if defined(__AVX512F__)
    if (mode == KMeansMode::kPQ) {
      MatrixRM centroids_soa(dim, k);
      for (Eigen::Index centroid = 0; centroid < centroids->rows(); ++centroid) {
        for (Eigen::Index d = 0; d < centroids->cols(); ++d) {
          centroids_soa(d, centroid) = (*centroids)(centroid, d);
        }
      }
      AssignKMeansPQSoA(X, centroids_soa, &assignments, use_parallel, num_threads);
    } else
#endif
    {
#ifdef _OPENMP
#pragma omp parallel for if (use_parallel) num_threads(num_threads) schedule(static)
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
    }

    // Update step.
    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int64_t> counts(static_cast<size_t>(k), 0);
#ifdef _OPENMP
    std::vector<MatrixRM> partial_sums(static_cast<size_t>(num_threads), MatrixRM::Zero(k, dim));
    std::vector<std::vector<int64_t>> partial_counts(
        static_cast<size_t>(num_threads), std::vector<int64_t>(static_cast<size_t>(k), 0));

#pragma omp parallel if (use_parallel) num_threads(num_threads)
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

size_t PrecomputedTableElementCount(const IndexData& data) {
  return static_cast<size_t>(data.nlist) * static_cast<size_t>(data.M) * static_cast<size_t>(data.Ks);
}

void BuildPrecomputedTable(IndexData* data) {
  if (data == nullptr) {
    return;
  }
  data->use_precomputed_table = false;
  data->pq_precomputed_table.clear();
  if (!data->use_pq || !data->pq_residual || data->Ks == 0 ||
      !HasValidPQSubspaces(*data)) {
    return;
  }
  if (data->pq_codebooks.size() != data->M || data->routing_centroids.rows() != data->nlist) {
    return;
  }

  const size_t elem_count = PrecomputedTableElementCount(*data);
  if (elem_count == 0) {
    return;
  }
  const size_t bytes = elem_count * sizeof(float);
  if (bytes > kPrecomputedTableMaxBytes) {
    return;
  }

  data->pq_precomputed_table.resize(elem_count);
  for (uint32_t list_id = 0; list_id < data->nlist; ++list_id) {
    for (uint32_t m = 0; m < data->M; ++m) {
      const MatrixRM& codebook = data->pq_codebooks[static_cast<size_t>(m)];
      const uint32_t subdim = PQSubDim(*data, m);
      if (codebook.rows() != static_cast<Eigen::Index>(data->Ks) ||
          codebook.cols() != static_cast<Eigen::Index>(subdim)) {
        data->pq_precomputed_table.clear();
        return;
      }
      Eigen::Map<const Eigen::VectorXf> centroid_sub(
          data->routing_centroids.row(static_cast<Eigen::Index>(list_id)).data() +
              static_cast<Eigen::Index>(PQSubOffset(*data, m)),
          static_cast<Eigen::Index>(subdim));
      for (uint32_t k = 0; k < data->Ks; ++k) {
        Eigen::Map<const Eigen::VectorXf> z(
            codebook.row(static_cast<Eigen::Index>(k)).data(),
            static_cast<Eigen::Index>(subdim));
        const float precomputed = z.squaredNorm() + 2.0f * centroid_sub.dot(z);
        const size_t idx = (static_cast<size_t>(list_id) * data->M + m) * data->Ks + k;
        data->pq_precomputed_table[idx] = precomputed;
      }
    }
  }
  data->use_precomputed_table = true;
}

float AccumulateDistanceFastScan(const float* distance_table,
                                 const uint8_t* code_ptr,
                                 uint32_t M,
                                 uint32_t Ks) {
  float sum = 0.0f;
  uint32_t m = 0;
  const float* table_ptr = distance_table;
  for (; m + 3 < M; m += 4) {
    sum += table_ptr[code_ptr[0]];
    sum += table_ptr[Ks + code_ptr[1]];
    sum += table_ptr[2 * Ks + code_ptr[2]];
    sum += table_ptr[3 * Ks + code_ptr[3]];
    code_ptr += 4;
    table_ptr += 4 * Ks;
  }
  for (; m < M; ++m) {
    sum += table_ptr[*code_ptr++];
    table_ptr += Ks;
  }
  return sum;
}

bool HasContiguousPQCodesForList(const IndexData& data, uint32_t list_id) {
  if (!data.use_pq || data.M == 0) {
    return false;
  }
  if (list_id >= data.lists.size() || list_id >= data.pq_codes_by_list.size()) {
    return false;
  }
  const auto& flat = data.pq_codes_by_list[static_cast<size_t>(list_id)];
  return flat.size() == data.lists[static_cast<size_t>(list_id)].size() * data.M;
}

bool HasPQCodebookSoA(const IndexData& data) {
  if (!HasValidPQSubspaces(data) || data.Ks == 0 ||
      data.pq_codebooks_soa.size() != data.M) {
    return false;
  }
  for (uint32_t m = 0; m < data.M; ++m) {
    const MatrixRM& soa = data.pq_codebooks_soa[static_cast<size_t>(m)];
    if (soa.rows() != static_cast<Eigen::Index>(PQSubDim(data, m)) ||
        soa.cols() != static_cast<Eigen::Index>(data.Ks)) {
      return false;
    }
  }
  return true;
}

bool HasSoAPQCodesForList(const IndexData& data, uint32_t list_id) {
  if (!data.use_pq || data.M == 0) {
    return false;
  }
  if (list_id >= data.lists.size() || list_id >= data.pq_codes_soa_by_list.size()) {
    return false;
  }
  const auto& soa = data.pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  const size_t rows = data.lists[static_cast<size_t>(list_id)].size();
  return soa.rows == rows && soa.stride >= rows &&
         soa.codes.size() == static_cast<size_t>(data.M) * soa.stride;
}

void ClearListPQCodesSoA(IndexData* data, uint32_t list_id) {
  if (data == nullptr || list_id >= data->pq_codes_soa_by_list.size()) {
    return;
  }
  data->pq_codes_soa_by_list[static_cast<size_t>(list_id)] = ListPQCodesSoA{};
}

void ReserveListPQCodesSoA(IndexData* data, uint32_t list_id, size_t rows_needed) {
  if (data == nullptr || !data->use_pq || data->M == 0 || list_id >= data->lists.size()) {
    return;
  }
  if (data->pq_codes_soa_by_list.size() != data->lists.size()) {
    data->pq_codes_soa_by_list.resize(data->lists.size());
  }
  auto& soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  if (soa.stride >= rows_needed &&
      soa.codes.size() == static_cast<size_t>(data->M) * soa.stride) {
    return;
  }

  const bool can_preserve = soa.rows <= soa.stride &&
                            soa.codes.size() == static_cast<size_t>(data->M) * soa.stride;
  const size_t rows_to_copy = can_preserve ? soa.rows : 0;
  const size_t doubled = soa.stride == 0 ? 0 : soa.stride * 2;
  const size_t new_stride = std::max(rows_needed, std::max<size_t>(doubled, 8));
  std::vector<uint8_t> next(static_cast<size_t>(data->M) * new_stride, 0);
  if (rows_to_copy > 0) {
    for (uint32_t m = 0; m < data->M; ++m) {
      std::copy_n(soa.codes.data() + static_cast<size_t>(m) * soa.stride,
                  rows_to_copy,
                  next.data() + static_cast<size_t>(m) * new_stride);
    }
  }
  soa.rows = rows_to_copy;
  soa.stride = new_stride;
  soa.codes = std::move(next);
}

void RebuildListPQCodes(IndexData* data, uint32_t list_id) {
  if (data == nullptr || !data->use_pq || data->M == 0) {
    return;
  }
  if (list_id >= data->lists.size()) {
    return;
  }
  if (data->pq_codes_by_list.size() != data->lists.size()) {
    data->pq_codes_by_list.resize(data->lists.size());
  }
  if (data->pq_codes_soa_by_list.size() != data->lists.size()) {
    data->pq_codes_soa_by_list.resize(data->lists.size());
  }
  const auto& list = data->lists[static_cast<size_t>(list_id)];
  auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
  auto& soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  flat.clear();
  flat.reserve(list.size() * data->M);
  soa.rows = list.size();
  soa.stride = list.size();
  soa.codes.assign(static_cast<size_t>(data->M) * soa.stride, 0);
  for (size_t row = 0; row < list.size(); ++row) {
    const auto& entry = list[row];
    if (entry.pq_code.size() != data->M) {
      flat.clear();
      ClearListPQCodesSoA(data, list_id);
      return;
    }
    flat.insert(flat.end(), entry.pq_code.begin(), entry.pq_code.end());
    for (uint32_t m = 0; m < data->M; ++m) {
      soa.codes[static_cast<size_t>(m) * soa.stride + row] =
          entry.pq_code[static_cast<size_t>(m)];
    }
  }
}

void AppendListPQCodeSoA(IndexData* data,
                         uint32_t list_id,
                         size_t rows_before,
                         const std::vector<uint8_t>& code) {
  if (data == nullptr || !data->use_pq || data->M == 0 || list_id >= data->lists.size()) {
    return;
  }
  if (code.size() != data->M) {
    ClearListPQCodesSoA(data, list_id);
    return;
  }
  if (data->pq_codes_soa_by_list.size() != data->lists.size()) {
    data->pq_codes_soa_by_list.resize(data->lists.size());
  }
  if (data->lists[static_cast<size_t>(list_id)].size() != rows_before + 1) {
    RebuildListPQCodes(data, list_id);
    return;
  }

  auto& soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  const bool appendable = soa.rows == rows_before && soa.stride >= rows_before &&
                          soa.codes.size() == static_cast<size_t>(data->M) * soa.stride;
  if (!appendable) {
    RebuildListPQCodes(data, list_id);
    return;
  }
  ReserveListPQCodesSoA(data, list_id, rows_before + 1);
  auto& reserved = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  if (reserved.rows != rows_before || reserved.stride < rows_before + 1 ||
      reserved.codes.size() != static_cast<size_t>(data->M) * reserved.stride) {
    RebuildListPQCodes(data, list_id);
    return;
  }
  for (uint32_t m = 0; m < data->M; ++m) {
    reserved.codes[static_cast<size_t>(m) * reserved.stride + rows_before] =
        code[static_cast<size_t>(m)];
  }
  reserved.rows = rows_before + 1;
}

void SwapEraseListPQCode(IndexData* data, uint32_t list_id, size_t pos) {
  if (data == nullptr || !data->use_pq || data->M == 0) {
    return;
  }
  if (!HasContiguousPQCodesForList(*data, list_id)) {
    RebuildListPQCodes(data, list_id);
  }
  if (!HasSoAPQCodesForList(*data, list_id)) {
    RebuildListPQCodes(data, list_id);
  }
  if (list_id >= data->pq_codes_by_list.size() ||
      list_id >= data->pq_codes_soa_by_list.size()) {
    return;
  }
  auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
  if (flat.size() % data->M != 0) {
    RebuildListPQCodes(data, list_id);
  }
  size_t rows = flat.size() / data->M;
  if (pos >= rows) {
    RebuildListPQCodes(data, list_id);
    rows = flat.size() / data->M;
    if (flat.size() % data->M != 0 || pos >= rows) {
      return;
    }
  }
  auto& soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  if (!HasSoAPQCodesForList(*data, list_id) || pos >= soa.rows) {
    RebuildListPQCodes(data, list_id);
  }
  if (!HasSoAPQCodesForList(*data, list_id) ||
      pos >= data->pq_codes_soa_by_list[static_cast<size_t>(list_id)].rows) {
    return;
  }
  const size_t last = rows - 1;
  if (pos != last) {
    std::copy_n(flat.data() + last * data->M,
                data->M,
                flat.data() + pos * data->M);
  }
  flat.resize(last * data->M);

  auto& updated_soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  for (uint32_t m = 0; m < data->M; ++m) {
    const size_t offset = static_cast<size_t>(m) * updated_soa.stride;
    if (pos != last) {
      updated_soa.codes[offset + pos] = updated_soa.codes[offset + last];
    }
  }
  updated_soa.rows = last;
}

SearchResult SearchSingleQuery(const IndexData& data,
                               Eigen::Ref<const Eigen::VectorXf> qw,
                               uint32_t topk,
                               uint32_t probes,
                               uint8_t from_new,
                               bool collect_scan_trace) {
  auto heap_cmp = [](const Candidate& a, const Candidate& b) {
    return a.approx_dist < b.approx_dist;
  };

  const bool use_pq = data.use_pq && data.Ks > 0 && HasValidPQSubspaces(data) &&
                      data.pq_codebooks.size() == data.M;
  const bool use_precomputed_table =
      use_pq && data.use_precomputed_table &&
      data.pq_precomputed_table.size() == PrecomputedTableElementCount(data);
  const bool use_fast_scan = use_pq && data.M >= 4;
  const bool use_codebook_soa = use_pq && HasPQCodebookSoA(data);
  const bool use_vectorized_code_soa_scan =
      use_pq && internal::HasVectorizedPQCodeSoAScan();
  const bool has_routing_norms =
      data.routing_centroid_norms.size() == data.routing_centroids.rows();
  const float qw_norm = qw.squaredNorm();
  std::vector<std::pair<float, uint32_t>> centroid_dists(data.nlist);
  for (uint32_t i = 0; i < data.nlist; ++i) {
    const float dot = data.routing_centroids.row(static_cast<Eigen::Index>(i)).dot(qw);
    const float centroid_norm =
        has_routing_norms ? data.routing_centroid_norms(static_cast<Eigen::Index>(i))
                          : data.routing_centroids.row(static_cast<Eigen::Index>(i)).squaredNorm();
    const float dist = SquaredL2FromNormDot(centroid_norm, qw_norm, dot);
    centroid_dists[static_cast<size_t>(i)] = {dist, i};
  }
  std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + probes, centroid_dists.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });

  std::vector<Candidate> heap;
  heap.reserve(topk);
  uint64_t scanned = 0;
  std::vector<DocId> scanned_doc_ids;
  std::vector<float> scanned_approx_dists;
  const float qnorm = use_pq ? 0.0f : qw_norm;
  std::vector<float> distance_table;
  std::vector<float> query_term3_table;
  if (use_pq) {
    distance_table.resize(static_cast<size_t>(data.M) * data.Ks);
    if (use_precomputed_table) {
      query_term3_table.resize(static_cast<size_t>(data.M) * data.Ks, 0.0f);
      for (uint32_t m = 0; m < data.M; ++m) {
        const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
        const uint32_t subdim = PQSubDim(data, m);
        Eigen::Map<const Eigen::VectorXf> qsub(
            qw.data() + static_cast<Eigen::Index>(PQSubOffset(data, m)),
            static_cast<Eigen::Index>(subdim));
        float* table_ptr = query_term3_table.data() + static_cast<size_t>(m) * data.Ks;
        if (use_codebook_soa) {
          internal::BuildPQNeg2DotTableSoA(qsub.data(),
                                           subdim,
                                           data.pq_codebooks_soa[static_cast<size_t>(m)].data(),
                                           data.Ks,
                                           table_ptr);
        } else {
          for (uint32_t k = 0; k < data.Ks; ++k) {
            const float qz = qsub.dot(codebook.row(static_cast<Eigen::Index>(k)).transpose());
            table_ptr[k] = -2.0f * qz;
          }
        }
      }
    }
  }
  std::vector<float> soa_scan_dists;
  for (uint32_t pi = 0; pi < probes; ++pi) {
    const uint32_t list_id = centroid_dists[static_cast<size_t>(pi)].second;
    const float coarse_dist = centroid_dists[static_cast<size_t>(pi)].first;
    if (use_pq) {
      if (use_precomputed_table) {
        const size_t table_offset =
            static_cast<size_t>(list_id) * static_cast<size_t>(data.M) * static_cast<size_t>(data.Ks);
        for (size_t i = 0; i < distance_table.size(); ++i) {
          distance_table[i] = data.pq_precomputed_table[table_offset + i] + query_term3_table[i];
        }
      } else {
        Eigen::VectorXf qres = qw;
        if (data.pq_residual) {
          qres -= data.routing_centroids.row(static_cast<Eigen::Index>(list_id)).transpose();
        }
        for (uint32_t m = 0; m < data.M; ++m) {
          const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
          const uint32_t subdim = PQSubDim(data, m);
          Eigen::Map<const Eigen::VectorXf> qsub(
              qres.data() + static_cast<Eigen::Index>(PQSubOffset(data, m)),
              static_cast<Eigen::Index>(subdim));
          float* table_ptr = distance_table.data() + static_cast<size_t>(m) * data.Ks;
          if (use_codebook_soa) {
            internal::BuildPQDistanceTableSoA(qsub.data(),
                                              subdim,
                                              data.pq_codebooks_soa[static_cast<size_t>(m)].data(),
                                              data.Ks,
                                              table_ptr);
          } else {
            for (uint32_t k = 0; k < data.Ks; ++k) {
              const float dist =
                  (qsub - codebook.row(static_cast<Eigen::Index>(k)).transpose()).squaredNorm();
              table_ptr[k] = dist;
            }
          }
        }
      }
    }
    const auto& list = data.lists[static_cast<size_t>(list_id)];
    const bool use_contiguous_codes = use_pq && HasContiguousPQCodesForList(data, list_id);
    const uint8_t* contiguous_codes =
        use_contiguous_codes ? data.pq_codes_by_list[static_cast<size_t>(list_id)].data() : nullptr;
    const bool use_soa_codes =
        use_vectorized_code_soa_scan && HasSoAPQCodesForList(data, list_id);
    const ListPQCodesSoA* soa_codes =
        use_soa_codes ? &data.pq_codes_soa_by_list[static_cast<size_t>(list_id)] : nullptr;
    if (soa_codes != nullptr && !list.empty()) {
      soa_scan_dists.resize(list.size());
      internal::ScanPQCodesSoA(distance_table.data(),
                               soa_codes->codes.data(),
                               data.M,
                               data.Ks,
                               soa_codes->stride,
                               list.size(),
                               soa_scan_dists.data());
    }
    for (size_t li = 0; li < list.size(); ++li) {
      const auto& entry = list[li];
      ++scanned;
      Candidate cand;
      cand.doc_id = entry.doc_id;
      if (use_pq) {
        float approx = 0.0f;
        if (soa_codes != nullptr) {
          approx = soa_scan_dists[li];
        } else {
          const uint8_t* code_ptr = nullptr;
          if (use_contiguous_codes) {
            code_ptr = contiguous_codes + li * data.M;
          } else if (entry.pq_code.size() == data.M) {
            code_ptr = entry.pq_code.data();
          }
          if (code_ptr == nullptr) {
            continue;
          }
          approx = use_fast_scan
                       ? AccumulateDistanceFastScan(distance_table.data(), code_ptr, data.M, data.Ks)
                       : 0.0f;
          if (!use_fast_scan) {
            for (uint32_t m = 0; m < data.M; ++m) {
              const uint8_t code = code_ptr[m];
              approx += distance_table[static_cast<size_t>(m) * data.Ks + code];
            }
          }
        }
        if (use_precomputed_table) {
          approx += coarse_dist;
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
      if (collect_scan_trace) {
        scanned_doc_ids.push_back(cand.doc_id);
        scanned_approx_dists.push_back(cand.approx_dist);
      }
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

  SearchResult result;
  result.scanned_candidates = scanned;
  if (!heap.empty()) {
    std::sort(heap.begin(), heap.end(),
              [](const Candidate& a, const Candidate& b) { return a.approx_dist < b.approx_dist; });
    result.topk = std::move(heap);
  }
  if (collect_scan_trace) {
    result.scanned_doc_ids = std::move(scanned_doc_ids);
    result.scanned_approx_dists = std::move(scanned_approx_dists);
  }
  return result;
}

class KMeansIVFIndex : public IVFIndex {
  struct RemovedDoc;

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
      if (p.pq.M > dim) {
        return Status::InvalidArgument("PQ M must not exceed dim");
      }
      if (dim % p.pq.M != 0) {
        return Status::InvalidArgument("PQ requires dim divisible by M");
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
    data->routing_centroid_norms = data->routing_centroids.rowwise().squaredNorm();
    data->use_pq = p.pq.enable;
    data->pq_residual = p.pq.residual;
    if (data->use_pq) {
      data->M = p.pq.M;
      data->nbits = p.pq.nbits;
      data->Ks = 1u << data->nbits;
      data->dsub = dim / data->M;
      data->pq_sub_offsets = MakeUniformPQSubOffsets(dim, data->M);
    } else {
      data->M = 0;
      data->nbits = 8;
      data->Ks = 0;
      data->dsub = 0;
      data->pq_sub_offsets.clear();
      data->pq_codebooks.clear();
      data->pq_counts.clear();
    }
    data->lists.clear();
    data->lists.resize(nlist);
    data->pq_codes_by_list.clear();
    data->pq_codes_soa_by_list.clear();
    if (data->use_pq) {
      data->pq_codes_by_list.resize(nlist);
      data->pq_codes_soa_by_list.resize(nlist);
    }
    data->doc_locations.Clear();
    data->ntotal = 0;

    if (data->use_pq) {
      data->pq_codebooks.resize(data->M);
      if (p.use_fixed_pq_codebooks) {
        if (p.fixed_pq_codebooks.size() != data->M) {
          return Status::InvalidArgument("fixed_pq_codebooks size mismatch");
        }
        for (uint32_t m = 0; m < data->M; ++m) {
          const MatrixRM& book = p.fixed_pq_codebooks[static_cast<size_t>(m)];
          const uint32_t subdim = PQSubDim(*data, m);
          if (book.rows() != static_cast<Eigen::Index>(data->Ks) ||
              book.cols() != static_cast<Eigen::Index>(subdim)) {
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
#pragma omp parallel for schedule(static)
#endif
        for (uint32_t m = 0; m < data->M; ++m) {
          const Eigen::Index offset = static_cast<Eigen::Index>(PQSubOffset(*data, m));
          const Eigen::Index subdim = static_cast<Eigen::Index>(PQSubDim(*data, m));
          MatrixRM sub = residuals.block(0, offset, residuals.rows(), subdim);
          MatrixRM codebook = InitializeCentroids(sub, data->Ks);
          RunKMeans(sub, &codebook, p.kmeans_iterations, KMeansMode::kPQ);
          data->pq_codebooks[static_cast<size_t>(m)] = std::move(codebook);
        }
      }

      RebuildPQSoACache(data.get());

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

      const bool can_defer_stats = p.defer_pq_stats_to_add && !use_fixed_counts &&
                                   p.fixed_pq_baseline_nqe <= 0.0 &&
                                   p.fixed_pq_ema_nqe <= 0.0;
      if (can_defer_stats) {
        data->pq_counts.assign(data->M, std::vector<uint64_t>(data->Ks, 0));
        data->nqe_baseline = 0.0;
        data->nqe_ema = 0.0;
        data->defer_pq_stats_to_add = true;
      } else {
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
            const uint32_t subdim = PQSubDim(*data, m);
            Eigen::Map<const Eigen::VectorXf> sub(
                residual.data() + static_cast<Eigen::Index>(PQSubOffset(*data, m)),
                static_cast<Eigen::Index>(subdim));
            const MatrixRM* codebook_soa =
                data->pq_codebooks_soa.size() == data->M
                    ? &data->pq_codebooks_soa[static_cast<size_t>(m)]
                    : nullptr;
            float best_dist = 0.0f;
            const uint32_t k = NearestCodeword(
                sub, data->pq_codebooks[static_cast<size_t>(m)], codebook_soa, &best_dist);
            computed_counts[static_cast<size_t>(m)][static_cast<size_t>(k)]++;
            err2 += static_cast<double>(best_dist);
            r2 += static_cast<double>(sub.squaredNorm());
          }
          nqe_sum += err2 / (r2 + kDefaultNQEEps);
        }
        const double computed_baseline = nqe_sum / static_cast<double>(Xw.rows());
        data->pq_counts = use_fixed_counts ? p.fixed_pq_counts : computed_counts;
        data->nqe_baseline =
            (p.fixed_pq_baseline_nqe > 0.0) ? p.fixed_pq_baseline_nqe : computed_baseline;
        data->nqe_ema =
            (p.fixed_pq_ema_nqe > 0.0) ? p.fixed_pq_ema_nqe : data->nqe_baseline;
      }
      BuildPrecomputedTable(data.get());
    }

    std::unique_lock lock(mu_);
    const VersionId version = (index_version == 0) ? next_version_++ : index_version;
    data->version = version;
    data_map_[version] = std::move(data);
    latest_version_ = version;
    return version;
  }

  Status Add(const AlignedVector<VectorRecord>& recs) override {
    return AddBatch(recs, true);
  }

  Status AddBatch(const AlignedVector<VectorRecord>& recs,
                  bool finalize_deferred_stats) override {
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
    const bool collect_initial_pq_stats = data.use_pq && data.defer_pq_stats_to_add;
    std::vector<double> per_record_nqe;
    if (collect_initial_pq_stats) {
      per_record_nqe.assign(recs.size(), 0.0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
      const size_t i = static_cast<size_t>(ii);
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
        double err2 = 0.0;
        double r2 = 0.0;
        for (uint32_t m = 0; m < data.M; ++m) {
          const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
          const uint32_t subdim = PQSubDim(data, m);
          Eigen::Map<const Eigen::VectorXf> sub(
              residual.data() + static_cast<Eigen::Index>(PQSubOffset(data, m)),
              static_cast<Eigen::Index>(subdim));
          const MatrixRM* codebook_soa =
              data.pq_codebooks_soa.size() == data.M
                  ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                  : nullptr;
          float best_dist = 0.0f;
          const uint32_t k = collect_initial_pq_stats
                                 ? NearestCodeword(sub, codebook, codebook_soa, &best_dist)
                                 : NearestCodeword(sub, codebook, codebook_soa, nullptr);
          entry.pq_code[static_cast<size_t>(m)] = static_cast<uint8_t>(k);
          if (collect_initial_pq_stats) {
            err2 += static_cast<double>(best_dist);
            r2 += static_cast<double>(sub.squaredNorm());
          }
        }
        if (collect_initial_pq_stats) {
          per_record_nqe[i] = err2 / (r2 + kDefaultNQEEps);
        }
      } else {
        entry.vector = rec.x;
        entry.norm = entry.vector.squaredNorm();
      }
      centroids[i] = centroid;
      entries[i] = std::move(entry);
    }

    if (collect_initial_pq_stats) {
      for (size_t i = 0; i < entries.size(); ++i) {
        data.deferred_nqe_sum += per_record_nqe[i];
        data.deferred_nqe_count++;
        for (uint32_t m = 0; m < data.M; ++m) {
          const uint32_t k = entries[i].pq_code[static_cast<size_t>(m)];
          data.pq_counts[static_cast<size_t>(m)][static_cast<size_t>(k)]++;
        }
      }
      if (finalize_deferred_stats && data.deferred_nqe_count > 0) {
        data.nqe_baseline =
            data.deferred_nqe_sum / static_cast<double>(data.deferred_nqe_count);
        data.nqe_ema = data.nqe_baseline;
        data.defer_pq_stats_to_add = false;
        data.deferred_nqe_sum = 0.0;
        data.deferred_nqe_count = 0;
      }
    }

    CommitPendingLocked(&data, centroids, &entries);
    return Status::OK();
  }

  Status RemoveDocIds(const VersionSet& route_versions,
                      const std::vector<DocId>& doc_ids) override {
    if (doc_ids.empty()) {
      return Status::OK();
    }
    std::unique_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("RemoveDocIds: index version not built");
    }
    IndexData& data = *it->second;
    Status validate = ValidateDeleteIdsLocked(data, doc_ids);
    if (!validate.ok()) {
      return validate;
    }
    return RemoveDocsCompactLocked(&data, doc_ids);
  }

  Result<OnlinePQUpdateStats> AddWithOnlinePQ(
      const AlignedVector<VectorRecord>& recs,
      const OnlinePQUpdateOptions& options) override {
    return AddWithOnlinePQSlidingWindow(recs, std::vector<DocId>{}, options);
  }

  Result<OnlinePQUpdateStats> AddWithOnlinePQSlidingWindow(
      const AlignedVector<VectorRecord>& recs,
      const std::vector<DocId>& delete_doc_ids,
      const OnlinePQUpdateOptions& options) override {
    return AddWithOnlinePQSlidingWindowImpl(recs, delete_doc_ids, nullptr, options);
  }

  Result<OnlinePQUpdateStats> AddWithOnlinePQSlidingWindowRecords(
      const AlignedVector<VectorRecord>& recs,
      const AlignedVector<VectorRecord>& delete_recs,
      const OnlinePQUpdateOptions& options) override {
    std::vector<DocId> delete_doc_ids;
    delete_doc_ids.reserve(delete_recs.size());
    for (const auto& rec : delete_recs) {
      delete_doc_ids.push_back(rec.doc_id);
    }
    return AddWithOnlinePQSlidingWindowImpl(recs, delete_doc_ids, &delete_recs, options);
  }

  Result<OnlinePQUpdateStats> AddWithOnlinePQSlidingWindowImpl(
      const AlignedVector<VectorRecord>& recs,
      const std::vector<DocId>& delete_doc_ids,
      const AlignedVector<VectorRecord>* delete_recs,
      const OnlinePQUpdateOptions& options) {
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
    stats.deleted_vectors = static_cast<uint32_t>(delete_doc_ids.size());

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
    Status validate_deletes = ValidateDeleteIdsLocked(data, delete_doc_ids);
    if (!validate_deletes.ok()) {
      return validate_deletes;
    }
    std::unordered_map<DocId, size_t> delete_rec_pos;
    if (delete_recs != nullptr) {
      if (delete_recs->size() != delete_doc_ids.size()) {
        return Status::InvalidArgument("delete record/doc_id size mismatch");
      }
      delete_rec_pos.reserve(delete_recs->size() * 2 + 1);
      for (size_t i = 0; i < delete_recs->size(); ++i) {
        const auto& rec = (*delete_recs)[i];
        if (static_cast<uint32_t>(rec.x.size()) != data.dim) {
          return Status::InvalidArgument("delete record dim mismatch");
        }
        if (!delete_rec_pos.emplace(rec.doc_id, i).second) {
          return Status::InvalidArgument("duplicate delete record doc_id");
        }
      }
      for (DocId doc_id : delete_doc_ids) {
        if (delete_rec_pos.find(doc_id) == delete_rec_pos.end()) {
          return Status::InvalidArgument("delete doc_id missing external vector");
        }
      }
    }
    if (recs.empty() && delete_doc_ids.empty()) {
      stats.use_online_pq = data.use_pq && data.pq_residual;
      stats.nqe_baseline = data.nqe_baseline;
      stats.nqe_ema = data.nqe_ema;
      stats.qe_ratio = (data.nqe_baseline > 0.0) ? (data.nqe_ema / data.nqe_baseline) : 1.0;
      return stats;
    }

    const bool can_online = data.use_pq && data.pq_residual && data.Ks > 0 &&
                            HasValidPQSubspaces(data) &&
                            data.pq_codebooks.size() == data.M &&
                            data.pq_counts.size() == data.M;
    if (!can_online) {
      if (!delete_doc_ids.empty()) {
        Timer delete_timer;
        Status remove_status = RemoveDocsLocked(&data, delete_doc_ids, nullptr);
        if (!remove_status.ok()) {
          return remove_status;
        }
        stats.delete_ms += delete_timer.ElapsedMillis();
      }
      std::vector<int> centroids(recs.size(), 0);
      AlignedVector<ListEntry> entries(recs.size());
      Timer encode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
        const size_t i = static_cast<size_t>(ii);
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
            const uint32_t subdim = PQSubDim(data, m);
            Eigen::Map<const Eigen::VectorXf> sub(
                residual.data() + static_cast<Eigen::Index>(PQSubOffset(data, m)),
                static_cast<Eigen::Index>(subdim));
            const MatrixRM* codebook_soa =
                data.pq_codebooks_soa.size() == data.M
                    ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                    : nullptr;
            entry.pq_code[static_cast<size_t>(m)] =
                static_cast<uint8_t>(NearestCodeword(sub, codebook, codebook_soa, nullptr));
          }
        } else {
          entry.vector = rec.x;
          entry.norm = entry.vector.squaredNorm();
        }
        centroids[i] = centroid;
        entries[i] = std::move(entry);
      }
      stats.insert_encode_ms += encode_timer.ElapsedMillis();
      Timer commit_timer;
      CommitPendingLocked(&data, centroids, &entries);
      stats.insert_commit_ms += commit_timer.ElapsedMillis();
      stats.insert_ms += stats.insert_encode_ms + stats.insert_commit_ms;
      stats.maintenance_ms += stats.delete_ms;
      return stats;
    }

    stats.use_online_pq = true;
    const double eps = std::max(1e-12, options.nqe_eps);
    const double ema_alpha = options.ema_alpha;
    const uint32_t n = static_cast<uint32_t>(recs.size());

    std::vector<int> centroids(recs.size(), 0);
    MatrixRM residuals(static_cast<Eigen::Index>(recs.size()), data.dim);
    std::vector<uint8_t> codes_before(recs.size() * static_cast<size_t>(data.M), 0);
    std::vector<float> best_dists(recs.size() * static_cast<size_t>(data.M), 0.0f);
    std::vector<float> sub_energies(recs.size() * static_cast<size_t>(data.M), 0.0f);
    std::vector<double> nqe_by_record(recs.size(), 0.0);
    std::vector<std::vector<uint32_t>> batch_cnt(
        data.M, std::vector<uint32_t>(data.Ks, 0));
    std::vector<MatrixRM> sum_vec;
    std::vector<MatrixRM> sum_err;
    sum_vec.reserve(data.M);
    sum_err.reserve(data.M);
    for (uint32_t m = 0; m < data.M; ++m) {
      const uint32_t subdim = PQSubDim(data, m);
      sum_vec.emplace_back(MatrixRM::Zero(data.Ks, subdim));
      sum_err.emplace_back(MatrixRM::Zero(data.Ks, subdim));
    }
    std::vector<double> sub_err_sum(data.M, 0.0);
    std::vector<double> sub_energy_sum(data.M, 0.0);
    std::vector<std::vector<uint32_t>> delete_cnt(
        data.M, std::vector<uint32_t>(data.Ks, 0));
    std::vector<MatrixRM> delete_sum_vec;
    delete_sum_vec.reserve(data.M);
    for (uint32_t m = 0; m < data.M; ++m) {
      delete_sum_vec.emplace_back(MatrixRM::Zero(data.Ks, PQSubDim(data, m)));
    }

    std::vector<RemovedDoc> removed_docs;
    if (!delete_doc_ids.empty()) {
      Timer delete_timer;
      Status remove_status = RemoveDocsLocked(&data, delete_doc_ids, &removed_docs);
      if (!remove_status.ok()) {
        return remove_status;
      }
      for (const auto& removed : removed_docs) {
        const Eigen::VectorXf* removed_vector = nullptr;
        if (static_cast<uint32_t>(removed.entry.vector.size()) == data.dim) {
          removed_vector = &removed.entry.vector;
        } else if (delete_recs != nullptr) {
          auto vec_it = delete_rec_pos.find(removed.doc_id);
          if (vec_it != delete_rec_pos.end()) {
            removed_vector = &(*delete_recs)[vec_it->second].x;
          }
        }
        if (removed.entry.pq_code.size() != data.M || removed_vector == nullptr) {
          continue;
        }
        Eigen::VectorXf residual = *removed_vector;
        if (data.pq_residual) {
          residual -= data.routing_centroids
                          .row(static_cast<Eigen::Index>(removed.list_id))
                          .transpose();
        }
        for (uint32_t m = 0; m < data.M; ++m) {
          const uint32_t k = static_cast<uint32_t>(removed.entry.pq_code[static_cast<size_t>(m)]);
          if (k >= data.Ks) {
            continue;
          }
          delete_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)]++;
          const uint32_t subdim = PQSubDim(data, m);
          Eigen::Map<const Eigen::VectorXf> sub(
              residual.data() + static_cast<Eigen::Index>(PQSubOffset(data, m)),
              static_cast<Eigen::Index>(subdim));
          delete_sum_vec[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(k)) +=
              sub.transpose();
        }
      }
      stats.delete_ms += delete_timer.ElapsedMillis();
    }

    Timer encode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& rec = recs[i];
      const int centroid = NearestCentroid(rec.x, data);
      centroids[i] = centroid;

      residuals.row(static_cast<Eigen::Index>(i)) = rec.x.transpose();
      if (data.pq_residual) {
        residuals.row(static_cast<Eigen::Index>(i)) -= data.routing_centroids.row(centroid);
      }

      double err2 = 0.0;
      double r2 = 0.0;
      const float* residual_ptr = residuals.row(static_cast<Eigen::Index>(i)).data();
      for (uint32_t m = 0; m < data.M; ++m) {
        const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
        const uint32_t subdim = PQSubDim(data, m);
        Eigen::Map<const Eigen::VectorXf> sub(
            residual_ptr + static_cast<Eigen::Index>(PQSubOffset(data, m)),
            static_cast<Eigen::Index>(subdim));
        const MatrixRM* codebook_soa =
            data.pq_codebooks_soa.size() == data.M
                ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                : nullptr;
        float best_dist = 0.0f;
        const uint32_t best_idx =
            NearestCodeword(sub, codebook, codebook_soa, &best_dist);
        const size_t offset = i * static_cast<size_t>(data.M) + m;
        codes_before[offset] = static_cast<uint8_t>(best_idx);
        best_dists[offset] = best_dist;
        sub_energies[offset] = sub.squaredNorm();
        err2 += static_cast<double>(best_dist);
        r2 += static_cast<double>(sub_energies[offset]);
      }
      nqe_by_record[i] = err2 / (r2 + eps);
    }
    stats.insert_encode_ms += encode_timer.ElapsedMillis();

    Timer onlinepq_stats_timer;
    double nqe_sum = 0.0;
    for (size_t i = 0; i < recs.size(); ++i) {
      const float* residual_ptr = residuals.row(static_cast<Eigen::Index>(i)).data();
      for (uint32_t m = 0; m < data.M; ++m) {
        const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
        const size_t offset = i * static_cast<size_t>(data.M) + m;
        const uint32_t best_idx = codes_before[offset];
        const uint32_t subdim = PQSubDim(data, m);
        Eigen::Map<const Eigen::VectorXf> sub(
            residual_ptr + static_cast<Eigen::Index>(PQSubOffset(data, m)), subdim);
        batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(best_idx)]++;
        sum_vec[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(best_idx)) +=
            sub.transpose();
        sum_err[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(best_idx)) +=
            (sub - codebook.row(static_cast<Eigen::Index>(best_idx)).transpose()).transpose();
        sub_err_sum[static_cast<size_t>(m)] += best_dists[offset];
        sub_energy_sum[static_cast<size_t>(m)] += sub_energies[offset];
      }
      nqe_sum += nqe_by_record[i];
    }
    stats.nqe_batch = (n > 0) ? (nqe_sum / static_cast<double>(n)) : data.nqe_ema;
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
      const bool force_periodic_update =
          options.force_update_interval > 0 &&
          (data.online_pq_batch_count % options.force_update_interval == 0);
      // Keep qe_ratio for diagnostics, but update is only controlled by force interval.
      stats.trigger_update = options.enable && force_periodic_update;
      stats.in_warmup = false;
      stats.warmup_batches_left = 0;
    }
    stats.onlinepq_stats_ms += onlinepq_stats_timer.ElapsedMillis();

    std::vector<uint8_t> codes_for_insert = codes_before;
    if (stats.trigger_update) {
      Timer codebook_update_timer;
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
            const uint32_t ins = batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)];
            const uint32_t del = delete_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)];
            if (ins == 0 && del == 0) {
              continue;
            }
            const double score_add =
                sum_err[static_cast<size_t>(m)]
                    .row(static_cast<Eigen::Index>(k))
                    .squaredNorm();
            const double score_del =
                delete_sum_vec[static_cast<size_t>(m)]
                    .row(static_cast<Eigen::Index>(k))
                    .squaredNorm();
            const double score = score_add + score_del;
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
            if (batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)] > 0 ||
                delete_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)] > 0) {
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
          const uint32_t ins = batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)];
          const uint32_t del = delete_cnt[static_cast<size_t>(m)][static_cast<size_t>(k)];
          if (ins == 0 && del == 0) {
            continue;
          }

          const uint64_t old_n = data.pq_counts[static_cast<size_t>(m)][static_cast<size_t>(k)];
          const int64_t new_n_i64 =
              static_cast<int64_t>(old_n) + static_cast<int64_t>(ins) - static_cast<int64_t>(del);
          const uint64_t new_n = (new_n_i64 <= 0) ? 0ull : static_cast<uint64_t>(new_n_i64);
          const Eigen::VectorXf z_old =
              data.pq_codebooks[static_cast<size_t>(m)]
                  .row(static_cast<Eigen::Index>(k))
                  .transpose();
          Eigen::VectorXf numerator = static_cast<float>(old_n) * z_old;
          if (ins > 0) {
            const Eigen::VectorXf mu_add =
                sum_vec[static_cast<size_t>(m)]
                    .row(static_cast<Eigen::Index>(k))
                    .transpose() /
                static_cast<float>(ins);
            numerator += static_cast<float>(ins) * mu_add;
          }
          if (del > 0) {
            const Eigen::VectorXf mu_del =
                delete_sum_vec[static_cast<size_t>(m)]
                    .row(static_cast<Eigen::Index>(k))
                    .transpose() /
                static_cast<float>(del);
            numerator -= static_cast<float>(del) * mu_del;
          }
          const Eigen::VectorXf z_new =
              (new_n > 0) ? (numerator / static_cast<float>(new_n)) : z_old;
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
      if (stats.updated_codebook) {
        RebuildPQSoACache(&data);
        BuildPrecomputedTable(&data);
      }
      stats.codebook_update_ms += codebook_update_timer.ElapsedMillis();

      if (stats.updated_codebook && options.reencode_batch_after_update) {
        Timer reencode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
          const size_t i = static_cast<size_t>(ii);
          const float* residual_ptr = residuals.row(static_cast<Eigen::Index>(i)).data();
          for (uint32_t m = 0; m < data.M; ++m) {
            const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
            const uint32_t subdim = PQSubDim(data, m);
            Eigen::Map<const Eigen::VectorXf> sub(
                residual_ptr + static_cast<Eigen::Index>(PQSubOffset(data, m)),
                static_cast<Eigen::Index>(subdim));
            const MatrixRM* codebook_soa =
                data.pq_codebooks_soa.size() == data.M
                    ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                    : nullptr;
            codes_for_insert[i * static_cast<size_t>(data.M) + m] =
                static_cast<uint8_t>(NearestCodeword(sub, codebook, codebook_soa, nullptr));
          }
        }
        stats.reencode_ms += reencode_timer.ElapsedMillis();
        stats.reencoded_batch = true;
      }
    }

    AlignedVector<ListEntry> entries(recs.size());
    Timer entry_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& rec = recs[i];
      ListEntry entry;
      entry.doc_id = rec.doc_id;
      entry.versions = rec.versions;
      entry.versions.index_version = data.version;
      const auto first = codes_for_insert.begin() + i * static_cast<size_t>(data.M);
      entry.pq_code.assign(first, first + data.M);
      entries[i] = std::move(entry);
    }
    stats.insert_entry_ms += entry_timer.ElapsedMillis();
    Timer commit_timer;
    CommitPendingLocked(&data, centroids, &entries);
    stats.insert_commit_ms += commit_timer.ElapsedMillis();
    stats.insert_ms += stats.insert_encode_ms + stats.insert_entry_ms + stats.insert_commit_ms;
    stats.maintenance_ms += stats.delete_ms + stats.onlinepq_stats_ms +
                            stats.codebook_update_ms + stats.reencode_ms;
    return stats;
  }

  Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                              uint32_t topk,
                              uint32_t nprobe,
                              const VersionSet& route_versions,
                              uint8_t from_new,
                              bool collect_scan_trace) const override {
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
    return SearchSingleQuery(data, qw, topk, probes, from_new, collect_scan_trace);
  }

  Result<std::vector<SearchResult>> SearchBatch(
      Eigen::Ref<const MatrixRM> qw_batch,
      uint32_t topk,
      uint32_t nprobe,
      const VersionSet& route_versions,
      uint8_t from_new,
      bool collect_scan_trace) const override {
    if (topk == 0) {
      return Status::InvalidArgument("topk must be positive");
    }
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    if (qw_batch.cols() != static_cast<Eigen::Index>(data.dim)) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    std::vector<SearchResult> out(static_cast<size_t>(qw_batch.rows()));
    if (qw_batch.rows() == 0) {
      return out;
    }
    const uint32_t probes = std::max<uint32_t>(1, std::min<uint32_t>(nprobe, data.nlist));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t qi = 0; qi < static_cast<int64_t>(qw_batch.rows()); ++qi) {
      Eigen::VectorXf q = qw_batch.row(static_cast<Eigen::Index>(qi)).transpose();
      out[static_cast<size_t>(qi)] =
          SearchSingleQuery(data, q, topk, probes, from_new, collect_scan_trace);
    }
    return out;
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
    state.sub_offsets = data.pq_sub_offsets;
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
    DocLocation location;
    if (!data.doc_locations.Get(doc_id, &location) || location.list_id >= data.lists.size()) {
      return Status::NotFound("doc_id not found");
    }
    const auto& list = data.lists[static_cast<size_t>(location.list_id)];
    if (location.offset >= list.size() || list[location.offset].doc_id != doc_id) {
      return Status::Internal("doc location is inconsistent");
    }
    if (list[location.offset].pq_code.empty()) {
      return Status::NotFound("doc has no pq code");
    }
    return list[location.offset].pq_code;
  }

  Result<std::vector<DocId>> SnapshotDocIds(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    std::vector<DocId> out;
    out.reserve(static_cast<size_t>(data.ntotal));
    for (const auto& list : data.lists) {
      for (const auto& entry : list) {
        out.push_back(entry.doc_id);
      }
    }
    if (out.size() != data.ntotal || data.doc_locations.size() != data.ntotal) {
      return Status::Internal("SnapshotDocIds: live index accounting mismatch");
    }
    return out;
  }

  Result<AlignedVector<VectorRecord>> SnapshotRecords(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    if (data.use_pq) {
      return Status::InvalidArgument("SnapshotRecords: full vectors are not stored for PQ IVF");
    }
    AlignedVector<VectorRecord> out;
    out.reserve(static_cast<size_t>(data.ntotal));
    for (uint32_t list_id = 0; list_id < data.nlist; ++list_id) {
      const auto& list = data.lists[static_cast<size_t>(list_id)];
      for (const auto& entry : list) {
        VectorRecord rec;
        rec.doc_id = entry.doc_id;
        rec.dim = data.dim;
        rec.versions = entry.versions;
        rec.versions.index_version = data.version;
        rec.ivf_id = list_id;
        rec.x = entry.vector;
        out.push_back(std::move(rec));
      }
    }
    return out;
  }

  Result<AlignedVector<VectorRecord>> GetPartitionRecords(
      const VersionSet& route_versions,
      uint32_t partition_id) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    if (partition_id >= data.nlist) {
      return Status::InvalidArgument("partition_id out of range");
    }
    if (data.use_pq) {
      return Status::InvalidArgument("GetPartitionRecords: full vectors are not stored for PQ IVF");
    }
    const auto& list = data.lists[static_cast<size_t>(partition_id)];
    AlignedVector<VectorRecord> out;
    out.reserve(list.size());
    for (const auto& entry : list) {
      VectorRecord rec;
      rec.doc_id = entry.doc_id;
      rec.dim = data.dim;
      rec.versions = entry.versions;
      rec.versions.index_version = data.version;
      rec.ivf_id = partition_id;
      rec.x = entry.vector;
      out.push_back(std::move(rec));
    }
    return out;
  }

  Result<std::vector<DocId>> GetPartitionDocIds(
      const VersionSet& route_versions,
      uint32_t partition_id) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    if (partition_id >= data.nlist) {
      return Status::InvalidArgument("partition_id out of range");
    }
    const auto& list = data.lists[static_cast<size_t>(partition_id)];
    std::vector<DocId> out;
    out.reserve(list.size());
    for (const auto& entry : list) {
      out.push_back(entry.doc_id);
    }
    return out;
  }

  Result<std::vector<uint32_t>> GetPartitionSizes(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    std::vector<uint32_t> sizes(data.nlist, 0);
    for (uint32_t list_id = 0; list_id < data.nlist; ++list_id) {
      sizes[static_cast<size_t>(list_id)] =
          static_cast<uint32_t>(data.lists[static_cast<size_t>(list_id)].size());
    }
    return sizes;
  }

  Status CommitPartitionPatch(const VersionSet& route_versions,
                              const PartitionPatch& patch) override {
    if (patch.partition_ids.size() != patch.replacement_records.size()) {
      return Status::InvalidArgument(
          "CommitPartitionPatch: partition_ids size mismatch with replacement_records");
    }

    std::unique_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    IndexData& data = *it->second;
    data.last_patch_pq_reencode_ms = 0.0;
    std::unordered_set<uint32_t> patch_partitions;
    patch_partitions.reserve(patch.partition_ids.size());
    for (uint32_t partition_id : patch.partition_ids) {
      if (partition_id >= data.nlist) {
        return Status::InvalidArgument("CommitPartitionPatch: partition_id out of range");
      }
      if (!patch_partitions.insert(partition_id).second) {
        return Status::InvalidArgument("CommitPartitionPatch: duplicate partition_id");
      }
    }

    std::unordered_set<DocId> patch_doc_ids;
    size_t replacement_total = 0;
    for (const auto& records : patch.replacement_records) {
      replacement_total += records.size();
    }
    patch_doc_ids.reserve(replacement_total * 2 + 1);
    for (size_t i = 0; i < patch.partition_ids.size(); ++i) {
      const auto& records = patch.replacement_records[i];
      for (const auto& rec : records) {
        if (static_cast<uint32_t>(rec.x.size()) != data.dim) {
          return Status::InvalidArgument("CommitPartitionPatch: record dim mismatch");
        }
        if (!patch_doc_ids.insert(rec.doc_id).second) {
          return Status::AlreadyExists("CommitPartitionPatch: duplicate doc_id in patch");
        }
        DocLocation existing;
        if (data.doc_locations.Get(rec.doc_id, &existing) &&
            patch_partitions.find(existing.list_id) == patch_partitions.end()) {
          return Status::AlreadyExists(
              "CommitPartitionPatch: doc_id collides with unaffected partitions");
        }
      }
    }

    uint64_t old_patch_total = 0;
    for (uint32_t partition_id : patch.partition_ids) {
      const auto& list = data.lists[static_cast<size_t>(partition_id)];
      old_patch_total += list.size();
      for (const auto& entry : list) {
        data.doc_locations.Erase(entry.doc_id);
      }
    }

    double pq_reencode_ms = 0.0;
    for (size_t i = 0; i < patch.partition_ids.size(); ++i) {
      const uint32_t partition_id = patch.partition_ids[i];
      const auto& records = patch.replacement_records[i];
      auto& dst = data.lists[static_cast<size_t>(partition_id)];
      dst.resize(records.size());
      Timer pq_reencode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int64_t jj = 0; jj < static_cast<int64_t>(records.size()); ++jj) {
        const auto& rec = records[static_cast<size_t>(jj)];
        ListEntry entry;
        entry.doc_id = rec.doc_id;
        entry.versions = rec.versions;
        entry.versions.index_version = data.version;
        if (data.use_pq) {
          entry.pq_code.resize(data.M);
          Eigen::VectorXf residual = rec.x;
          if (data.pq_residual) {
            residual -=
                data.routing_centroids.row(static_cast<Eigen::Index>(partition_id)).transpose();
          }
          for (uint32_t m = 0; m < data.M; ++m) {
            const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
            const uint32_t subdim = PQSubDim(data, m);
            Eigen::Map<const Eigen::VectorXf> sub(
                residual.data() + static_cast<Eigen::Index>(PQSubOffset(data, m)),
                static_cast<Eigen::Index>(subdim));
            const MatrixRM* codebook_soa =
                data.pq_codebooks_soa.size() == data.M
                    ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                    : nullptr;
            entry.pq_code[static_cast<size_t>(m)] =
                static_cast<uint8_t>(NearestCodeword(sub, codebook, codebook_soa, nullptr));
          }
        } else {
          entry.vector = rec.x;
          entry.norm = entry.vector.squaredNorm();
        }
        dst[static_cast<size_t>(jj)] = std::move(entry);
      }
      if (data.use_pq) {
        pq_reencode_ms += pq_reencode_timer.ElapsedMillis();
        RebuildListPQCodes(&data, partition_id);
      }
      for (size_t offset = 0; offset < dst.size(); ++offset) {
        data.doc_locations.Set(dst[offset].doc_id,
                               DocLocation{partition_id, static_cast<uint32_t>(offset)});
      }
    }
    if (old_patch_total > data.ntotal) {
      return Status::Internal("CommitPartitionPatch: patched rows exceed ntotal");
    }
    data.ntotal = data.ntotal - old_patch_total + replacement_total;
    if (data.doc_locations.size() != data.ntotal) {
      return Status::Internal("CommitPartitionPatch: location count mismatch");
    }
    data.last_patch_pq_reencode_ms = pq_reencode_ms;
    return Status::OK();
  }

  Result<double> GetLastPatchPQReencodeMs(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    return it->second->last_patch_pq_reencode_ms;
  }

  Result<std::vector<uint8_t>> Serialize() const override { return std::vector<uint8_t>{}; }

  Status Deserialize(const std::vector<uint8_t>&) override { return Status::OK(); }

 private:
  struct RemovedDoc {
    DocId doc_id{0};
    uint32_t list_id{0};
    ListEntry entry;
  };

  Status ValidateRecordsForInsertLocked(const IndexData& data,
                                        const AlignedVector<VectorRecord>& recs) const {
    std::unordered_set<DocId> batch_ids;
    batch_ids.reserve(recs.size());
    for (const auto& rec : recs) {
      if (static_cast<uint32_t>(rec.x.size()) != data.dim) {
        return Status::InvalidArgument("Record dim mismatch");
      }
      if (data.doc_locations.Contains(rec.doc_id)) {
        return Status::AlreadyExists("doc_id already present in IVF");
      }
      if (!batch_ids.insert(rec.doc_id).second) {
        return Status::AlreadyExists("duplicate doc_id in batch");
      }
    }
    return Status::OK();
  }

  Status ValidateDeleteIdsLocked(const IndexData& data,
                                 const std::vector<DocId>& doc_ids) const {
    std::unordered_set<DocId> seen;
    seen.reserve(doc_ids.size());
    for (DocId doc_id : doc_ids) {
      if (!seen.insert(doc_id).second) {
        return Status::InvalidArgument("duplicate delete doc_id in batch");
      }
      if (!data.doc_locations.Contains(doc_id)) {
        return Status::NotFound("delete doc_id not found in IVF");
      }
    }
    return Status::OK();
  }

  Status RemoveDocsCompactLocked(IndexData* data,
                                const std::vector<DocId>& doc_ids) {
    if (data == nullptr) {
      return Status::InvalidArgument("RemoveDocsCompactLocked: null data");
    }
    if (doc_ids.empty()) {
      return Status::OK();
    }

    std::vector<std::pair<uint32_t, DocId>> deletes_by_list;
    deletes_by_list.reserve(doc_ids.size());
    for (DocId doc_id : doc_ids) {
      DocLocation location;
      if (!data->doc_locations.Get(doc_id, &location)) {
        return Status::NotFound("delete doc_id list mapping not found");
      }
      const uint32_t list_id = location.list_id;
      if (list_id >= data->lists.size()) {
        return Status::InvalidArgument("delete list_id out of range");
      }
      deletes_by_list.emplace_back(list_id, doc_id);
    }
    std::sort(deletes_by_list.begin(), deletes_by_list.end());

    uint64_t removed_total = 0;
    for (size_t group_begin = 0; group_begin < deletes_by_list.size();) {
      const uint32_t list_id = deletes_by_list[group_begin].first;
      size_t group_end = group_begin + 1;
      while (group_end < deletes_by_list.size() &&
             deletes_by_list[group_end].first == list_id) {
        ++group_end;
      }
      std::unordered_set<DocId> delete_set;
      delete_set.reserve((group_end - group_begin) * 2 + 1);
      for (size_t i = group_begin; i < group_end; ++i) {
        delete_set.insert(deletes_by_list[i].second);
      }

      auto& list = data->lists[static_cast<size_t>(list_id)];
      size_t write = 0;
      size_t removed_in_list = 0;
      for (size_t read = 0; read < list.size(); ++read) {
        const DocId doc_id = list[read].doc_id;
        if (delete_set.find(doc_id) != delete_set.end()) {
          data->doc_locations.Erase(doc_id);
          ++removed_in_list;
          continue;
        }
        if (write != read) {
          list[write] = std::move(list[read]);
        }
        data->doc_locations.Set(
            list[write].doc_id, DocLocation{list_id, static_cast<uint32_t>(write)});
        ++write;
      }
      if (removed_in_list != group_end - group_begin) {
        return Status::NotFound("delete doc_id not found in list");
      }
      list.resize(write);
      if (list.capacity() > list.size() * 2 + 1024) {
        list.shrink_to_fit();
      }
      if (data->use_pq) {
        RebuildListPQCodes(data, list_id);
        if (list_id < data->pq_codes_by_list.size()) {
          auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
          if (flat.capacity() > flat.size() * 2 + 1024) {
            flat.shrink_to_fit();
          }
        }
      }
      removed_total += removed_in_list;
      group_begin = group_end;
    }

    if (removed_total > data->ntotal) {
      return Status::InvalidArgument("delete count exceeds ntotal");
    }
    data->ntotal -= removed_total;
    if (data->doc_locations.size() != data->ntotal) {
      return Status::Internal("compact delete location count mismatch");
    }
    return Status::OK();
  }

  Status RemoveDocsLocked(IndexData* data,
                          const std::vector<DocId>& doc_ids,
                          std::vector<RemovedDoc>* removed) {
    if (data == nullptr) {
      return Status::InvalidArgument("RemoveDocsLocked: null data");
    }
    if (removed != nullptr) {
      removed->clear();
      removed->reserve(doc_ids.size());
    }
    for (DocId doc_id : doc_ids) {
      DocLocation location;
      if (!data->doc_locations.Get(doc_id, &location)) {
        return Status::NotFound("delete doc_id list mapping not found");
      }
      const uint32_t list_id = location.list_id;
      if (list_id >= data->lists.size()) {
        return Status::InvalidArgument("delete list_id out of range");
      }
      auto& list = data->lists[static_cast<size_t>(list_id)];
      const size_t pos = location.offset;
      if (pos >= list.size() || list[pos].doc_id != doc_id) {
        return Status::Internal("delete doc location is inconsistent");
      }
      if (removed != nullptr) {
        RemovedDoc r;
        r.doc_id = doc_id;
        r.list_id = list_id;
        r.entry = std::move(list[pos]);
        removed->push_back(std::move(r));
      }
      if (data->use_pq) {
        SwapEraseListPQCode(data, list_id, pos);
      }
      if (pos != list.size() - 1) {
        list[pos] = std::move(list.back());
        data->doc_locations.Set(
            list[pos].doc_id, DocLocation{list_id, static_cast<uint32_t>(pos)});
      }
      list.pop_back();
      data->doc_locations.Erase(doc_id);
      if (data->ntotal > 0) {
        --data->ntotal;
      }
    }
    if (data->doc_locations.size() != data->ntotal) {
      return Status::Internal("delete location count mismatch");
    }
    return Status::OK();
  }

  void CommitPendingLocked(IndexData* data,
                           const std::vector<int>& centroids,
                           AlignedVector<ListEntry>* entries) {
    if (data->use_pq && data->pq_codes_by_list.size() != data->lists.size()) {
      data->pq_codes_by_list.resize(data->lists.size());
    }
    if (data->use_pq && data->pq_codes_soa_by_list.size() != data->lists.size()) {
      data->pq_codes_soa_by_list.resize(data->lists.size());
    }
    for (size_t i = 0; i < entries->size(); ++i) {
      const uint32_t list_id = static_cast<uint32_t>(centroids[i]);
      auto& list = data->lists[static_cast<size_t>(list_id)];
      const size_t rows_before_push = list.size();
      list.push_back(std::move((*entries)[i]));
      data->doc_locations.Set(
          list.back().doc_id,
          DocLocation{list_id, static_cast<uint32_t>(list.size() - 1)});
      if (data->use_pq) {
        auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
        if (list.back().pq_code.size() == data->M) {
          const size_t expected_before_push = rows_before_push * data->M;
          if (flat.size() == expected_before_push) {
            flat.insert(flat.end(), list.back().pq_code.begin(), list.back().pq_code.end());
            AppendListPQCodeSoA(data, list_id, rows_before_push, list.back().pq_code);
          } else {
            RebuildListPQCodes(data, list_id);
          }
        } else {
          flat.clear();
          ClearListPQCodesSoA(data, list_id);
        }
      }
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
