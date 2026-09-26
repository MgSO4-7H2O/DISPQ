#include "index/ivf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

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
constexpr uint32_t kPQTrainingConcurrency = 1;
constexpr uint32_t kInvalidListId = std::numeric_limits<uint32_t>::max();
constexpr size_t kDenseDocCompactMinGaps = 65536;
constexpr uint32_t kIVFSerializeMagic = 0x49564637;  // 'IVF7'
constexpr uint32_t kIVFSerializeFormatVersion = 1;

void AppendRaw(std::vector<uint8_t>* bytes, const void* data, size_t len) {
  if (len == 0) {
    return;
  }
  const auto* ptr = reinterpret_cast<const uint8_t*>(data);
  bytes->insert(bytes->end(), ptr, ptr + len);
}

template <typename T>
void AppendPod(std::vector<uint8_t>* bytes, const T& value) {
  AppendRaw(bytes, &value, sizeof(T));
}

void AppendBool(std::vector<uint8_t>* bytes, bool value) {
  const uint8_t stored = value ? 1 : 0;
  AppendPod(bytes, stored);
}

template <typename T>
Status ReadPod(const std::vector<uint8_t>& bytes, size_t* offset, T* value) {
  if (offset == nullptr || value == nullptr || *offset > bytes.size() ||
      bytes.size() - *offset < sizeof(T)) {
    return Status::InvalidArgument("IVF payload truncated");
  }
  std::memcpy(value, bytes.data() + *offset, sizeof(T));
  *offset += sizeof(T);
  return Status::OK();
}

Status ReadBool(const std::vector<uint8_t>& bytes, size_t* offset, bool* value) {
  uint8_t stored = 0;
  Status status = ReadPod(bytes, offset, &stored);
  if (!status.ok()) {
    return status;
  }
  if (stored > 1) {
    return Status::InvalidArgument("Invalid bool in IVF payload");
  }
  *value = (stored != 0);
  return Status::OK();
}

Status ReadRaw(const std::vector<uint8_t>& bytes, size_t* offset, void* dst, size_t len) {
  if (len == 0) {
    return Status::OK();
  }
  if (offset == nullptr || dst == nullptr || *offset > bytes.size() ||
      bytes.size() - *offset < len) {
    return Status::InvalidArgument("IVF payload truncated");
  }
  std::memcpy(dst, bytes.data() + *offset, len);
  *offset += len;
  return Status::OK();
}

void AppendMatrix(std::vector<uint8_t>* bytes, const MatrixRM& matrix) {
  const uint32_t rows = static_cast<uint32_t>(matrix.rows());
  const uint32_t cols = static_cast<uint32_t>(matrix.cols());
  AppendPod(bytes, rows);
  AppendPod(bytes, cols);
  AppendRaw(bytes, matrix.data(), sizeof(float) * static_cast<size_t>(matrix.size()));
}

Result<MatrixRM> ReadMatrix(const std::vector<uint8_t>& bytes, size_t* offset) {
  uint32_t rows = 0;
  uint32_t cols = 0;
  Status status = ReadPod(bytes, offset, &rows);
  if (!status.ok()) {
    return status;
  }
  status = ReadPod(bytes, offset, &cols);
  if (!status.ok()) {
    return status;
  }
  const uint64_t count = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
  if (count > static_cast<uint64_t>(std::numeric_limits<Eigen::Index>::max())) {
    return Status::InvalidArgument("Matrix in IVF payload is too large");
  }
  MatrixRM matrix(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  status = ReadRaw(bytes, offset, matrix.data(), sizeof(float) * static_cast<size_t>(count));
  if (!status.ok()) {
    return status;
  }
  return matrix;
}

void AppendFloatVector(std::vector<uint8_t>* bytes, const Eigen::VectorXf& vec) {
  const uint32_t size = static_cast<uint32_t>(vec.size());
  AppendPod(bytes, size);
  AppendRaw(bytes, vec.data(), sizeof(float) * static_cast<size_t>(size));
}

Result<Eigen::VectorXf> ReadFloatVector(const std::vector<uint8_t>& bytes, size_t* offset) {
  uint32_t size = 0;
  Status status = ReadPod(bytes, offset, &size);
  if (!status.ok()) {
    return status;
  }
  Eigen::VectorXf vec(static_cast<Eigen::Index>(size));
  status = ReadRaw(bytes, offset, vec.data(), sizeof(float) * static_cast<size_t>(size));
  if (!status.ok()) {
    return status;
  }
  return vec;
}

void AppendByteVector(std::vector<uint8_t>* bytes, const std::vector<uint8_t>& values) {
  const uint64_t size = static_cast<uint64_t>(values.size());
  AppendPod(bytes, size);
  AppendRaw(bytes, values.data(), static_cast<size_t>(size));
}

Result<std::vector<uint8_t>> ReadByteVector(const std::vector<uint8_t>& bytes,
                                            size_t* offset) {
  uint64_t size = 0;
  Status status = ReadPod(bytes, offset, &size);
  if (!status.ok()) {
    return status;
  }
  if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return Status::InvalidArgument("Byte vector in IVF payload is too large");
  }
  std::vector<uint8_t> values(static_cast<size_t>(size));
  status = ReadRaw(bytes, offset, values.data(), values.size());
  if (!status.ok()) {
    return status;
  }
  return values;
}

void AppendUint64Vector(std::vector<uint8_t>* bytes, const std::vector<uint64_t>& values) {
  const uint32_t size = static_cast<uint32_t>(values.size());
  AppendPod(bytes, size);
  AppendRaw(bytes, values.data(), sizeof(uint64_t) * values.size());
}

Result<std::vector<uint64_t>> ReadUint64Vector(const std::vector<uint8_t>& bytes,
                                               size_t* offset) {
  uint32_t size = 0;
  Status status = ReadPod(bytes, offset, &size);
  if (!status.ok()) {
    return status;
  }
  std::vector<uint64_t> values(static_cast<size_t>(size));
  status = ReadRaw(bytes, offset, values.data(), sizeof(uint64_t) * values.size());
  if (!status.ok()) {
    return status;
  }
  return values;
}

struct DenseDocListMap {
  DocId base{0};
  std::vector<uint32_t> list_ids;
  size_t leading_invalid{0};

  bool Get(DocId doc_id, uint32_t* list_id) const {
    if (list_ids.empty() || doc_id < base) {
      return false;
    }
    const size_t offset = static_cast<size_t>(doc_id - base);
    if (offset >= list_ids.size() || list_ids[offset] == kInvalidListId) {
      return false;
    }
    if (list_id != nullptr) {
      *list_id = list_ids[offset];
    }
    return true;
  }

  bool Contains(DocId doc_id) const { return Get(doc_id, nullptr); }

  void Clear() {
    base = 0;
    list_ids.clear();
    leading_invalid = 0;
  }

  void EnsureRange(DocId min_doc_id, DocId max_doc_id) {
    if (min_doc_id > max_doc_id) {
      return;
    }
    if (list_ids.empty()) {
      base = min_doc_id;
      list_ids.assign(static_cast<size_t>(max_doc_id - min_doc_id) + 1, kInvalidListId);
      leading_invalid = list_ids.size();
      return;
    }

    const DocId old_end = base + static_cast<DocId>(list_ids.size() - 1);
    const DocId new_base = std::min(base, min_doc_id);
    const DocId new_end = std::max(old_end, max_doc_id);
    if (new_base == base) {
      const size_t new_size = static_cast<size_t>(new_end - base) + 1;
      if (new_size > list_ids.size()) {
        list_ids.resize(new_size, kInvalidListId);
      }
      return;
    }

    std::vector<uint32_t> expanded(
        static_cast<size_t>(new_end - new_base) + 1, kInvalidListId);
    std::copy(list_ids.begin(), list_ids.end(),
              expanded.begin() + static_cast<size_t>(base - new_base));
    leading_invalid += static_cast<size_t>(base - new_base);
    base = new_base;
    list_ids.swap(expanded);
  }

  void Set(DocId doc_id, uint32_t list_id) {
    if (!list_ids.empty() && doc_id >= base) {
      const size_t offset = static_cast<size_t>(doc_id - base);
      if (offset < list_ids.size()) {
        list_ids[offset] = list_id;
        if (offset < leading_invalid) {
          leading_invalid = offset;
        }
        return;
      }
    }
    EnsureRange(doc_id, doc_id);
    const size_t offset = static_cast<size_t>(doc_id - base);
    list_ids[offset] = list_id;
    if (offset < leading_invalid) {
      leading_invalid = offset;
    }
  }

  void Erase(DocId doc_id) {
    if (list_ids.empty() || doc_id < base) {
      return;
    }
    const size_t offset = static_cast<size_t>(doc_id - base);
    if (offset < list_ids.size()) {
      list_ids[offset] = kInvalidListId;
      if (offset == leading_invalid) {
        while (leading_invalid < list_ids.size() &&
               list_ids[leading_invalid] == kInvalidListId) {
          ++leading_invalid;
        }
      }
    }
  }

  void MaybeCompact() {
    if (leading_invalid == list_ids.size()) {
      Clear();
      return;
    }
    if (leading_invalid < kDenseDocCompactMinGaps ||
        leading_invalid < (list_ids.size() + 3) / 4) {
      return;
    }
    std::vector<uint32_t> compacted(list_ids.begin() + leading_invalid, list_ids.end());
    base += static_cast<DocId>(leading_invalid);
    list_ids.swap(compacted);
    leading_invalid = 0;
  }
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
  bool pq_codebook_dimension_major{true};
  bool pq_codes_subquantizer_major{true};
  uint32_t M{0};
  uint32_t nbits{8};
  uint32_t Ks{0};
  uint32_t dsub{0};
  // Changes whenever list contents or PQ encoding inputs change. Prepared
  // patch bytes are valid only for the generation they were produced from.
  uint64_t mutation_generation{0};
  MatrixRM routing_centroids;
  Eigen::VectorXf routing_centroid_norms;
  std::vector<MatrixRM> pq_codebooks;
  // Existing codes remain reusable until an in-memory codebook update makes
  // previously stored codes stale. This is conservative after OnlinePQ.
  bool pq_code_reuse_safe{true};
  // Derived SoA cache rebuilt from the serialized AoS codebooks. It is
  // intentionally not part of the index payload.
  std::vector<MatrixRM> pq_codebooks_soa;
  std::vector<std::vector<DocId>> doc_ids_by_list;
  std::vector<std::vector<uint8_t>> pq_codes_by_list;
  std::vector<ListPQCodesSoA> pq_codes_soa_by_list;
  bool use_precomputed_table{false};
  std::vector<float> pq_precomputed_table;
  std::vector<std::vector<uint64_t>> pq_counts;
  double nqe_baseline{0.0};
  double nqe_ema{0.0};
  bool defer_pq_stats_to_add{false};
  uint64_t online_pq_batch_count{0};
  double warmup_nqe_sum{0.0};
  uint32_t warmup_seen_batches{0};
  std::vector<AlignedVector<ListEntry>> lists;
  DenseDocListMap doc_to_list;
  double deferred_nqe_sum{0.0};
  uint64_t deferred_nqe_count{0};
  uint64_t ntotal{0};
  double last_patch_pq_reencode_ms{0.0};
  PatchProfiling last_patch_profiling;
  IngestProfiling last_ingest_profiling;
  IVFBuildProfiling build_profiling;
};

void RebuildPQSoACache(IndexData* data) {
  if (data == nullptr) {
    return;
  }
  data->pq_codebooks_soa.clear();
  if (!data->pq_codebook_dimension_major || !data->use_pq || data->dsub == 0 || data->Ks == 0 ||
      data->pq_codebooks.size() != data->M) {
    return;
  }

  data->pq_codebooks_soa.resize(data->M);
  for (uint32_t m = 0; m < data->M; ++m) {
    const MatrixRM& aos = data->pq_codebooks[static_cast<size_t>(m)];
    if (aos.rows() != static_cast<Eigen::Index>(data->Ks) ||
        aos.cols() != static_cast<Eigen::Index>(data->dsub)) {
      data->pq_codebooks_soa.clear();
      return;
    }
    MatrixRM soa(static_cast<Eigen::Index>(data->dsub),
                 static_cast<Eigen::Index>(data->Ks));
    for (Eigen::Index k = 0; k < aos.rows(); ++k) {
      for (Eigen::Index d = 0; d < aos.cols(); ++d) {
        soa(d, k) = aos(k, d);
      }
    }
    data->pq_codebooks_soa[static_cast<size_t>(m)] = std::move(soa);
  }
}

void AppendIndexData(std::vector<uint8_t>* bytes, const IndexData& data) {
  AppendPod(bytes, data.dim);
  AppendPod(bytes, data.nlist);
  AppendPod(bytes, data.version);
  AppendBool(bytes, data.use_pq);
  AppendBool(bytes, data.pq_residual);
  AppendPod(bytes, data.M);
  AppendPod(bytes, data.nbits);
  AppendPod(bytes, data.Ks);
  AppendPod(bytes, data.dsub);
  AppendMatrix(bytes, data.routing_centroids);
  AppendPod(bytes, data.nqe_baseline);
  AppendPod(bytes, data.nqe_ema);
  AppendBool(bytes, data.defer_pq_stats_to_add);
  AppendPod(bytes, data.online_pq_batch_count);
  AppendPod(bytes, data.warmup_nqe_sum);
  AppendPod(bytes, data.warmup_seen_batches);
  AppendPod(bytes, data.deferred_nqe_sum);
  AppendPod(bytes, data.deferred_nqe_count);
  AppendPod(bytes, data.last_patch_pq_reencode_ms);

  const uint32_t codebook_count = static_cast<uint32_t>(data.pq_codebooks.size());
  AppendPod(bytes, codebook_count);
  for (const MatrixRM& codebook : data.pq_codebooks) {
    AppendMatrix(bytes, codebook);
  }

  const uint32_t counts_count = static_cast<uint32_t>(data.pq_counts.size());
  AppendPod(bytes, counts_count);
  for (const auto& counts : data.pq_counts) {
    AppendUint64Vector(bytes, counts);
  }

  const uint32_t list_count = static_cast<uint32_t>(data.lists.size());
  AppendPod(bytes, list_count);
  for (const auto& list : data.lists) {
    const uint64_t entry_count = static_cast<uint64_t>(list.size());
    AppendPod(bytes, entry_count);
    for (const ListEntry& entry : list) {
      AppendPod(bytes, entry.doc_id);
      AppendPod(bytes, entry.versions.whiten_version);
      AppendPod(bytes, entry.versions.index_version);
      AppendPod(bytes, entry.norm);
      AppendFloatVector(bytes, entry.vector);
      AppendByteVector(bytes, entry.pq_code);
    }
  }
}

Result<std::unique_ptr<IndexData>> ReadIndexData(const std::vector<uint8_t>& bytes,
                                                 size_t* offset) {
  auto data = std::make_unique<IndexData>();
  Status status = ReadPod(bytes, offset, &data->dim);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->nlist);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->version);
  if (!status.ok()) return status;
  status = ReadBool(bytes, offset, &data->use_pq);
  if (!status.ok()) return status;
  status = ReadBool(bytes, offset, &data->pq_residual);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->M);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->nbits);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->Ks);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->dsub);
  if (!status.ok()) return status;
  auto centroids_res = ReadMatrix(bytes, offset);
  if (!centroids_res.ok()) return centroids_res.status();
  data->routing_centroids = std::move(centroids_res.value());
  status = ReadPod(bytes, offset, &data->nqe_baseline);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->nqe_ema);
  if (!status.ok()) return status;
  status = ReadBool(bytes, offset, &data->defer_pq_stats_to_add);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->online_pq_batch_count);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->warmup_nqe_sum);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->warmup_seen_batches);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->deferred_nqe_sum);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->deferred_nqe_count);
  if (!status.ok()) return status;
  status = ReadPod(bytes, offset, &data->last_patch_pq_reencode_ms);
  if (!status.ok()) return status;
  data->pq_code_reuse_safe = (data->online_pq_batch_count == 0);

  if (data->dim == 0 || data->nlist == 0 || data->version == 0) {
    return Status::InvalidArgument("Invalid IVF metadata in payload");
  }
  if (data->routing_centroids.rows() != static_cast<Eigen::Index>(data->nlist) ||
      data->routing_centroids.cols() != static_cast<Eigen::Index>(data->dim)) {
    return Status::InvalidArgument("Routing centroid shape mismatch in IVF payload");
  }
  data->routing_centroid_norms.resize(static_cast<Eigen::Index>(data->nlist));
  for (uint32_t i = 0; i < data->nlist; ++i) {
    data->routing_centroid_norms(static_cast<Eigen::Index>(i)) =
        data->routing_centroids.row(static_cast<Eigen::Index>(i)).squaredNorm();
  }
  if (data->use_pq) {
    if (data->M == 0 || data->Ks == 0 || data->dsub == 0 ||
        data->M * data->dsub != data->dim || data->Ks > 256) {
      return Status::InvalidArgument("Invalid PQ metadata in IVF payload");
    }
  }

  uint32_t codebook_count = 0;
  status = ReadPod(bytes, offset, &codebook_count);
  if (!status.ok()) return status;
  data->pq_codebooks.resize(static_cast<size_t>(codebook_count));
  for (uint32_t i = 0; i < codebook_count; ++i) {
    auto book_res = ReadMatrix(bytes, offset);
    if (!book_res.ok()) return book_res.status();
    data->pq_codebooks[static_cast<size_t>(i)] = std::move(book_res.value());
  }
  if (data->use_pq && data->pq_codebooks.size() != data->M) {
    return Status::InvalidArgument("PQ codebook count mismatch in IVF payload");
  }
  for (const MatrixRM& codebook : data->pq_codebooks) {
    if (codebook.rows() != static_cast<Eigen::Index>(data->Ks) ||
        codebook.cols() != static_cast<Eigen::Index>(data->dsub)) {
      return Status::InvalidArgument("PQ codebook shape mismatch in IVF payload");
    }
  }
  RebuildPQSoACache(data.get());

  uint32_t counts_count = 0;
  status = ReadPod(bytes, offset, &counts_count);
  if (!status.ok()) return status;
  data->pq_counts.resize(static_cast<size_t>(counts_count));
  for (uint32_t i = 0; i < counts_count; ++i) {
    auto counts_res = ReadUint64Vector(bytes, offset);
    if (!counts_res.ok()) return counts_res.status();
    data->pq_counts[static_cast<size_t>(i)] = std::move(counts_res.value());
  }
  if (data->use_pq && data->pq_counts.size() != data->M) {
    return Status::InvalidArgument("PQ count table count mismatch in IVF payload");
  }
  for (const auto& counts : data->pq_counts) {
    if (counts.size() != data->Ks) {
      return Status::InvalidArgument("PQ count table shape mismatch in IVF payload");
    }
  }

  uint32_t list_count = 0;
  status = ReadPod(bytes, offset, &list_count);
  if (!status.ok()) return status;
  if (list_count != data->nlist) {
    return Status::InvalidArgument("List count mismatch in IVF payload");
  }
  data->lists.resize(static_cast<size_t>(list_count));
  for (uint32_t list_id = 0; list_id < list_count; ++list_id) {
    uint64_t entry_count = 0;
    status = ReadPod(bytes, offset, &entry_count);
    if (!status.ok()) return status;
    if (entry_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return Status::InvalidArgument("List in IVF payload is too large");
    }
    auto& list = data->lists[static_cast<size_t>(list_id)];
    list.resize(static_cast<size_t>(entry_count));
    for (uint64_t i = 0; i < entry_count; ++i) {
      ListEntry entry;
      status = ReadPod(bytes, offset, &entry.doc_id);
      if (!status.ok()) return status;
      status = ReadPod(bytes, offset, &entry.versions.whiten_version);
      if (!status.ok()) return status;
      status = ReadPod(bytes, offset, &entry.versions.index_version);
      if (!status.ok()) return status;
      status = ReadPod(bytes, offset, &entry.norm);
      if (!status.ok()) return status;
      auto vector_res = ReadFloatVector(bytes, offset);
      if (!vector_res.ok()) return vector_res.status();
      entry.vector = std::move(vector_res.value());
      auto pq_code_res = ReadByteVector(bytes, offset);
      if (!pq_code_res.ok()) return pq_code_res.status();
      entry.pq_code = std::move(pq_code_res.value());
      if (entry.versions.index_version != data->version) {
        return Status::InvalidArgument("Entry index version mismatch in IVF payload");
      }
      if (data->use_pq) {
        if (entry.pq_code.size() != data->M) {
          return Status::InvalidArgument("PQ code size mismatch in IVF payload");
        }
      } else if (entry.vector.size() != static_cast<Eigen::Index>(data->dim)) {
        return Status::InvalidArgument("Vector dimension mismatch in IVF payload");
      }
      list[static_cast<size_t>(i)] = std::move(entry);
    }
  }

  return data;
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

#if defined(__AVX512F__)
void UpdateNearestCentroidsFromDotsAvx512(const float* dots,
                                          Eigen::Index dots_stride,
                                          Eigen::Index point_count,
                                          Eigen::Index centroid_count,
                                          const float* centroid_norms,
                                          Eigen::Index centroid_begin,
                                          float* best_scores,
                                          int* best_clusters) {
  constexpr Eigen::Index kSimdWidth = 16;
  const __m512 two = _mm512_set1_ps(2.0f);
  for (Eigen::Index point = 0; point < point_count; ++point) {
    float best_score = best_scores[point];
    int best_cluster = best_clusters[point];
    const float* dot_row = dots + point * dots_stride;
    Eigen::Index centroid = 0;
    for (; centroid + kSimdWidth <= centroid_count; centroid += kSimdWidth) {
      const __m512 dot = _mm512_loadu_ps(dot_row + centroid);
      const __m512 norm = _mm512_loadu_ps(
          centroid_norms + centroid_begin + centroid);
      // Keep the scalar path's non-fused norm - 2 * dot arithmetic.
      const __m512 scores = _mm512_sub_ps(norm, _mm512_mul_ps(two, dot));
      const float block_best = _mm512_reduce_min_ps(scores);
      if (block_best < best_score) {
        const __mmask16 equal_mask = _mm512_cmp_ps_mask(
            scores, _mm512_set1_ps(block_best), _CMP_EQ_OQ);
        if (equal_mask != 0) {
          best_score = block_best;
          best_cluster = static_cast<int>(centroid_begin + centroid +
                                          static_cast<Eigen::Index>(
                                              __builtin_ctz(static_cast<unsigned int>(equal_mask))));
        }
      }
    }
    for (; centroid < centroid_count; ++centroid) {
      const float score = centroid_norms[centroid_begin + centroid] -
                          2.0f * dot_row[centroid];
      if (score < best_score) {
        best_score = score;
        best_cluster = static_cast<int>(centroid_begin + centroid);
      }
    }
    best_scores[point] = best_score;
    best_clusters[point] = best_cluster;
  }
}
#endif

void AssignNearestCentroidsBlocked(Eigen::Ref<const MatrixRM> X,
                                   Eigen::Ref<const MatrixRM> centroids,
                                   std::vector<int>* assignments) {
  constexpr int kPointBlockSize = 32;
  constexpr int kCentroidBlockSize = 128;
  const int64_t num_vecs = X.rows();
  const int64_t k = centroids.rows();
  const Eigen::VectorXf centroid_norms = centroids.rowwise().squaredNorm();
  assignments->resize(static_cast<size_t>(num_vecs));
#ifdef _OPENMP
  const bool use_parallel = !omp_in_parallel();
#pragma omp parallel for if (use_parallel) schedule(static)
#endif
  for (int64_t point_begin = 0; point_begin < num_vecs;
       point_begin += kPointBlockSize) {
    const Eigen::Index point_count = static_cast<Eigen::Index>(
        std::min<int64_t>(kPointBlockSize, num_vecs - point_begin));
    Eigen::Matrix<float, kPointBlockSize, kCentroidBlockSize, Eigen::RowMajor> dots;
    std::array<float, kPointBlockSize> best_scores;
    std::array<int, kPointBlockSize> best_clusters{};
    std::fill_n(best_scores.begin(), static_cast<size_t>(point_count),
                std::numeric_limits<float>::max());

    for (int64_t centroid_begin = 0; centroid_begin < k;
         centroid_begin += kCentroidBlockSize) {
      const Eigen::Index centroid_count = static_cast<Eigen::Index>(
          std::min<int64_t>(kCentroidBlockSize, k - centroid_begin));
      dots.topLeftCorner(point_count, centroid_count).noalias() =
          X.middleRows(static_cast<Eigen::Index>(point_begin), point_count) *
          centroids
              .middleRows(static_cast<Eigen::Index>(centroid_begin), centroid_count)
              .transpose();
#if defined(__AVX512F__)
      UpdateNearestCentroidsFromDotsAvx512(
          dots.data(), dots.outerStride(), point_count, centroid_count,
          centroid_norms.data(), static_cast<Eigen::Index>(centroid_begin),
          best_scores.data(), best_clusters.data());
#else
      for (Eigen::Index point = 0; point < point_count; ++point) {
        for (Eigen::Index centroid = 0; centroid < centroid_count; ++centroid) {
          const Eigen::Index centroid_idx =
              static_cast<Eigen::Index>(centroid_begin) + centroid;
          const float score =
              centroid_norms(centroid_idx) - 2.0f * dots(point, centroid);
          if (score < best_scores[static_cast<size_t>(point)]) {
            best_scores[static_cast<size_t>(point)] = score;
            best_clusters[static_cast<size_t>(point)] =
                static_cast<int>(centroid_idx);
          }
        }
      }
#endif
    }
    for (Eigen::Index point = 0; point < point_count; ++point) {
      (*assignments)[static_cast<size_t>(point_begin + point)] =
          best_clusters[static_cast<size_t>(point)];
    }
  }
}

void ComputeRoutingAssignmentsForPQ(Eigen::Ref<const MatrixRM> Xw,
                                    const IndexData& data,
                                    std::vector<int>* assignments) {
  AssignNearestCentroidsBlocked(Xw, data.routing_centroids, assignments);
}

MatrixRM MaterializePQResidualSubspace(Eigen::Ref<const MatrixRM> Xw,
                                       const std::vector<int>& routing_assignments,
                                       const IndexData& data,
                                       uint32_t m) {
  const Eigen::Index offset = static_cast<Eigen::Index>(m * data.dsub);
  const Eigen::Index subdim = static_cast<Eigen::Index>(data.dsub);
  MatrixRM sub(Xw.rows(), subdim);
  auto materialize_row = [&](int64_t i) {
    sub.row(i) = Xw.row(i).segment(offset, subdim);
    if (data.pq_residual) {
      sub.row(i) -= data.routing_centroids
                        .row(routing_assignments[static_cast<size_t>(i)])
                        .segment(offset, subdim);
    }
  };
#ifdef _OPENMP
  if (!omp_in_parallel() && Xw.rows() > 1) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < Xw.rows(); ++i) {
      materialize_row(i);
    }
  } else
#endif
  {
    for (int64_t i = 0; i < Xw.rows(); ++i) {
      materialize_row(i);
    }
  }
  return sub;
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
  if (codebook_soa != nullptr && sub.size() == codebook.cols() &&
      codebook_soa->rows() == codebook.cols() &&
      codebook_soa->cols() == codebook.rows()) {
    return internal::AssignPQSoA(
        sub.data(), static_cast<uint32_t>(codebook.cols()), codebook_soa->data(),
        static_cast<uint32_t>(codebook.rows()), best_dist_out);
  }
  return NearestCodeword(sub, codebook, best_dist_out);
}

Status ValidatePQEncodingState(const IndexData& data) {
  if (!data.use_pq || data.M == 0 || data.nbits == 0 || data.nbits > 8 ||
      data.Ks != (1u << data.nbits) || data.dsub == 0 ||
      data.dim != data.M * data.dsub || data.routing_centroids.rows() != data.nlist ||
      data.routing_centroids.cols() != static_cast<Eigen::Index>(data.dim) ||
      data.pq_codebooks.size() != data.M) {
    return Status::InvalidArgument("Invalid PQ encoding state");
  }
  for (const MatrixRM& codebook : data.pq_codebooks) {
    if (codebook.rows() != static_cast<Eigen::Index>(data.Ks) ||
        codebook.cols() != static_cast<Eigen::Index>(data.dsub)) {
      return Status::InvalidArgument("Invalid PQ codebook shape");
    }
  }
  return Status::OK();
}

void EncodePQForTargetPartition(const IndexData& data,
                                uint32_t partition_id,
                                Eigen::Ref<const Eigen::VectorXf> vector,
                                uint8_t* code_out) {
  // Callers validate the IndexData, partition, vector, and output shape before
  // invoking this shared prepare/legacy-commit encoding implementation.
  Eigen::VectorXf residual = vector;
  if (data.pq_residual) {
    residual -=
        data.routing_centroids.row(static_cast<Eigen::Index>(partition_id)).transpose();
  }
  for (uint32_t m = 0; m < data.M; ++m) {
    const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
    Eigen::Map<const Eigen::VectorXf> sub(
        residual.data() + static_cast<Eigen::Index>(m * data.dsub),
        static_cast<Eigen::Index>(data.dsub));
    const MatrixRM* codebook_soa =
        data.pq_codebooks_soa.size() == data.M
            ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
            : nullptr;
    code_out[m] =
        static_cast<uint8_t>(NearestCodeword(sub, codebook, codebook_soa, nullptr));
  }
}

PQEncodingContext MakePQEncodingContext(const IndexData& data) {
  PQEncodingContext context;
  context.index_version = data.version;
  context.mutation_generation = data.mutation_generation;
  context.use_pq = data.use_pq;
  context.pq_residual = data.pq_residual;
  context.M = data.M;
  context.nbits = data.nbits;
  context.Ks = data.Ks;
  context.dsub = data.dsub;
  return context;
}

bool MatchesPQEncodingContext(const IndexData& data,
                              const PQEncodingContext& context) {
  return context.index_version == data.version &&
         context.mutation_generation == data.mutation_generation &&
         context.use_pq == data.use_pq &&
         context.pq_residual == data.pq_residual && context.M == data.M &&
         context.nbits == data.nbits && context.Ks == data.Ks &&
         context.dsub == data.dsub;
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

enum class KMeansMode {
  kGeneric,
  kPQ,
};

struct KMeansProfiling {
  double assignment_us{0.0};
  double update_us{0.0};
};

void RunKMeans(Eigen::Ref<const MatrixRM> X,
               MatrixRM* centroids,
               uint32_t iterations,
               KMeansMode mode = KMeansMode::kGeneric,
               KMeansProfiling* profiling = nullptr) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  const int64_t k = centroids->rows();
  std::vector<int> assignments(num_vecs, 0);

  const uint32_t kmeans_iters = (iterations == 0) ? kDefaultKMeansIterations : iterations;
#ifdef _OPENMP
  // Avoid nested teams when KMeans is invoked from any parallel caller.
  const bool use_parallel = !omp_in_parallel();
  const int num_threads = use_parallel ? std::max(1, omp_get_max_threads()) : 1;
#endif
  MatrixRM new_centroids(k, dim);
  std::vector<int64_t> counts(static_cast<size_t>(k));
#ifdef _OPENMP
  std::vector<MatrixRM> partial_sums(
      static_cast<size_t>(num_threads), MatrixRM(k, dim));
  std::vector<std::vector<int64_t>> partial_counts(
      static_cast<size_t>(num_threads), std::vector<int64_t>(static_cast<size_t>(k)));
#endif
  for (uint32_t iter = 0; iter < kmeans_iters; ++iter) {
    Timer assignment_timer;
#if defined(__AVX512F__)
    if (mode == KMeansMode::kPQ) {
      MatrixRM centroids_soa(dim, k);
      for (Eigen::Index centroid = 0; centroid < centroids->rows(); ++centroid) {
        for (Eigen::Index d = 0; d < centroids->cols(); ++d) {
          centroids_soa(d, centroid) = (*centroids)(centroid, d);
        }
      }
#ifdef _OPENMP
#pragma omp parallel for if (use_parallel) schedule(static)
#endif
      for (int64_t i = 0; i < num_vecs; ++i) {
        assignments[static_cast<size_t>(i)] = static_cast<int>(
            internal::AssignPQSoA(X.row(i).data(), static_cast<uint32_t>(dim),
                                  centroids_soa.data(), static_cast<uint32_t>(k), nullptr));
      }
    } else
#endif
    {
      // Assignment step.
      AssignNearestCentroidsBlocked(X, *centroids, &assignments);
    }
    if (profiling != nullptr) {
      profiling->assignment_us += assignment_timer.ElapsedMicros();
    }

    // Update step.
    Timer update_timer;
    new_centroids.setZero();
    std::fill(counts.begin(), counts.end(), 0);
#ifdef _OPENMP
    for (MatrixRM& partial_sum : partial_sums) {
      partial_sum.setZero();
    }
    for (auto& partial_count : partial_counts) {
      std::fill(partial_count.begin(), partial_count.end(), 0);
    }

#pragma omp parallel if (use_parallel)
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
    if (profiling != nullptr) {
      profiling->update_us += update_timer.ElapsedMicros();
    }
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
  if (!data->use_pq || !data->pq_residual || data->M == 0 || data->Ks == 0 || data->dsub == 0) {
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
      if (codebook.rows() != static_cast<Eigen::Index>(data->Ks) ||
          codebook.cols() != static_cast<Eigen::Index>(data->dsub)) {
        data->pq_precomputed_table.clear();
        return;
      }
      Eigen::Map<const Eigen::VectorXf> centroid_sub(
          data->routing_centroids.row(static_cast<Eigen::Index>(list_id)).data() +
              static_cast<Eigen::Index>(m * data->dsub),
          static_cast<Eigen::Index>(data->dsub));
      for (uint32_t k = 0; k < data->Ks; ++k) {
        Eigen::Map<const Eigen::VectorXf> z(
            codebook.row(static_cast<Eigen::Index>(k)).data(),
            static_cast<Eigen::Index>(data->dsub));
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

bool HasCompactDocIdsForList(const IndexData& data, uint32_t list_id) {
  if (!data.use_pq) {
    return false;
  }
  if (list_id >= data.lists.size() || list_id >= data.doc_ids_by_list.size()) {
    return false;
  }
  return data.doc_ids_by_list[static_cast<size_t>(list_id)].size() ==
         data.lists[static_cast<size_t>(list_id)].size();
}

bool HasPQCodebookSoA(const IndexData& data) {
  if (!data.use_pq || data.M == 0 || data.Ks == 0 || data.dsub == 0 ||
      data.pq_codebooks_soa.size() != data.M) {
    return false;
  }
  for (uint32_t m = 0; m < data.M; ++m) {
    const MatrixRM& soa = data.pq_codebooks_soa[static_cast<size_t>(m)];
    if (soa.rows() != static_cast<Eigen::Index>(data.dsub) ||
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

bool HasCompactPQListForSearch(const IndexData& data, uint32_t list_id) {
  return HasCompactDocIdsForList(data, list_id) &&
         (HasSoAPQCodesForList(data, list_id) ||
          HasContiguousPQCodesForList(data, list_id));
}

void ClearListPQCodesSoA(IndexData* data, uint32_t list_id) {
  if (data == nullptr || list_id >= data->pq_codes_soa_by_list.size()) {
    return;
  }
  data->pq_codes_soa_by_list[static_cast<size_t>(list_id)] = ListPQCodesSoA{};
}

void ReserveListPQCodesSoA(IndexData* data, uint32_t list_id, size_t rows_needed) {
  if (data == nullptr || !data->use_pq || !data->pq_codes_subquantizer_major ||
      data->M == 0 || list_id >= data->lists.size()) {
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
  if (data->doc_ids_by_list.size() != data->lists.size()) {
    data->doc_ids_by_list.resize(data->lists.size());
  }
  if (data->pq_codes_subquantizer_major &&
      data->pq_codes_soa_by_list.size() != data->lists.size()) {
    data->pq_codes_soa_by_list.resize(data->lists.size());
  }
  const auto& list = data->lists[static_cast<size_t>(list_id)];
  auto& doc_ids = data->doc_ids_by_list[static_cast<size_t>(list_id)];
  auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
  ListPQCodesSoA* soa = data->pq_codes_subquantizer_major
                            ? &data->pq_codes_soa_by_list[static_cast<size_t>(list_id)]
                            : nullptr;
  doc_ids.clear();
  doc_ids.reserve(list.size());
  flat.clear();
  flat.reserve(list.size() * data->M);
  if (soa != nullptr) {
    soa->rows = list.size();
    soa->stride = list.size();
    soa->codes.assign(static_cast<size_t>(data->M) * soa->stride, 0);
  }
  for (size_t row = 0; row < list.size(); ++row) {
    const auto& entry = list[row];
    if (entry.pq_code.size() != data->M) {
      doc_ids.clear();
      flat.clear();
      ClearListPQCodesSoA(data, list_id);
      return;
    }
    doc_ids.push_back(entry.doc_id);
    flat.insert(flat.end(), entry.pq_code.begin(), entry.pq_code.end());
    if (soa != nullptr) {
      for (uint32_t m = 0; m < data->M; ++m) {
        soa->codes[static_cast<size_t>(m) * soa->stride + row] =
            entry.pq_code[static_cast<size_t>(m)];
      }
    }
  }
}

void AppendListPQCaches(IndexData* data,
                        uint32_t list_id,
                        size_t rows_before,
                        const ListEntry& entry) {
  if (data == nullptr || !data->use_pq || data->M == 0 || list_id >= data->lists.size()) {
    return;
  }
  const std::vector<uint8_t>& code = entry.pq_code;
  if (code.size() != data->M) {
    if (list_id < data->doc_ids_by_list.size()) {
      data->doc_ids_by_list[static_cast<size_t>(list_id)].clear();
    }
    ClearListPQCodesSoA(data, list_id);
    return;
  }
  if (data->doc_ids_by_list.size() != data->lists.size()) {
    data->doc_ids_by_list.resize(data->lists.size());
  }
  if (data->pq_codes_subquantizer_major &&
      data->pq_codes_soa_by_list.size() != data->lists.size()) {
    data->pq_codes_soa_by_list.resize(data->lists.size());
  }
  if (data->lists[static_cast<size_t>(list_id)].size() != rows_before + 1) {
    RebuildListPQCodes(data, list_id);
    return;
  }

  auto& doc_ids = data->doc_ids_by_list[static_cast<size_t>(list_id)];
  if (!data->pq_codes_subquantizer_major) {
    auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
    if (doc_ids.size() != rows_before || flat.size() != rows_before * data->M) {
      RebuildListPQCodes(data, list_id);
      return;
    }
    doc_ids.push_back(entry.doc_id);
    flat.insert(flat.end(), code.begin(), code.end());
    return;
  }
  auto& soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
  const bool appendable = doc_ids.size() == rows_before &&
                          soa.rows == rows_before && soa.stride >= rows_before &&
                          soa.codes.size() == static_cast<size_t>(data->M) * soa.stride;
  if (!appendable) {
    RebuildListPQCodes(data, list_id);
    return;
  }
  doc_ids.push_back(entry.doc_id);
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
  if (!HasCompactDocIdsForList(*data, list_id)) {
    RebuildListPQCodes(data, list_id);
  }
  if (data->pq_codes_subquantizer_major && !HasSoAPQCodesForList(*data, list_id)) {
    RebuildListPQCodes(data, list_id);
  }
  if (list_id >= data->doc_ids_by_list.size() ||
      list_id >= data->pq_codes_by_list.size() ||
      (data->pq_codes_subquantizer_major &&
       list_id >= data->pq_codes_soa_by_list.size())) {
    return;
  }
  auto& doc_ids = data->doc_ids_by_list[static_cast<size_t>(list_id)];
  auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
  if (flat.size() % data->M != 0) {
    RebuildListPQCodes(data, list_id);
  }
  size_t rows = flat.size() / data->M;
  if (pos >= rows || doc_ids.size() != rows) {
    RebuildListPQCodes(data, list_id);
    rows = flat.size() / data->M;
    if (flat.size() % data->M != 0 ||
        list_id >= data->doc_ids_by_list.size() ||
        data->doc_ids_by_list[static_cast<size_t>(list_id)].size() != rows ||
        pos >= rows) {
      return;
    }
  }
  if (data->pq_codes_subquantizer_major) {
    auto& soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
    if (!HasSoAPQCodesForList(*data, list_id) || pos >= soa.rows) {
      RebuildListPQCodes(data, list_id);
    }
    if (!HasSoAPQCodesForList(*data, list_id) ||
        pos >= data->pq_codes_soa_by_list[static_cast<size_t>(list_id)].rows) {
      return;
    }
  }
  const size_t last = rows - 1;
  auto& updated_doc_ids = data->doc_ids_by_list[static_cast<size_t>(list_id)];
  if (pos != last) {
    updated_doc_ids[pos] = updated_doc_ids[last];
    std::copy_n(flat.data() + last * data->M,
                data->M,
                flat.data() + pos * data->M);
  }
  updated_doc_ids.resize(last);
  flat.resize(last * data->M);

  if (data->pq_codes_subquantizer_major) {
    auto& updated_soa = data->pq_codes_soa_by_list[static_cast<size_t>(list_id)];
    for (uint32_t m = 0; m < data->M; ++m) {
      const size_t offset = static_cast<size_t>(m) * updated_soa.stride;
      if (pos != last) {
        updated_soa.codes[offset + pos] = updated_soa.codes[offset + last];
      }
    }
    updated_soa.rows = last;
  }
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

  const bool use_pq = data.use_pq && data.M > 0 && data.Ks > 0 && data.dsub > 0 &&
                      data.pq_codebooks.size() == data.M;
  const bool use_precomputed_table =
      use_pq && data.use_precomputed_table &&
      data.pq_precomputed_table.size() == PrecomputedTableElementCount(data);
  const bool use_fast_scan = use_pq && data.M >= 4;
  const bool use_codebook_soa = use_pq && data.pq_codebook_dimension_major &&
                                HasPQCodebookSoA(data);
  const bool use_vectorized_code_soa_scan =
      use_pq && data.pq_codes_subquantizer_major &&
      internal::HasVectorizedPQCodeSoAScan();
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
  double pq_lut_build_us = 0.0;
  Timer query_lut_timer;
  if (use_pq) {
    distance_table.resize(static_cast<size_t>(data.M) * data.Ks);
    if (use_precomputed_table) {
      query_term3_table.resize(static_cast<size_t>(data.M) * data.Ks, 0.0f);
      for (uint32_t m = 0; m < data.M; ++m) {
        Eigen::Map<const Eigen::VectorXf> qsub(
            qw.data() + static_cast<Eigen::Index>(m * data.dsub),
            static_cast<Eigen::Index>(data.dsub));
        float* table_ptr = query_term3_table.data() + static_cast<size_t>(m) * data.Ks;
        if (use_codebook_soa) {
          internal::BuildPQNeg2DotTableSoA(qsub.data(),
                                           data.dsub,
                                           data.pq_codebooks_soa[static_cast<size_t>(m)].data(),
                                           data.Ks,
                                           table_ptr);
        } else {
          const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
          for (uint32_t k = 0; k < data.Ks; ++k) {
            const float qz = qsub.dot(codebook.row(static_cast<Eigen::Index>(k)).transpose());
            table_ptr[k] = -2.0f * qz;
          }
        }
      }
    }
  }
  if (use_pq) {
    pq_lut_build_us += query_lut_timer.ElapsedMicros();
  }
  std::vector<float> soa_scan_dists;
  double pq_adc_scan_us = 0.0;
  for (uint32_t pi = 0; pi < probes; ++pi) {
    const uint32_t list_id = centroid_dists[static_cast<size_t>(pi)].second;
    const float coarse_dist = centroid_dists[static_cast<size_t>(pi)].first;
    if (use_pq) {
      Timer lut_timer;
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
          Eigen::Map<const Eigen::VectorXf> qsub(
              qres.data() + static_cast<Eigen::Index>(m * data.dsub),
              static_cast<Eigen::Index>(data.dsub));
          float* table_ptr = distance_table.data() + static_cast<size_t>(m) * data.Ks;
          if (use_codebook_soa) {
            internal::BuildPQDistanceTableSoA(qsub.data(),
                                              data.dsub,
                                              data.pq_codebooks_soa[static_cast<size_t>(m)].data(),
                                              data.Ks,
                                              table_ptr);
          } else {
            const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
            for (uint32_t k = 0; k < data.Ks; ++k) {
              const float dist =
                  (qsub - codebook.row(static_cast<Eigen::Index>(k)).transpose()).squaredNorm();
              table_ptr[k] = dist;
            }
          }
        }
      }
      pq_lut_build_us += lut_timer.ElapsedMicros();
    }
    const auto& list = data.lists[static_cast<size_t>(list_id)];
    const bool use_compact_pq_scan = use_pq && HasCompactPQListForSearch(data, list_id);
    const std::vector<DocId>* compact_doc_ids =
        use_compact_pq_scan ? &data.doc_ids_by_list[static_cast<size_t>(list_id)] : nullptr;
    const size_t list_size = use_compact_pq_scan ? compact_doc_ids->size() : list.size();
    const bool use_contiguous_codes = use_pq && HasContiguousPQCodesForList(data, list_id);
    const uint8_t* contiguous_codes =
        use_contiguous_codes ? data.pq_codes_by_list[static_cast<size_t>(list_id)].data() : nullptr;
    const bool use_soa_codes =
        use_vectorized_code_soa_scan && HasSoAPQCodesForList(data, list_id);
    const ListPQCodesSoA* soa_codes =
        use_soa_codes ? &data.pq_codes_soa_by_list[static_cast<size_t>(list_id)] : nullptr;
    Timer scan_timer;
    if (soa_codes != nullptr && list_size > 0) {
      soa_scan_dists.resize(list_size);
      internal::ScanPQCodesSoA(distance_table.data(),
                               soa_codes->codes.data(),
                               data.M,
                               data.Ks,
                               soa_codes->stride,
                               list_size,
                               soa_scan_dists.data());
    }
    for (size_t li = 0; li < list_size; ++li) {
      ++scanned;
      DocId doc_id = 0;
      float approx_dist = 0.0f;
      if (use_pq) {
        doc_id = use_compact_pq_scan ? (*compact_doc_ids)[li] : list[li].doc_id;
        if (soa_codes != nullptr) {
          approx_dist = soa_scan_dists[li];
        } else {
          const uint8_t* code_ptr = nullptr;
          if (use_contiguous_codes) {
            code_ptr = contiguous_codes + li * data.M;
          } else if (list[li].pq_code.size() == data.M) {
            code_ptr = list[li].pq_code.data();
          }
          if (code_ptr == nullptr) {
            continue;
          }
          approx_dist =
              use_fast_scan ? AccumulateDistanceFastScan(distance_table.data(), code_ptr, data.M, data.Ks)
                            : 0.0f;
          if (!use_fast_scan) {
            for (uint32_t m = 0; m < data.M; ++m) {
              const uint8_t code = code_ptr[m];
              approx_dist += distance_table[static_cast<size_t>(m) * data.Ks + code];
            }
          }
        }
        if (use_precomputed_table) {
          approx_dist += coarse_dist;
        }
      } else {
        const auto& entry = list[li];
        doc_id = entry.doc_id;
        const float dot = entry.vector.dot(qw);
        approx_dist = qnorm + entry.norm - 2.0f * dot;
      }
      if (collect_scan_trace) {
        scanned_doc_ids.push_back(doc_id);
        scanned_approx_dists.push_back(approx_dist);
      }
      if (heap.size() < topk ||
          (!heap.empty() && approx_dist < heap.front().approx_dist)) {
        const auto& entry = list[li];
        Candidate cand;
        cand.doc_id = doc_id;
        cand.approx_dist = approx_dist;
        cand.rerank_dist = approx_dist;
        cand.versions = entry.versions;
        cand.versions.index_version = data.version;
        cand.from_new = from_new;
        if (heap.size() == topk) {
          std::pop_heap(heap.begin(), heap.end(), heap_cmp);
          heap.back() = std::move(cand);
          std::push_heap(heap.begin(), heap.end(), heap_cmp);
          continue;
        }
        heap.push_back(std::move(cand));
        std::push_heap(heap.begin(), heap.end(), heap_cmp);
      }
    }
    if (use_pq) {
      pq_adc_scan_us += scan_timer.ElapsedMicros();
    }
  }

  SearchResult result;
  result.scanned_candidates = scanned;
  result.pq_lut_build_us = pq_lut_build_us;
  result.pq_adc_scan_us = pq_adc_scan_us;
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
                          VersionId index_version,
                          std::vector<int>* routing_assignments_out) override {
    if (routing_assignments_out != nullptr) {
      routing_assignments_out->clear();
    }
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

    IVFBuildProfiling build_profiling;
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
      Timer centroid_init_timer;
      centroids = InitializeCentroids(Xw, nlist);
      build_profiling.build_coarse_centroid_init_us = centroid_init_timer.ElapsedMicros();
      KMeansProfiling coarse_kmeans_profiling;
      Timer coarse_kmeans_timer;
      RunKMeans(Xw,
                &centroids,
                p.kmeans_iterations,
                KMeansMode::kGeneric,
                &coarse_kmeans_profiling);
      build_profiling.build_coarse_kmeans_us = coarse_kmeans_timer.ElapsedMicros();
      build_profiling.build_coarse_kmeans_assignment_us =
          coarse_kmeans_profiling.assignment_us;
      build_profiling.build_coarse_kmeans_update_us = coarse_kmeans_profiling.update_us;
    }
    build_profiling.build_coarse_kmeans_rows = static_cast<uint64_t>(Xw.rows());
    build_profiling.build_coarse_kmeans_k = nlist;
    build_profiling.build_coarse_kmeans_dim = dim;
    build_profiling.build_coarse_kmeans_iterations =
        p.use_fixed_routing_centroids ? 0 : p.kmeans_iterations;

    auto data = std::make_unique<IndexData>();
    data->build_profiling = build_profiling;
    data->dim = dim;
    data->nlist = nlist;
    data->routing_centroids = std::move(centroids);
    data->routing_centroid_norms = data->routing_centroids.rowwise().squaredNorm();
    data->use_pq = p.pq.enable;
    data->pq_residual = p.pq.residual;
    data->pq_codebook_dimension_major = p.pq_codebook_dimension_major;
    data->pq_codes_subquantizer_major = p.pq_codes_subquantizer_major;
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
    data->doc_ids_by_list.clear();
    data->pq_codes_by_list.clear();
    data->pq_codes_soa_by_list.clear();
    if (data->use_pq) {
      data->doc_ids_by_list.resize(nlist);
      data->pq_codes_by_list.resize(nlist);
      if (data->pq_codes_subquantizer_major) {
        data->pq_codes_soa_by_list.resize(nlist);
      }
    }
    data->doc_to_list.Clear();
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
        Timer pq_training_timer;
        std::vector<int> local_routing_assignments;
        std::vector<int>& routing_assignments =
            routing_assignments_out != nullptr ? *routing_assignments_out
                                               : local_routing_assignments;
        Timer routing_assignment_timer;
        ComputeRoutingAssignmentsForPQ(Xw, *data, &routing_assignments);
        data->build_profiling.build_pq_routing_assignment_us =
            routing_assignment_timer.ElapsedMicros();
        data->build_profiling.build_pq_routing_assignment_rows =
            static_cast<uint64_t>(Xw.rows());
        data->build_profiling.build_pq_training_concurrency = kPQTrainingConcurrency;
        data->build_profiling.build_pq_max_live_subspaces = 1;

        for (uint32_t m = 0; m < data->M; ++m) {
          Timer materialize_timer;
          MatrixRM sub =
              MaterializePQResidualSubspace(Xw, routing_assignments, *data, m);
          data->build_profiling.build_pq_subspace_materialize_us +=
              materialize_timer.ElapsedMicros();
          data->build_profiling.build_pq_subspace_materialized_rows +=
              static_cast<uint64_t>(sub.rows());
          data->build_profiling.build_pq_subspace_materialized_bytes +=
              static_cast<uint64_t>(sub.size()) * sizeof(float);

          Timer kmeans_timer;
          Timer centroid_init_timer;
          MatrixRM codebook = InitializeCentroids(sub, data->Ks);
          data->build_profiling.build_pq_centroid_init_us +=
              centroid_init_timer.ElapsedMicros();
          KMeansProfiling pq_kmeans_profiling;
          RunKMeans(sub,
                    &codebook,
                    p.kmeans_iterations,
                    KMeansMode::kPQ,
                    &pq_kmeans_profiling);
          data->build_profiling.build_pq_kmeans_assignment_us +=
              pq_kmeans_profiling.assignment_us;
          data->build_profiling.build_pq_kmeans_update_us +=
              pq_kmeans_profiling.update_us;
          data->build_profiling.build_pq_kmeans_us += kmeans_timer.ElapsedMicros();
          data->pq_codebooks[static_cast<size_t>(m)] = std::move(codebook);
        }
        data->build_profiling.build_pq_training_total_us =
            pq_training_timer.ElapsedMicros();
      }

      Timer codebook_soa_timer;
      RebuildPQSoACache(data.get());
      data->build_profiling.build_pq_codebook_soa_us =
          codebook_soa_timer.ElapsedMicros();

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
            Eigen::Map<const Eigen::VectorXf> sub(
                residual.data() + static_cast<Eigen::Index>(m * data->dsub),
                static_cast<Eigen::Index>(data->dsub));
            float best_dist = 0.0f;
            const MatrixRM* codebook_soa =
                data->pq_codebooks_soa.size() == data->M
                    ? &data->pq_codebooks_soa[static_cast<size_t>(m)]
                    : nullptr;
            const uint32_t k =
                NearestCodeword(sub, data->pq_codebooks[static_cast<size_t>(m)], codebook_soa,
                                &best_dist);
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
        data->nqe_ema = (p.fixed_pq_ema_nqe > 0.0) ? p.fixed_pq_ema_nqe : data->nqe_baseline;
      }
      Timer precomputed_table_timer;
      BuildPrecomputedTable(data.get());
      data->build_profiling.build_pq_precomputed_table_us =
          precomputed_table_timer.ElapsedMicros();
    }

    Timer publication_timer;
    std::unique_lock lock(mu_);
    const VersionId version = (index_version == 0) ? next_version_++ : index_version;
    data->version = version;
    data->mutation_generation = next_mutation_generation_++;
    IndexData* published_data = data.get();
    data_map_[version] = std::move(data);
    latest_version_ = version;
    published_data->build_profiling.build_publication_us =
        publication_timer.ElapsedMicros();
    return version;
  }

  Status Add(const AlignedVector<VectorRecord>& recs) override {
    return AddBatch(recs, true, nullptr);
  }

  Status AddBatch(const AlignedVector<VectorRecord>& recs,
                  bool finalize_deferred_stats,
                  const int* precomputed_assignments) override {
    std::unique_lock lock(mu_);
    if (latest_version_ == 0) {
      return Status::InvalidArgument("Index not built");
    }
    auto it = data_map_.find(latest_version_);
    if (it == data_map_.end()) {
      return Status::NotFound("Latest version missing");
    }
    IndexData& data = *it->second;
    Timer validation_timer;
    Status validate = ValidateRecordsForInsertLocked(data, recs);
    const double validation_us = validation_timer.ElapsedMicros();
    if (!validate.ok()) {
      return validate;
    }
    if (!recs.empty()) {
      data.mutation_generation = next_mutation_generation_++;
    }

    data.last_ingest_profiling = IngestProfiling{};
    data.last_ingest_profiling.records = recs.size();
    data.last_ingest_profiling.validation_us = validation_us;
    std::vector<int> computed_centroids;
    const int* centroids = precomputed_assignments;
    if (centroids == nullptr) {
      computed_centroids.resize(recs.size(), 0);
      centroids = computed_centroids.data();
    }
    AlignedVector<ListEntry> entries(recs.size());
    const bool collect_initial_pq_stats = data.use_pq && data.defer_pq_stats_to_add;
    std::vector<double> per_record_nqe;
    if (collect_initial_pq_stats) {
      per_record_nqe.assign(recs.size(), 0.0);
    }

    Timer assignment_timer;
    if (precomputed_assignments == nullptr) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        computed_centroids[i] = NearestCentroid(recs[i].x, data);
      }
    }
    data.last_ingest_profiling.assignment_us = assignment_timer.ElapsedMicros();

    Timer encode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& rec = recs[i];
      ListEntry entry;
      const int centroid = centroids[i];
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
          Eigen::Map<const Eigen::VectorXf> sub(
              residual.data() + static_cast<Eigen::Index>(m * data.dsub),
              static_cast<Eigen::Index>(data.dsub));
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
      entries[i] = std::move(entry);
    }
    data.last_ingest_profiling.encode_us = encode_timer.ElapsedMicros();

    if (collect_initial_pq_stats) {
      Timer deferred_pq_stats_timer;
      for (size_t i = 0; i < entries.size(); ++i) {
        data.deferred_nqe_sum += per_record_nqe[i];
        data.deferred_nqe_count++;
        for (uint32_t m = 0; m < data.M; ++m) {
          const uint32_t k = static_cast<uint32_t>(entries[i].pq_code[static_cast<size_t>(m)]);
          if (k < data.Ks) {
            data.pq_counts[static_cast<size_t>(m)][static_cast<size_t>(k)]++;
          }
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
      data.last_ingest_profiling.deferred_pq_stats_us =
          deferred_pq_stats_timer.ElapsedMicros();
    }

    Timer commit_timer;
    CommitPendingLocked(&data, centroids, &entries);
    data.last_ingest_profiling.commit_us = commit_timer.ElapsedMicros();
    return Status::OK();
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
    data.last_ingest_profiling = IngestProfiling{};
    data.last_ingest_profiling.records = recs.size();
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
    // Invalidate every prepared patch before the first possible mutation.
    // Conservatively keeping the new generation on a later error is safe.
    data.mutation_generation = next_mutation_generation_++;

    const bool can_online = data.use_pq && data.pq_residual && data.M > 0 && data.Ks > 0 &&
                            data.dsub > 0 && data.pq_codebooks.size() == data.M &&
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
      Timer assignment_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        centroids[i] = NearestCentroid(recs[i].x, data);
      }
      stats.insert_assignment_us += assignment_timer.ElapsedMicros();
      data.last_ingest_profiling.assignment_us = stats.insert_assignment_us;

      Timer encode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        const auto& rec = recs[i];
        ListEntry entry;
        const int centroid = centroids[i];
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
        entries[i] = std::move(entry);
      }
      stats.insert_encode_ms += encode_timer.ElapsedMillis();
      data.last_ingest_profiling.encode_us = stats.insert_encode_ms * 1000.0;
      Timer commit_timer;
      CommitPendingLocked(&data, centroids.data(), &entries);
      stats.insert_commit_ms += commit_timer.ElapsedMillis();
      data.last_ingest_profiling.commit_us = stats.insert_commit_ms * 1000.0;
      stats.insert_ms += stats.insert_assignment_us / 1000.0 +
                         stats.insert_encode_ms + stats.insert_commit_ms;
      stats.maintenance_ms += stats.delete_ms;
      return stats;
    }

    stats.use_online_pq = true;
    const double eps = std::max(1e-12, options.nqe_eps);
    const double ema_alpha = options.ema_alpha;
    const uint32_t n = static_cast<uint32_t>(recs.size());

    std::vector<int> centroids(recs.size(), 0);
    MatrixRM residuals(static_cast<Eigen::Index>(recs.size()),
                       static_cast<Eigen::Index>(data.dim));
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
      sum_vec.emplace_back(MatrixRM::Zero(data.Ks, data.dsub));
      sum_err.emplace_back(MatrixRM::Zero(data.Ks, data.dsub));
    }
    std::vector<double> sub_err_sum(data.M, 0.0);
    std::vector<double> sub_energy_sum(data.M, 0.0);
    std::vector<std::vector<uint32_t>> delete_cnt(
        data.M, std::vector<uint32_t>(data.Ks, 0));
    std::vector<MatrixRM> delete_sum_vec;
    delete_sum_vec.reserve(data.M);
    for (uint32_t m = 0; m < data.M; ++m) {
      delete_sum_vec.emplace_back(MatrixRM::Zero(data.Ks, data.dsub));
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
          Eigen::Map<const Eigen::VectorXf> sub(
              residual.data() + static_cast<Eigen::Index>(m * data.dsub),
              static_cast<Eigen::Index>(data.dsub));
          delete_sum_vec[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(k)) +=
              sub.transpose();
        }
      }
      stats.delete_ms += delete_timer.ElapsedMillis();
    }

    Timer assignment_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      centroids[i] = NearestCentroid(recs[i].x, data);
    }
    stats.insert_assignment_us += assignment_timer.ElapsedMicros();
    data.last_ingest_profiling.assignment_us = stats.insert_assignment_us;

    Timer encode_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(recs.size()); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const auto& rec = recs[i];
      const int centroid = centroids[i];
      residuals.row(static_cast<Eigen::Index>(i)) = rec.x.transpose();
      if (data.pq_residual) {
        residuals.row(static_cast<Eigen::Index>(i)) -=
            data.routing_centroids.row(static_cast<Eigen::Index>(centroid));
      }

      double err2 = 0.0;
      double r2 = 0.0;
      const float* residual_ptr = residuals.row(static_cast<Eigen::Index>(i)).data();
      for (uint32_t m = 0; m < data.M; ++m) {
        const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
        Eigen::Map<const Eigen::VectorXf> sub(
            residual_ptr + static_cast<Eigen::Index>(m * data.dsub),
            static_cast<Eigen::Index>(data.dsub));
        const MatrixRM* codebook_soa =
            data.pq_codebooks_soa.size() == data.M
                ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                : nullptr;
        float best_dist = 0.0f;
        const uint32_t best_idx = NearestCodeword(sub, codebook, codebook_soa, &best_dist);
        const size_t code_offset = i * static_cast<size_t>(data.M) + static_cast<size_t>(m);
        codes_before[code_offset] = static_cast<uint8_t>(best_idx);
        best_dists[code_offset] = best_dist;
        sub_energies[code_offset] = sub.squaredNorm();
        err2 += static_cast<double>(best_dist);
        r2 += static_cast<double>(sub_energies[code_offset]);
      }
      nqe_by_record[i] = err2 / (r2 + eps);
    }
    stats.insert_encode_ms += encode_timer.ElapsedMillis();
    data.last_ingest_profiling.encode_us = stats.insert_encode_ms * 1000.0;

    Timer onlinepq_stats_timer;
    double nqe_sum = 0.0;
    for (size_t i = 0; i < recs.size(); ++i) {
      const float* residual_ptr = residuals.row(static_cast<Eigen::Index>(i)).data();
      for (uint32_t m = 0; m < data.M; ++m) {
        const MatrixRM& codebook = data.pq_codebooks[static_cast<size_t>(m)];
        const size_t code_offset = i * static_cast<size_t>(data.M) + static_cast<size_t>(m);
        const uint32_t best_idx = static_cast<uint32_t>(codes_before[code_offset]);
        Eigen::Map<const Eigen::VectorXf> sub(
            residual_ptr + static_cast<Eigen::Index>(m * data.dsub),
            static_cast<Eigen::Index>(data.dsub));
        batch_cnt[static_cast<size_t>(m)][static_cast<size_t>(best_idx)]++;
        sum_vec[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(best_idx)) +=
            sub.transpose();
        sum_err[static_cast<size_t>(m)].row(static_cast<Eigen::Index>(best_idx)) +=
            (sub - codebook.row(static_cast<Eigen::Index>(best_idx)).transpose()).transpose();
        const double e2 = static_cast<double>(best_dists[code_offset]);
        const double en = static_cast<double>(sub_energies[code_offset]);
        sub_err_sum[static_cast<size_t>(m)] += e2;
        sub_energy_sum[static_cast<size_t>(m)] += en;
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
        data.pq_code_reuse_safe = false;
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
            Eigen::Map<const Eigen::VectorXf> sub(
                residual_ptr + static_cast<Eigen::Index>(m * data.dsub),
                static_cast<Eigen::Index>(data.dsub));
            const MatrixRM* codebook_soa =
                data.pq_codebooks_soa.size() == data.M
                    ? &data.pq_codebooks_soa[static_cast<size_t>(m)]
                    : nullptr;
            codes_for_insert[static_cast<size_t>(i) * static_cast<size_t>(data.M) +
                             static_cast<size_t>(m)] =
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
      const uint8_t* code_begin =
          codes_for_insert.data() + i * static_cast<size_t>(data.M);
      entry.pq_code.assign(code_begin, code_begin + data.M);
      entries[i] = std::move(entry);
    }
    stats.insert_entry_ms += entry_timer.ElapsedMillis();
    Timer commit_timer;
    CommitPendingLocked(&data, centroids.data(), &entries);
    stats.insert_commit_ms += commit_timer.ElapsedMillis();
    data.last_ingest_profiling.commit_us = stats.insert_commit_ms * 1000.0;
    stats.insert_ms += stats.insert_assignment_us / 1000.0 +
                       stats.insert_encode_ms + stats.insert_entry_ms +
                       stats.insert_commit_ms;
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
    uint32_t list_id = 0;
    if (!data.doc_to_list.Get(doc_id, &list_id) || list_id >= data.lists.size()) {
      return Status::NotFound("doc_id not found");
    }
    for (const auto& entry : data.lists[static_cast<size_t>(list_id)]) {
      if (entry.doc_id == doc_id) {
        if (entry.pq_code.empty()) {
          return Status::NotFound("doc has no pq code");
        }
        return entry.pq_code;
      }
    }
    return Status::NotFound("doc_id not found");
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

  Result<std::optional<PreparedPQPayload>> PreparePartitionPQCodes(
      const VersionSet& route_versions,
      const std::vector<uint32_t>& partition_ids,
      const std::vector<AlignedVector<VectorRecord>>& replacement_records) const override {
    if (partition_ids.size() != replacement_records.size()) {
      return Status::InvalidArgument(
          "PreparePartitionPQCodes: partition/record shape mismatch");
    }

    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    if (!data.use_pq) {
      return std::optional<PreparedPQPayload>{};
    }
    Status encoding_state = ValidatePQEncodingState(data);
    if (!encoding_state.ok()) {
      return encoding_state;
    }

    PreparedPQPayload payload;
    payload.context = MakePQEncodingContext(data);
    payload.partition_codes.resize(partition_ids.size());
    for (size_t partition_pos = 0; partition_pos < partition_ids.size(); ++partition_pos) {
      const uint32_t partition_id = partition_ids[partition_pos];
      if (partition_id >= data.nlist) {
        return Status::InvalidArgument(
            "PreparePartitionPQCodes: partition_id out of range");
      }
      const auto& records = replacement_records[partition_pos];
      if (records.size() > std::numeric_limits<size_t>::max() /
                               static_cast<size_t>(data.M)) {
        return Status::InvalidArgument(
            "PreparePartitionPQCodes: prepared code buffer is too large");
      }
      for (const auto& rec : records) {
        if (static_cast<uint32_t>(rec.x.size()) != data.dim) {
          return Status::InvalidArgument(
              "PreparePartitionPQCodes: record dim mismatch");
        }
      }

      PreparedPartitionPQCodes& prepared = payload.partition_codes[partition_pos];
      prepared.record_count = static_cast<uint64_t>(records.size());
      prepared.code_size = data.M;
      prepared.final_codes.resize(records.size() * static_cast<size_t>(data.M));

      std::vector<const uint8_t*> reusable_codes(records.size(), nullptr);
      if (data.pq_code_reuse_safe) {
        Timer reuse_lookup_timer;
        std::unordered_map<DocId, const uint8_t*> old_code_by_doc;
        const bool has_compact_codes =
            HasContiguousPQCodesForList(data, partition_id) &&
            HasCompactDocIdsForList(data, partition_id);
        if (has_compact_codes) {
          const auto& old_doc_ids =
              data.doc_ids_by_list[static_cast<size_t>(partition_id)];
          const auto& old_codes =
              data.pq_codes_by_list[static_cast<size_t>(partition_id)];
          old_code_by_doc.reserve(old_doc_ids.size());
          for (size_t row = 0; row < old_doc_ids.size(); ++row) {
            old_code_by_doc.emplace(
                old_doc_ids[row],
                old_codes.data() + row * static_cast<size_t>(data.M));
          }
        } else {
          const auto& old_entries = data.lists[static_cast<size_t>(partition_id)];
          old_code_by_doc.reserve(old_entries.size());
          for (const auto& entry : old_entries) {
            if (entry.pq_code.size() == data.M) {
              old_code_by_doc.emplace(entry.doc_id, entry.pq_code.data());
            }
          }
        }
        for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
          const auto old_it = old_code_by_doc.find(records[record_pos].doc_id);
          if (old_it != old_code_by_doc.end()) {
            reusable_codes[record_pos] = old_it->second;
          }
        }
        payload.copy_or_reuse_us += reuse_lookup_timer.ElapsedMicros();
      }

      Timer encode_timer;
      for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
        if (reusable_codes[record_pos] != nullptr) {
          continue;
        }
        uint8_t* code_out = prepared.final_codes.data() +
                            record_pos * static_cast<size_t>(data.M);
        EncodePQForTargetPartition(
            data, partition_id, records[record_pos].x, code_out);
        ++payload.codes_reencoded;
      }
      payload.encode_us += encode_timer.ElapsedMicros();

      Timer reuse_copy_timer;
      for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
        if (reusable_codes[record_pos] == nullptr) {
          continue;
        }
        const size_t destination_offset = record_pos * static_cast<size_t>(data.M);
        std::copy_n(reusable_codes[record_pos],
                    data.M,
                    prepared.final_codes.data() + destination_offset);
        ++payload.codes_reused;
      }
      payload.copy_or_reuse_us += reuse_copy_timer.ElapsedMicros();
    }
    return std::optional<PreparedPQPayload>{std::move(payload)};
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
    const bool has_prepared_pq = patch.prepared_pq.has_value();
    if (data.use_pq) {
      Status encoding_state = ValidatePQEncodingState(data);
      if (!encoding_state.ok()) {
        return encoding_state;
      }
    }
    if (has_prepared_pq) {
      const PreparedPQPayload& prepared = *patch.prepared_pq;
      if (!data.use_pq || !MatchesPQEncodingContext(data, prepared.context)) {
        return Status::InvalidArgument(
            "CommitPartitionPatch: prepared PQ encoding context expired");
      }
      if (prepared.partition_codes.size() != patch.partition_ids.size()) {
        return Status::InvalidArgument(
            "CommitPartitionPatch: prepared PQ partition shape mismatch");
      }
    }

    std::vector<uint8_t> patch_partitions(data.nlist, 0);
    for (uint32_t partition_id : patch.partition_ids) {
      if (partition_id >= data.nlist) {
        return Status::InvalidArgument("CommitPartitionPatch: partition_id out of range");
      }
      if (patch_partitions[static_cast<size_t>(partition_id)] != 0) {
        return Status::InvalidArgument("CommitPartitionPatch: duplicate partition_id");
      }
      patch_partitions[static_cast<size_t>(partition_id)] = 1;
    }

    std::unordered_set<DocId> patch_doc_ids;
    size_t patch_record_count = 0;
    for (const auto& records : patch.replacement_records) {
      patch_record_count += records.size();
    }
    patch_doc_ids.reserve(patch_record_count);
    for (size_t i = 0; i < patch.partition_ids.size(); ++i) {
      const auto& records = patch.replacement_records[i];
      if (has_prepared_pq) {
        const PreparedPartitionPQCodes& prepared_partition =
            patch.prepared_pq->partition_codes[i];
        if (prepared_partition.record_count != static_cast<uint64_t>(records.size()) ||
            prepared_partition.code_size != data.M ||
            records.size() > std::numeric_limits<size_t>::max() /
                                 static_cast<size_t>(data.M) ||
            prepared_partition.final_codes.size() !=
                records.size() * static_cast<size_t>(data.M)) {
          return Status::InvalidArgument(
              "CommitPartitionPatch: prepared PQ record shape mismatch");
        }
      }
      for (const auto& rec : records) {
        const bool dense_vector_required = !has_prepared_pq;
        if ((dense_vector_required && static_cast<uint32_t>(rec.x.size()) != data.dim) ||
            (has_prepared_pq &&
             (rec.dim != data.dim || rec.ivf_id != patch.partition_ids[i]))) {
          return Status::InvalidArgument(
              "CommitPartitionPatch: record shape or target mismatch");
        }
        if (!patch_doc_ids.insert(rec.doc_id).second) {
          return Status::AlreadyExists("CommitPartitionPatch: duplicate doc_id in patch");
        }
        uint32_t existing_list = 0;
        if (data.doc_to_list.Get(rec.doc_id, &existing_list) &&
            patch_partitions[static_cast<size_t>(existing_list)] == 0) {
          return Status::AlreadyExists(
              "CommitPartitionPatch: doc_id collides with unaffected partitions");
        }
      }
    }
    if (has_prepared_pq &&
        patch.prepared_pq->codes_reused + patch.prepared_pq->codes_reencoded !=
            patch_record_count) {
      return Status::InvalidArgument(
          "CommitPartitionPatch: prepared PQ profiling count mismatch");
    }

    // All fallible validation of the prepared representation is complete.
    // No persistent list has been changed above this point.
    data.last_patch_pq_reencode_ms = 0.0;
    data.last_patch_profiling = PatchProfiling{};
    data.last_patch_profiling.patch_records =
        static_cast<uint64_t>(patch_record_count);

    double pq_code_assignment_us = 0.0;
    double pq_code_copy_or_reuse_us = 0.0;
    double pq_list_flatten_us = 0.0;
    uint64_t pq_codes_reused = 0;
    uint64_t pq_codes_reencoded = 0;
    bool mutation_started = false;
    if (has_prepared_pq) {
      pq_code_assignment_us = patch.prepared_pq->encode_us;
      pq_code_copy_or_reuse_us = patch.prepared_pq->copy_or_reuse_us;
      pq_codes_reused = patch.prepared_pq->codes_reused;
      pq_codes_reencoded = patch.prepared_pq->codes_reencoded;
    }
    for (size_t i = 0; i < patch.partition_ids.size(); ++i) {
      const uint32_t partition_id = patch.partition_ids[i];
      const auto& records = patch.replacement_records[i];
      auto& dst = data.lists[static_cast<size_t>(partition_id)];
      std::unordered_map<DocId, const ListEntry*> old_by_doc;
      std::vector<const ListEntry*> reusable;
      if (data.use_pq && !has_prepared_pq) {
        reusable.assign(records.size(), nullptr);
      }
      if (data.use_pq && !has_prepared_pq && data.pq_code_reuse_safe) {
        Timer pq_reuse_timer;
        old_by_doc.reserve(dst.size());
        for (const auto& old_entry : dst) {
          old_by_doc.emplace(old_entry.doc_id, &old_entry);
        }
        for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
          auto old_it = old_by_doc.find(records[record_pos].doc_id);
          if (old_it != old_by_doc.end() && old_it->second->pq_code.size() == data.M) {
            reusable[record_pos] = old_it->second;
          }
        }
        pq_code_copy_or_reuse_us += pq_reuse_timer.ElapsedMicros();
      }
      std::vector<uint8_t> prepared_codes;
      const std::vector<uint8_t>* final_codes = nullptr;
      if (has_prepared_pq) {
        final_codes = &patch.prepared_pq->partition_codes[i].final_codes;
      } else if (data.use_pq) {
        prepared_codes.resize(records.size() * static_cast<size_t>(data.M));
        Timer pq_reencode_timer;
        for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
          if (reusable[record_pos] != nullptr) {
            continue;
          }
          uint8_t* code_out =
              prepared_codes.data() + record_pos * static_cast<size_t>(data.M);
          EncodePQForTargetPartition(
              data, partition_id, records[record_pos].x, code_out);
          ++pq_codes_reencoded;
        }
        pq_code_assignment_us += pq_reencode_timer.ElapsedMicros();

        Timer pq_copy_timer;
        for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
          const ListEntry* old_entry = reusable[record_pos];
          if (old_entry == nullptr) {
            continue;
          }
          std::copy(old_entry->pq_code.begin(),
                    old_entry->pq_code.end(),
                    prepared_codes.begin() +
                        static_cast<std::ptrdiff_t>(record_pos * static_cast<size_t>(data.M)));
          ++pq_codes_reused;
        }
        pq_code_copy_or_reuse_us += pq_copy_timer.ElapsedMicros();
        final_codes = &prepared_codes;
      }

      // Legacy encoding and reuse copying also finish before the first list
      // mutation, so every explicit failure path remains preflight-only.
      if (!mutation_started) {
        data.mutation_generation = next_mutation_generation_++;
        mutation_started = true;
      }
      dst.clear();
      dst.reserve(records.size());
      Timer prepared_copy_timer;
      for (size_t record_pos = 0; record_pos < records.size(); ++record_pos) {
        const auto& rec = records[record_pos];
        ListEntry entry;
        entry.doc_id = rec.doc_id;
        entry.versions = rec.versions;
        entry.versions.index_version = data.version;
        if (data.use_pq) {
          const uint8_t* code_begin =
              final_codes->data() + record_pos * static_cast<size_t>(data.M);
          entry.pq_code.assign(code_begin, code_begin + data.M);
        } else {
          entry.vector = rec.x;
          entry.norm = entry.vector.squaredNorm();
        }
        dst.push_back(std::move(entry));
      }
      if (has_prepared_pq) {
        pq_code_copy_or_reuse_us += prepared_copy_timer.ElapsedMicros();
      }
      if (data.use_pq) {
        Timer pq_list_flatten_timer;
        RebuildListPQCodes(&data, partition_id);
        pq_list_flatten_us += pq_list_flatten_timer.ElapsedMicros();
      }
    }

    RebuildDocMapLocked(&data);
    data.last_patch_pq_reencode_ms = pq_code_assignment_us / 1000.0;
    data.last_patch_profiling.pq_code_assignment_us = pq_code_assignment_us;
    data.last_patch_profiling.pq_code_copy_or_reuse_us = pq_code_copy_or_reuse_us;
    data.last_patch_profiling.pq_list_flatten_us = pq_list_flatten_us;
    data.last_patch_profiling.pq_codes_reused = pq_codes_reused;
    data.last_patch_profiling.pq_codes_reencoded = pq_codes_reencoded;
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

  Result<PatchProfiling> GetLastPatchProfiling(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    return it->second->last_patch_profiling;
  }

  Result<IngestProfiling> GetLastIngestProfiling(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    return it->second->last_ingest_profiling;
  }

  Result<IVFMemoryUsage> EstimateMemoryUsage(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    const IndexData& data = *it->second;
    auto matrix_bytes = [](const MatrixRM& matrix) -> uint64_t {
      return static_cast<uint64_t>(matrix.rows()) *
             static_cast<uint64_t>(matrix.cols()) * sizeof(float);
    };
    auto vector_float_bytes = [](const Eigen::VectorXf& vector) -> uint64_t {
      return static_cast<uint64_t>(vector.size()) * sizeof(float);
    };

    IVFMemoryUsage usage;
    usage.ntotal = data.ntotal;
    usage.nlist = data.nlist;
    usage.routing_centroids_bytes = matrix_bytes(data.routing_centroids);
    usage.routing_centroid_norms_bytes =
        vector_float_bytes(data.routing_centroid_norms);

    usage.pq_codebooks_bytes =
        static_cast<uint64_t>(data.pq_codebooks.capacity()) * sizeof(MatrixRM);
    for (const MatrixRM& codebook : data.pq_codebooks) {
      usage.pq_codebooks_bytes += matrix_bytes(codebook);
    }
    usage.pq_codebooks_soa_bytes =
        static_cast<uint64_t>(data.pq_codebooks_soa.capacity()) * sizeof(MatrixRM);
    for (const MatrixRM& codebook_soa : data.pq_codebooks_soa) {
      usage.pq_codebooks_soa_bytes += matrix_bytes(codebook_soa);
    }
    usage.pq_precomputed_table_bytes =
        static_cast<uint64_t>(data.pq_precomputed_table.capacity()) * sizeof(float);
    usage.pq_counts_bytes =
        static_cast<uint64_t>(data.pq_counts.capacity()) * sizeof(std::vector<uint64_t>);
    for (const auto& counts : data.pq_counts) {
      usage.pq_counts_bytes +=
          static_cast<uint64_t>(counts.capacity()) * sizeof(uint64_t);
    }

    usage.lists_vector_bytes =
        static_cast<uint64_t>(data.lists.capacity()) * sizeof(AlignedVector<ListEntry>);
    for (const auto& list : data.lists) {
      usage.list_entry_struct_bytes +=
          static_cast<uint64_t>(list.capacity()) * sizeof(ListEntry);
      for (const ListEntry& entry : list) {
        usage.list_entry_vectors_bytes +=
            static_cast<uint64_t>(entry.vector.size()) * sizeof(float);
        usage.list_entry_vectors_bytes +=
            static_cast<uint64_t>(entry.pq_code.capacity()) * sizeof(uint8_t);
      }
    }

    usage.compact_doc_ids_bytes =
        static_cast<uint64_t>(data.doc_ids_by_list.capacity()) *
        sizeof(std::vector<DocId>);
    for (const auto& doc_ids : data.doc_ids_by_list) {
      usage.compact_doc_ids_bytes +=
          static_cast<uint64_t>(doc_ids.capacity()) * sizeof(DocId);
    }
    usage.compact_pq_codes_bytes =
        static_cast<uint64_t>(data.pq_codes_by_list.capacity()) *
        sizeof(std::vector<uint8_t>);
    for (const auto& codes : data.pq_codes_by_list) {
      usage.compact_pq_codes_bytes +=
          static_cast<uint64_t>(codes.capacity()) * sizeof(uint8_t);
    }
    usage.compact_pq_codes_soa_bytes =
        static_cast<uint64_t>(data.pq_codes_soa_by_list.capacity()) *
        sizeof(ListPQCodesSoA);
    for (const ListPQCodesSoA& soa : data.pq_codes_soa_by_list) {
      usage.compact_pq_codes_soa_bytes +=
          static_cast<uint64_t>(soa.codes.capacity()) * sizeof(uint8_t);
    }
    usage.doc_to_list_bytes =
        static_cast<uint64_t>(data.doc_to_list.list_ids.capacity()) * sizeof(uint32_t);

    usage.total_bytes =
        usage.routing_centroids_bytes +
        usage.routing_centroid_norms_bytes +
        usage.pq_codebooks_bytes +
        usage.pq_codebooks_soa_bytes +
        usage.pq_precomputed_table_bytes +
        usage.pq_counts_bytes +
        usage.lists_vector_bytes +
        usage.list_entry_struct_bytes +
        usage.list_entry_vectors_bytes +
        usage.compact_doc_ids_bytes +
        usage.compact_pq_codes_bytes +
        usage.compact_pq_codes_soa_bytes +
        usage.doc_to_list_bytes;
    return usage;
  }

  Result<IVFBuildProfiling> GetBuildProfiling(
      const VersionSet& route_versions) const override {
    std::shared_lock lock(mu_);
    auto it = data_map_.find(route_versions.index_version);
    if (it == data_map_.end()) {
      return Status::NotFound("Index version not built");
    }
    return it->second->build_profiling;
  }

  Result<std::vector<uint8_t>> Serialize() const override {
    std::shared_lock lock(mu_);
    if (latest_version_ == 0) {
      return Status::InvalidArgument("Serialize: index not built");
    }
    auto it = data_map_.find(latest_version_);
    if (it == data_map_.end() || !it->second) {
      return Status::NotFound("Serialize: latest index version missing");
    }

    std::vector<uint8_t> bytes;
    AppendPod(&bytes, kIVFSerializeMagic);
    AppendPod(&bytes, kIVFSerializeFormatVersion);
    AppendPod(&bytes, latest_version_);
    AppendPod(&bytes, next_version_);
    const uint32_t data_count = 1;
    AppendPod(&bytes, data_count);
    AppendIndexData(&bytes, *it->second);
    return bytes;
  }

  Status Deserialize(const std::vector<uint8_t>& bytes) override {
    size_t offset = 0;
    uint32_t magic = 0;
    Status status = ReadPod(bytes, &offset, &magic);
    if (!status.ok()) {
      return status;
    }
    if (magic != kIVFSerializeMagic) {
      return Status::InvalidArgument("Bad IVF magic");
    }

    uint32_t format_version = 0;
    status = ReadPod(bytes, &offset, &format_version);
    if (!status.ok()) {
      return status;
    }
    if (format_version != kIVFSerializeFormatVersion) {
      return Status::InvalidArgument("Unsupported IVF payload format version");
    }

    VersionId latest_version = 0;
    VersionId next_version = 1;
    uint32_t data_count = 0;
    status = ReadPod(bytes, &offset, &latest_version);
    if (!status.ok()) return status;
    status = ReadPod(bytes, &offset, &next_version);
    if (!status.ok()) return status;
    status = ReadPod(bytes, &offset, &data_count);
    if (!status.ok()) return status;
    if (latest_version == 0 || data_count == 0) {
      return Status::InvalidArgument("Invalid IVF payload header");
    }

    std::unordered_map<VersionId, std::unique_ptr<IndexData>> loaded;
    VersionId max_version = 0;
    for (uint32_t i = 0; i < data_count; ++i) {
      auto data_res = ReadIndexData(bytes, &offset);
      if (!data_res.ok()) {
        return data_res.status();
      }
      std::unique_ptr<IndexData> data = std::move(data_res.value());
      if (!loaded.emplace(data->version, std::move(data)).second) {
        return Status::AlreadyExists("Duplicate index version in IVF payload");
      }
    }
    if (offset != bytes.size()) {
      return Status::InvalidArgument("Trailing bytes in IVF payload");
    }
    if (loaded.find(latest_version) == loaded.end()) {
      return Status::InvalidArgument("Latest index version missing in IVF payload");
    }

    for (auto& kv : loaded) {
      IndexData* data = kv.second.get();
      max_version = std::max(max_version, data->version);
      if (data->use_pq) {
        data->doc_ids_by_list.clear();
        data->doc_ids_by_list.resize(data->lists.size());
        data->pq_codes_by_list.clear();
        data->pq_codes_by_list.resize(data->lists.size());
        data->pq_codes_soa_by_list.clear();
        if (data->pq_codes_subquantizer_major) {
          data->pq_codes_soa_by_list.resize(data->lists.size());
        }
        for (uint32_t list_id = 0; list_id < data->nlist; ++list_id) {
          RebuildListPQCodes(data, list_id);
        }
        BuildPrecomputedTable(data);
      }
      RebuildDocMapLocked(data);
    }

    std::unique_lock lock(mu_);
    for (auto& kv : loaded) {
      kv.second->mutation_generation = next_mutation_generation_++;
    }
    data_map_ = std::move(loaded);
    latest_version_ = latest_version;
    next_version_ = std::max(next_version, max_version + 1);
    return Status::OK();
  }

 private:
  struct RemovedDoc {
    DocId doc_id{0};
    uint32_t list_id{0};
    ListEntry entry;
  };

  void RebuildDocMapLocked(IndexData* data) {
    data->doc_to_list.Clear();
    data->ntotal = 0;

    bool has_docs = false;
    DocId min_doc_id = 0;
    DocId max_doc_id = 0;
    for (const auto& list : data->lists) {
      for (const auto& entry : list) {
        if (!has_docs) {
          min_doc_id = entry.doc_id;
          max_doc_id = entry.doc_id;
          has_docs = true;
        } else {
          min_doc_id = std::min(min_doc_id, entry.doc_id);
          max_doc_id = std::max(max_doc_id, entry.doc_id);
        }
      }
    }
    if (!has_docs) {
      return;
    }

    data->doc_to_list.EnsureRange(min_doc_id, max_doc_id);
    for (uint32_t list_id = 0; list_id < data->nlist; ++list_id) {
      for (const auto& entry : data->lists[static_cast<size_t>(list_id)]) {
        data->doc_to_list.Set(entry.doc_id, list_id);
        ++data->ntotal;
      }
    }
  }

  Status ValidateRecordsForInsertLocked(const IndexData& data,
                                        const AlignedVector<VectorRecord>& recs) const {
    std::unordered_set<DocId> batch_ids;
    batch_ids.reserve(recs.size());
    for (const auto& rec : recs) {
      if (static_cast<uint32_t>(rec.x.size()) != data.dim) {
        return Status::InvalidArgument("Record dim mismatch");
      }
      if (data.doc_to_list.Contains(rec.doc_id)) {
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
      if (!data.doc_to_list.Contains(doc_id)) {
        return Status::NotFound("delete doc_id not found in IVF");
      }
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
      uint32_t list_id = 0;
      if (!data->doc_to_list.Get(doc_id, &list_id)) {
        return Status::NotFound("delete doc_id list mapping not found");
      }
      if (list_id >= data->lists.size()) {
        return Status::InvalidArgument("delete list_id out of range");
      }
      auto& list = data->lists[static_cast<size_t>(list_id)];
      size_t pos = list.size();
      for (size_t i = 0; i < list.size(); ++i) {
        if (list[i].doc_id == doc_id) {
          pos = i;
          break;
        }
      }
      if (pos == list.size()) {
        return Status::NotFound("delete doc_id not found in list");
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
        data->doc_to_list.Set(list[pos].doc_id, list_id);
      }
      list.pop_back();
      data->doc_to_list.Erase(doc_id);
      if (data->ntotal > 0) {
        --data->ntotal;
      }
    }
    data->doc_to_list.MaybeCompact();
    return Status::OK();
  }

  void CommitPendingLocked(IndexData* data,
                           const int* centroids,
                           AlignedVector<ListEntry>* entries) {
    if (data->use_pq && data->doc_ids_by_list.size() != data->lists.size()) {
      data->doc_ids_by_list.resize(data->lists.size());
    }
    if (data->use_pq && data->pq_codes_by_list.size() != data->lists.size()) {
      data->pq_codes_by_list.resize(data->lists.size());
    }
    if (data->use_pq && data->pq_codes_subquantizer_major &&
        data->pq_codes_soa_by_list.size() != data->lists.size()) {
      data->pq_codes_soa_by_list.resize(data->lists.size());
    }
    if (!entries->empty()) {
      DocId min_doc_id = (*entries)[0].doc_id;
      DocId max_doc_id = min_doc_id;
      for (const auto& entry : *entries) {
        min_doc_id = std::min(min_doc_id, entry.doc_id);
        max_doc_id = std::max(max_doc_id, entry.doc_id);
      }
      data->doc_to_list.EnsureRange(min_doc_id, max_doc_id);
    }
    for (size_t i = 0; i < entries->size(); ++i) {
      const uint32_t list_id = static_cast<uint32_t>(centroids[i]);
      auto& list = data->lists[static_cast<size_t>(list_id)];
      const size_t rows_before_push = list.size();
      list.push_back(std::move((*entries)[i]));
      data->doc_to_list.Set(list.back().doc_id, list_id);
      if (data->use_pq) {
        auto& flat = data->pq_codes_by_list[static_cast<size_t>(list_id)];
        if (list.back().pq_code.size() == data->M) {
          const size_t expected_before_push = rows_before_push * data->M;
          if (flat.size() == expected_before_push) {
            flat.insert(flat.end(), list.back().pq_code.begin(), list.back().pq_code.end());
            AppendListPQCaches(data, list_id, rows_before_push, list.back());
          } else {
            RebuildListPQCodes(data, list_id);
          }
        } else {
          if (list_id < data->doc_ids_by_list.size()) {
            data->doc_ids_by_list[static_cast<size_t>(list_id)].clear();
          }
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
  uint64_t next_mutation_generation_{1};
};

}  // namespace

std::shared_ptr<IVFIndex> CreateIVFIndex() { return std::make_shared<KMeansIVFIndex>(); }

}  // namespace ann
