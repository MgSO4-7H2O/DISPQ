#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct PQParams {
  bool enable{false};
  uint32_t M{0};
  uint32_t nbits{8};
  bool residual{true};
};

struct IVFParams {
  uint32_t nlist{1024};
  uint32_t dim{0};
  uint32_t kmeans_iterations{20};
  PQParams pq;
  // Store/use transposed PQ codebooks (dimension-major, dsub x Ks) for
  // codeword assignment and LUT construction.
  bool pq_codebook_dimension_major{true};
  // Store/use PQ codes in subquantizer-major order for ADC scans.
  bool pq_codes_subquantizer_major{true};
  bool use_fixed_routing_centroids{false};
  MatrixRM fixed_routing_centroids;
  bool use_fixed_pq_codebooks{false};
  std::vector<MatrixRM> fixed_pq_codebooks;
  std::vector<std::vector<uint64_t>> fixed_pq_counts;
  double fixed_pq_baseline_nqe{-1.0};
  double fixed_pq_ema_nqe{-1.0};
  bool defer_pq_stats_to_add{false};
};

struct OnlinePQUpdateOptions {
  bool enable{false};
  double qe_ratio_threshold{1.05};
  double ema_alpha{0.1};
  double nqe_eps{1e-6};
  bool warmup_enable{false};
  uint32_t warmup_batches{0};
  uint32_t force_update_interval{0};
  bool partial_top_alpha{false};
  double partial_alpha{1.0};
  bool partial_top_lambda{false};
  double partial_lambda{1.0};
  bool reencode_batch_after_update{true};
};

struct OnlinePQUpdateStats {
  bool use_online_pq{false};
  bool trigger_update{false};
  bool updated_codebook{false};
  bool reencoded_batch{false};
  bool in_warmup{false};
  uint32_t warmup_batches_left{0};
  uint32_t processed_vectors{0};
  uint32_t deleted_vectors{0};
  uint32_t updated_subspaces{0};
  uint32_t updated_codewords{0};
  double nqe_batch{0.0};
  double nqe_ema{0.0};
  double nqe_baseline{0.0};
  double qe_ratio{1.0};
  double codebook_drift_l2{0.0};
  // Wall-clock time spent assigning inserted records to this IVF's own lists.
  // This is distinct from merge-time frozen-delta-to-main assignment.
  double insert_assignment_us{0.0};
  double record_build_ms{0.0};
  double insert_ms{0.0};
  double insert_encode_ms{0.0};
  double insert_entry_ms{0.0};
  double insert_commit_ms{0.0};
  double delete_ms{0.0};
  double onlinepq_stats_ms{0.0};
  double codebook_update_ms{0.0};
  double reencode_ms{0.0};
  double maintenance_ms{0.0};
};

struct IngestProfiling {
  uint64_t records{0};
  double record_construction_us{0.0};
  double validation_us{0.0};
  double assignment_us{0.0};
  double encode_us{0.0};
  double deferred_pq_stats_us{0.0};
  double commit_us{0.0};
};

struct PQRuntimeState {
  bool use_pq{false};
  bool pq_residual{true};
  uint32_t M{0};
  uint32_t Ks{0};
  uint32_t dsub{0};
  std::vector<MatrixRM> codebooks;
  std::vector<std::vector<uint64_t>> counts;
  double nqe_baseline{0.0};
  double nqe_ema{0.0};
  double qe_ratio{1.0};
  uint64_t ntotal{0};
};

struct IVFMemoryUsage {
  uint64_t total_bytes{0};
  uint64_t routing_centroids_bytes{0};
  uint64_t routing_centroid_norms_bytes{0};
  uint64_t pq_codebooks_bytes{0};
  uint64_t pq_codebooks_soa_bytes{0};
  uint64_t pq_precomputed_table_bytes{0};
  uint64_t pq_counts_bytes{0};
  uint64_t lists_vector_bytes{0};
  uint64_t list_entry_struct_bytes{0};
  uint64_t list_entry_vectors_bytes{0};
  uint64_t compact_doc_ids_bytes{0};
  uint64_t compact_pq_codes_bytes{0};
  uint64_t compact_pq_codes_soa_bytes{0};
  uint64_t doc_to_list_bytes{0};
  uint64_t ntotal{0};
  uint32_t nlist{0};
};

struct IVFBuildProfiling {
  double build_coarse_centroid_init_us{0.0};
  double build_coarse_kmeans_us{0.0};
  double build_coarse_kmeans_assignment_us{0.0};
  double build_coarse_kmeans_update_us{0.0};
  uint64_t build_coarse_kmeans_rows{0};
  uint32_t build_coarse_kmeans_k{0};
  uint32_t build_coarse_kmeans_dim{0};
  uint32_t build_coarse_kmeans_iterations{0};
  double build_pq_routing_assignment_us{0.0};
  uint64_t build_pq_routing_assignment_rows{0};
  double build_pq_subspace_materialize_us{0.0};
  uint64_t build_pq_subspace_materialized_rows{0};
  uint64_t build_pq_subspace_materialized_bytes{0};
  uint32_t build_pq_max_live_subspaces{0};
  double build_pq_centroid_init_us{0.0};
  double build_pq_kmeans_assignment_us{0.0};
  double build_pq_kmeans_update_us{0.0};
  double build_pq_kmeans_us{0.0};
  double build_pq_training_total_us{0.0};
  double build_pq_codebook_soa_us{0.0};
  double build_pq_precomputed_table_us{0.0};
  double build_publication_us{0.0};
  uint64_t build_pq_full_residual_bytes{0};
  uint32_t build_pq_training_concurrency{0};
};

struct PQEncodingContext {
  VersionId index_version{0};
  uint64_t mutation_generation{0};
  bool use_pq{false};
  bool pq_residual{true};
  uint32_t M{0};
  uint32_t nbits{0};
  uint32_t Ks{0};
  uint32_t dsub{0};
};

struct PreparedPartitionPQCodes {
  uint64_t record_count{0};
  uint32_t code_size{0};
  // Record-major: record i owns [i * code_size, (i + 1) * code_size).
  std::vector<uint8_t> final_codes;
};

struct PreparedPQPayload {
  PQEncodingContext context;
  // One entry per PartitionPatch partition, in exactly the same order.
  std::vector<PreparedPartitionPQCodes> partition_codes;
  uint64_t codes_reused{0};
  uint64_t codes_reencoded{0};
  double encode_us{0.0};
  double copy_or_reuse_us{0.0};
};

struct PartitionPatch {
  std::vector<uint32_t> partition_ids;
  std::vector<AlignedVector<VectorRecord>> replacement_records;
  // Present only when every PQ replacement record already has its final code.
  // The payload expires if the target index mutation generation changes.
  std::optional<PreparedPQPayload> prepared_pq;
};

struct PatchProfiling {
  double pq_code_assignment_us{0.0};
  double pq_code_copy_or_reuse_us{0.0};
  double pq_list_flatten_us{0.0};
  uint64_t patch_records{0};
  uint64_t pq_codes_reused{0};
  uint64_t pq_codes_reencoded{0};
};

class IVFIndex {
 public:
  virtual ~IVFIndex() = default;

  // Optionally returns the final assignments already used by residual PQ training.
  virtual Result<VersionId> Build(Eigen::Ref<const MatrixRM> Xw,
                                  const std::vector<DocId>& ids,
                                  const IVFParams& p,
                                  VersionId index_version,
                                  std::vector<int>* routing_assignments = nullptr) = 0;

  // NTS: Adds vector records to the mutable shard.
  virtual Status Add(const AlignedVector<VectorRecord>& recs) = 0;

  // NTS: Adds one chunk and optionally consumes recs.size() known assignments.
  virtual Status AddBatch(const AlignedVector<VectorRecord>& recs,
                          bool finalize_deferred_stats,
                          const int* precomputed_assignments = nullptr) = 0;

  // NTS: Adds records and optionally applies residual OnlinePQ codebook update.
  virtual Result<OnlinePQUpdateStats> AddWithOnlinePQ(
      const AlignedVector<VectorRecord>& recs,
      const OnlinePQUpdateOptions& options) = 0;

  // NTS: Sliding-window OnlinePQ step with insertion and deletion in one update.
  virtual Result<OnlinePQUpdateStats> AddWithOnlinePQSlidingWindow(
      const AlignedVector<VectorRecord>& recs,
      const std::vector<DocId>& delete_doc_ids,
      const OnlinePQUpdateOptions& options) = 0;

  // NTS: Sliding-window OnlinePQ step with external vectors for deleted records.
  virtual Result<OnlinePQUpdateStats> AddWithOnlinePQSlidingWindowRecords(
      const AlignedVector<VectorRecord>& recs,
      const AlignedVector<VectorRecord>& delete_recs,
      const OnlinePQUpdateOptions& options) = 0;

  // TS: Searches specified versions using whitened query.
  virtual Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                                      uint32_t topk,
                                      uint32_t nprobe,
                                      const VersionSet& route_versions,
                                      uint8_t from_new,
                                      bool collect_scan_trace = false) const = 0;

  // TS: Batch search for multiple whitened queries.
  virtual Result<std::vector<SearchResult>> SearchBatch(
      Eigen::Ref<const MatrixRM> qw_batch,
      uint32_t topk,
      uint32_t nprobe,
      const VersionSet& route_versions,
      uint8_t from_new,
      bool collect_scan_trace = false) const = 0;

  virtual Result<MatrixRM> GetRoutingCentroids(const VersionSet& route_versions) const = 0;
  virtual Result<PQRuntimeState> GetPQRuntimeState(const VersionSet& route_versions) const = 0;
  virtual Result<std::vector<uint8_t>> GetDocPQCode(const VersionSet& route_versions,
                                                    DocId doc_id) const = 0;
  virtual Result<std::vector<DocId>> SnapshotDocIds(
      const VersionSet& route_versions) const = 0;
  virtual Result<AlignedVector<VectorRecord>> SnapshotRecords(
      const VersionSet& route_versions) const = 0;
  virtual Result<AlignedVector<VectorRecord>> GetPartitionRecords(
      const VersionSet& route_versions,
      uint32_t partition_id) const = 0;
  virtual Result<std::vector<DocId>> GetPartitionDocIds(
      const VersionSet& route_versions,
      uint32_t partition_id) const = 0;
  virtual Result<std::vector<uint32_t>> GetPartitionSizes(
      const VersionSet& route_versions) const = 0;
  // TS: snapshots same-partition reusable codes and encodes all remaining
  // records under one read lock. A non-PQ index returns std::nullopt.
  virtual Result<std::optional<PreparedPQPayload>> PreparePartitionPQCodes(
      const VersionSet& route_versions,
      const std::vector<uint32_t>& partition_ids,
      const std::vector<AlignedVector<VectorRecord>>& replacement_records) const = 0;
  virtual Status CommitPartitionPatch(const VersionSet& route_versions,
                                      const PartitionPatch& patch) = 0;
  virtual Result<double> GetLastPatchPQReencodeMs(
      const VersionSet& route_versions) const = 0;
  virtual Result<PatchProfiling> GetLastPatchProfiling(
      const VersionSet& route_versions) const = 0;
  virtual Result<IngestProfiling> GetLastIngestProfiling(
      const VersionSet& route_versions) const = 0;
  virtual Result<IVFMemoryUsage> EstimateMemoryUsage(
      const VersionSet& route_versions) const = 0;
  virtual Result<IVFBuildProfiling> GetBuildProfiling(
      const VersionSet& route_versions) const = 0;

  virtual Result<std::vector<uint8_t>> Serialize() const = 0;
  virtual Status Deserialize(const std::vector<uint8_t>& bytes) = 0;
};

std::shared_ptr<IVFIndex> CreateIVFIndex();

}  // namespace ann
