#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/result.h"
#include "common/types.h"
#include "common/vector_accessor.h"
#include "index/ivf.h"

namespace ann {

struct FrozenDelta {
  AlignedVector<VectorRecord> records;
};

struct DeltaAssignment {
  uint32_t frozen_index{0};
  uint32_t main_partition{0};
  float residual_dist{0.0f};
};

using PartitionAssignments = std::vector<AlignedVector<DeltaAssignment>>;

struct PartitionStats {
  uint32_t partition_id{0};
  uint32_t insert_count{0};
  uint32_t main_partition_size{0};
  double growth_ratio{0.0};
  double avg_residual_dist{0.0};
};

struct PartitionDecision {
  uint32_t partition_id{0};
  double score{0.0};
  bool use_recluster{false};
};

struct PartitionScoreResult {
  std::vector<PartitionDecision> decisions;
  std::vector<uint32_t> append_partitions;
  std::vector<uint32_t> recluster_partitions;
};

struct MergeOptions {
  double alpha{1.0};
  double beta{1.0};
  double recluster_threshold{1.0};
  uint32_t local_recluster_k{2};
  uint32_t local_kmeans_iterations{5};
  bool enable_local_centroid_refine{false};
  std::string assignment_mode{"balanced_append"};
  uint32_t assignment_top_r{4};
  double assignment_gamma{1.05};
  double assignment_hard_cap_ratio{1.5};
  double assignment_lambda{0.2};
};

struct DistributionSummary {
  double min{0.0};
  double p50{0.0};
  double p90{0.0};
  double max{0.0};
};

struct MergeProfiling {
  uint32_t effective_nlist{0};
  uint64_t frozen_records{0};
  uint64_t assignment_descriptor_records{0};
  uint64_t assignment_full_vector_copy_bytes{0};
  uint64_t frozen_payload_records_moved{0};
  uint64_t vector_accessor_materialize_calls{0};
  uint64_t vector_accessor_materialized_rows{0};
  uint64_t vector_accessor_materialized_bytes{0};
  uint32_t vector_accessor_max_materialize_rows{0};
  uint64_t seed_partitions{0};
  uint64_t neighborhoods{0};
  uint64_t main_records_loaded{0};
  uint64_t pooled_records{0};
  uint64_t repartitioned_records{0};
  uint64_t patch_records{0};
  uint64_t patch_dense_vector_bytes{0};
  uint64_t patch_retained_vector_bytes{0};
  uint64_t patch_elided_vector_bytes{0};
  uint64_t patch_final_pq_code_bytes{0};
  uint64_t pq_codes_reused{0};
  uint64_t pq_codes_reencoded{0};
  uint64_t prepare_pq_codes_reused{0};
  uint64_t prepare_pq_codes_reencoded{0};

  double merge_delta_to_main_assignment_us{0.0};
  double merge_assignment_distance_us{0.0};
  double merge_assignment_top_r_us{0.0};
  double merge_assignment_balance_us{0.0};
  double merge_assignment_materialize_us{0.0};
  double stats_us{0.0};
  double scoring_us{0.0};
  double top_r_neighbor_us{0.0};
  double fetch_main_records_us{0.0};
  double repartition_pool_us{0.0};
  double repartition_distance_us{0.0};
  double repartition_candidate_selection_us{0.0};
  double repartition_sort_us{0.0};
  double patch_prepare_us{0.0};
  double prepare_pq_encode_us{0.0};
  double commit_us{0.0};
  double prepare_publish_us{0.0};
  double commit_validation_us{0.0};
  double commit_docmap_us{0.0};
  double commit_partition_swap_us{0.0};
  std::chrono::steady_clock::time_point compute_done_at{};
  std::chrono::steady_clock::time_point commit_wait_start_at{};
  std::chrono::steady_clock::time_point commit_lock_acquired_at{};
  std::chrono::steady_clock::time_point commit_done_at{};
  double pq_code_assignment_us{0.0};
  double pq_code_copy_or_reuse_us{0.0};
  double pq_list_flatten_us{0.0};
};

struct MergeReport {
  uint32_t frozen_records{0};
  uint32_t patch_partitions{0};
  uint32_t append_partitions{0};
  uint32_t recluster_partitions{0};
  double merge_compute_ms{0.0};
  double codebook_rebuild_ms{0.0};
  double moved_delta_ratio{0.0};
  double avg_assignment_dist_ratio{1.0};
  double max_assignment_dist_ratio{1.0};
  double imbalance_before{0.0};
  double imbalance_after{0.0};
  MergeProfiling profiling;
  DistributionSummary score_summary;
  DistributionSummary residual_summary;
  DistributionSummary growth_summary;
  std::vector<PartitionStats> stats;
  PartitionScoreResult scoring;
};

Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions);
Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions,
                                 const MatrixRM& base_vectors);
Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions,
                                 const VectorAccessor& vector_accessor);

Result<PartitionAssignments> assign_delta_to_main_centroids(
    const FrozenDelta& frozen_delta,
    Eigen::Ref<const MatrixRM> main_centroids);

Result<std::vector<PartitionStats>> compute_partition_stats(
    const PartitionAssignments& assignments,
    const std::vector<uint32_t>& main_partition_sizes);

Result<PartitionScoreResult> score_partitions(const std::vector<PartitionStats>& stats,
                                              double alpha,
                                              double beta,
                                              double recluster_threshold);

Result<AlignedVector<VectorRecord>> merge_partition_append(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const FrozenDelta& frozen_delta,
    const AlignedVector<DeltaAssignment>& delta_partition_records);

Result<AlignedVector<VectorRecord>> merge_partition_recluster(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const FrozenDelta& frozen_delta,
    const AlignedVector<DeltaAssignment>& delta_partition_records,
    const MergeOptions& options);

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MergeOptions& options);
Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MatrixRM& base_vectors,
    const MergeOptions& options);
Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor& vector_accessor,
    const MergeOptions& options);

Status commit_partition_patch(const std::shared_ptr<IVFIndex>& main_ivf,
                              const VersionSet& main_versions,
                              const PartitionPatch& patch);

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MergeOptions& options);
Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MatrixRM& base_vectors,
                                                 const MergeOptions& options);
Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const VectorAccessor& vector_accessor,
                                                 const MergeOptions& options);

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const MergeOptions& options);
Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const MatrixRM& base_vectors,
                                                 const MergeOptions& options);
Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const VectorAccessor& vector_accessor,
                                                 const MergeOptions& options);

}  // namespace ann
