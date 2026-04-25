#pragma once

#include <memory>
#include <vector>

#include "common/result.h"
#include "common/types.h"
#include "index/ivf.h"

namespace ann {

struct FrozenDelta {
  AlignedVector<VectorRecord> records;
};

struct DeltaAssignment {
  VectorRecord record;
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
};

struct DistributionSummary {
  double min{0.0};
  double p50{0.0};
  double p90{0.0};
  double max{0.0};
};

struct MergeReport {
  uint32_t frozen_records{0};
  uint32_t patch_partitions{0};
  uint32_t append_partitions{0};
  uint32_t recluster_partitions{0};
  double codebook_rebuild_ms{0.0};
  DistributionSummary score_summary;
  DistributionSummary residual_summary;
  DistributionSummary growth_summary;
  std::vector<PartitionStats> stats;
  PartitionScoreResult scoring;
};

Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions);

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
    const AlignedVector<DeltaAssignment>& delta_partition_records);

Result<AlignedVector<VectorRecord>> merge_partition_recluster(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const AlignedVector<DeltaAssignment>& delta_partition_records,
    const MergeOptions& options);

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
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
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const MergeOptions& options);

}  // namespace ann
