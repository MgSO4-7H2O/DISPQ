#include "index/merge.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace ann {
namespace {

uint32_t NearestCentroid(Eigen::Ref<const Eigen::VectorXf> x,
                         Eigen::Ref<const MatrixRM> centroids,
                         float* dist_out) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_idx = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(centroids.rows()); ++i) {
    const float dist = (x - centroids.row(static_cast<Eigen::Index>(i)).transpose()).squaredNorm();
    if (dist < best) {
      best = dist;
      best_idx = i;
    }
  }
  if (dist_out != nullptr) {
    *dist_out = best;
  }
  return best_idx;
}

}  // namespace

Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions) {
  if (!delta_ivf) {
    return Status::InvalidArgument("freeze_delta: delta_ivf is null");
  }
  auto snapshot_res = delta_ivf->SnapshotRecords(delta_versions);
  if (!snapshot_res.ok()) {
    return snapshot_res.status();
  }
  FrozenDelta out;
  out.records = std::move(snapshot_res.value());
  return out;
}

Result<PartitionAssignments> assign_delta_to_main_centroids(
    const FrozenDelta& frozen_delta,
    Eigen::Ref<const MatrixRM> main_centroids) {
  if (main_centroids.rows() == 0 || main_centroids.cols() == 0) {
    return Status::InvalidArgument("assign_delta_to_main_centroids: empty main_centroids");
  }
  PartitionAssignments assigned(static_cast<size_t>(main_centroids.rows()));
  for (const auto& rec : frozen_delta.records) {
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument(
          "assign_delta_to_main_centroids: record dim mismatch with main centroids");
    }
    float residual_dist = 0.0f;
    const uint32_t partition = NearestCentroid(rec.x, main_centroids, &residual_dist);
    DeltaAssignment item;
    item.record = rec;
    item.record.ivf_id = partition;
    item.main_partition = partition;
    item.residual_dist = residual_dist;
    assigned[static_cast<size_t>(partition)].push_back(std::move(item));
  }
  return assigned;
}

Result<std::vector<PartitionStats>> compute_partition_stats(
    const PartitionAssignments& assignments,
    const std::vector<uint32_t>& main_partition_sizes) {
  if (assignments.size() != main_partition_sizes.size()) {
    return Status::InvalidArgument(
        "compute_partition_stats: assignments size mismatch with main_partition_sizes");
  }
  std::vector<PartitionStats> stats(assignments.size());
  for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
    const auto& bucket = assignments[static_cast<size_t>(p)];
    const uint32_t main_size = main_partition_sizes[static_cast<size_t>(p)];
    PartitionStats s;
    s.partition_id = p;
    s.insert_count = static_cast<uint32_t>(bucket.size());
    s.main_partition_size = main_size;
    s.growth_ratio =
        static_cast<double>(s.insert_count) / static_cast<double>(std::max<uint32_t>(main_size, 1));
    double dist_sum = 0.0;
    for (const auto& item : bucket) {
      dist_sum += static_cast<double>(item.residual_dist);
    }
    s.avg_residual_dist = bucket.empty() ? 0.0 : (dist_sum / static_cast<double>(bucket.size()));
    stats[static_cast<size_t>(p)] = s;
  }
  return stats;
}

Result<PartitionScoreResult> score_partitions(const std::vector<PartitionStats>& stats,
                                              double alpha,
                                              double beta,
                                              double recluster_threshold) {
  PartitionScoreResult out;
  out.decisions.reserve(stats.size());
  for (const auto& s : stats) {
    PartitionDecision d;
    d.partition_id = s.partition_id;
    d.score = alpha * s.growth_ratio + beta * s.avg_residual_dist;
    d.use_recluster = s.insert_count > 0 && d.score >= recluster_threshold;
    out.decisions.push_back(d);
    if (s.insert_count == 0) {
      continue;
    }
    if (d.use_recluster) {
      out.recluster_partitions.push_back(s.partition_id);
    } else {
      out.append_partitions.push_back(s.partition_id);
    }
  }
  return out;
}

Result<AlignedVector<VectorRecord>> merge_partition_append(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const AlignedVector<DeltaAssignment>& delta_partition_records) {
  AlignedVector<VectorRecord> merged;
  merged.reserve(main_partition_records.size() + delta_partition_records.size());
  for (const auto& rec : main_partition_records) {
    VectorRecord out = rec;
    out.ivf_id = partition_id;
    merged.push_back(std::move(out));
  }
  for (const auto& item : delta_partition_records) {
    VectorRecord out = item.record;
    out.ivf_id = partition_id;
    merged.push_back(std::move(out));
  }
  return merged;
}

Result<AlignedVector<VectorRecord>> merge_partition_recluster(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const AlignedVector<DeltaAssignment>& delta_partition_records,
    const MergeOptions& options) {
  auto append_res =
      merge_partition_append(partition_id, main_partition_records, delta_partition_records);
  if (!append_res.ok()) {
    return append_res.status();
  }
  AlignedVector<VectorRecord> merged = std::move(append_res.value());
  if (merged.size() < 2) {
    return merged;
  }
  const uint32_t dim = static_cast<uint32_t>(merged.front().x.size());
  for (const auto& rec : merged) {
    if (static_cast<uint32_t>(rec.x.size()) != dim) {
      return Status::InvalidArgument("merge_partition_recluster: inconsistent vector dims");
    }
  }

  // TODO: v1 keeps global IVF centroids unchanged and only performs local list re-organization.
  (void)options.enable_local_centroid_refine;
  const uint32_t k = std::max<uint32_t>(
      1, std::min<uint32_t>(options.local_recluster_k, static_cast<uint32_t>(merged.size())));
  if (k <= 1 || options.local_kmeans_iterations == 0) {
    return merged;
  }

  MatrixRM centers(k, dim);
  for (uint32_t c = 0; c < k; ++c) {
    const size_t src = (static_cast<size_t>(c) * merged.size()) / static_cast<size_t>(k);
    centers.row(static_cast<Eigen::Index>(c)) =
        merged[std::min(src, merged.size() - 1)].x.transpose();
  }

  std::vector<uint32_t> assignments(merged.size(), 0);
  std::vector<float> final_dists(merged.size(), 0.0f);
  for (uint32_t iter = 0; iter < options.local_kmeans_iterations; ++iter) {
    for (size_t i = 0; i < merged.size(); ++i) {
      float dist = 0.0f;
      assignments[i] = NearestCentroid(merged[i].x, centers, &dist);
      final_dists[i] = dist;
    }

    MatrixRM sum = MatrixRM::Zero(k, dim);
    std::vector<uint32_t> count(k, 0);
    for (size_t i = 0; i < merged.size(); ++i) {
      const uint32_t c = assignments[i];
      sum.row(static_cast<Eigen::Index>(c)) += merged[i].x.transpose();
      count[static_cast<size_t>(c)]++;
    }
    for (uint32_t c = 0; c < k; ++c) {
      if (count[static_cast<size_t>(c)] == 0) {
        const size_t fallback = (static_cast<size_t>(c) * merged.size()) / static_cast<size_t>(k);
        centers.row(static_cast<Eigen::Index>(c)) =
            merged[std::min(fallback, merged.size() - 1)].x.transpose();
      } else {
        centers.row(static_cast<Eigen::Index>(c)) /=
            static_cast<float>(count[static_cast<size_t>(c)]);
      }
    }
  }

  for (size_t i = 0; i < merged.size(); ++i) {
    float dist = 0.0f;
    assignments[i] = NearestCentroid(merged[i].x, centers, &dist);
    final_dists[i] = dist;
  }

  std::vector<size_t> order(merged.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    if (assignments[lhs] != assignments[rhs]) {
      return assignments[lhs] < assignments[rhs];
    }
    if (final_dists[lhs] != final_dists[rhs]) {
      return final_dists[lhs] < final_dists[rhs];
    }
    return merged[lhs].doc_id < merged[rhs].doc_id;
  });

  AlignedVector<VectorRecord> reordered;
  reordered.reserve(merged.size());
  for (size_t idx : order) {
    reordered.push_back(std::move(merged[idx]));
  }
  return reordered;
}

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("prepare_partition_patch: main_ivf is null");
  }
  std::vector<uint8_t> recluster_flags(assignments.size(), 0);
  for (const auto& d : score_result.decisions) {
    if (d.partition_id >= assignments.size()) {
      return Status::InvalidArgument("prepare_partition_patch: partition_id out of range");
    }
    recluster_flags[static_cast<size_t>(d.partition_id)] = d.use_recluster ? 1u : 0u;
  }

  PartitionPatch patch;
  for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
    const auto& delta_bucket = assignments[static_cast<size_t>(p)];
    if (delta_bucket.empty()) {
      continue;
    }
    auto main_records_res = main_ivf->GetPartitionRecords(main_versions, p);
    if (!main_records_res.ok()) {
      return main_records_res.status();
    }
    Result<AlignedVector<VectorRecord>> merged_res =
        recluster_flags[static_cast<size_t>(p)] != 0
            ? merge_partition_recluster(
                  p, main_records_res.value(), delta_bucket, options)
            : merge_partition_append(p, main_records_res.value(), delta_bucket);
    if (!merged_res.ok()) {
      return merged_res.status();
    }
    patch.partition_ids.push_back(p);
    patch.replacement_records.push_back(std::move(merged_res.value()));
  }
  return patch;
}

Status commit_partition_patch(const std::shared_ptr<IVFIndex>& main_ivf,
                              const VersionSet& main_versions,
                              const PartitionPatch& patch) {
  if (!main_ivf) {
    return Status::InvalidArgument("commit_partition_patch: main_ivf is null");
  }
  return main_ivf->CommitPartitionPatch(main_versions, patch);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("merge_frozen_delta_into_main: main_ivf is null");
  }
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    return centroids_res.status();
  }
  auto sizes_res = main_ivf->GetPartitionSizes(main_versions);
  if (!sizes_res.ok()) {
    return sizes_res.status();
  }
  auto assign_res = assign_delta_to_main_centroids(frozen_delta, centroids_res.value());
  if (!assign_res.ok()) {
    return assign_res.status();
  }
  auto stats_res = compute_partition_stats(assign_res.value(), sizes_res.value());
  if (!stats_res.ok()) {
    return stats_res.status();
  }
  auto score_res = score_partitions(
      stats_res.value(), options.alpha, options.beta, options.recluster_threshold);
  if (!score_res.ok()) {
    return score_res.status();
  }
  auto patch_res = prepare_partition_patch(
      main_ivf, main_versions, assign_res.value(), score_res.value(), options);
  if (!patch_res.ok()) {
    return patch_res.status();
  }
  Status commit = commit_partition_patch(main_ivf, main_versions, patch_res.value());
  if (!commit.ok()) {
    return commit;
  }

  MergeReport report;
  report.frozen_records = static_cast<uint32_t>(frozen_delta.records.size());
  report.patch_partitions = static_cast<uint32_t>(patch_res.value().partition_ids.size());
  report.append_partitions = static_cast<uint32_t>(score_res.value().append_partitions.size());
  report.recluster_partitions = static_cast<uint32_t>(score_res.value().recluster_partitions.size());
  report.stats = std::move(stats_res.value());
  report.scoring = std::move(score_res.value());
  return report;
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const MergeOptions& options) {
  auto frozen_res = freeze_delta(delta_ivf, delta_versions);
  if (!frozen_res.ok()) {
    return frozen_res.status();
  }
  return merge_frozen_delta_into_main(main_ivf, main_versions, frozen_res.value(), options);
}

}  // namespace ann
