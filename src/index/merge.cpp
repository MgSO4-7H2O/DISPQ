#include "index/merge.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/timer.h"

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

double QuantileFromSorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double clamped = std::clamp(q, 0.0, 1.0);
  const double pos = clamped * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return sorted[lo];
  }
  const double w = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

DistributionSummary SummarizeDistribution(std::vector<double> values) {
  DistributionSummary summary;
  if (values.empty()) {
    return summary;
  }
  std::sort(values.begin(), values.end());
  summary.min = values.front();
  summary.p50 = QuantileFromSorted(values, 0.50);
  summary.p90 = QuantileFromSorted(values, 0.90);
  summary.max = values.back();
  return summary;
}

struct AssignmentDiagnostics {
  double moved_delta_ratio{0.0};
  double avg_assignment_dist_ratio{1.0};
  double max_assignment_dist_ratio{1.0};
  double imbalance_before{0.0};
  double imbalance_after{0.0};
};

double ComputeImbalanceRatio(const std::vector<uint32_t>& sizes) {
  if (sizes.empty()) {
    return 0.0;
  }
  uint64_t total = 0;
  uint32_t max_size = 0;
  for (uint32_t sz : sizes) {
    total += static_cast<uint64_t>(sz);
    max_size = std::max<uint32_t>(max_size, sz);
  }
  const double avg = static_cast<double>(total) / static_cast<double>(sizes.size());
  return static_cast<double>(max_size) / std::max(1.0, avg);
}

Result<PartitionAssignments> assign_delta_to_main_centroids_for_merge(
    const FrozenDelta& frozen_delta,
    Eigen::Ref<const MatrixRM> main_centroids,
    const std::vector<uint32_t>& main_partition_sizes,
    const MergeOptions& options,
    AssignmentDiagnostics* diag_out) {
  if (main_centroids.rows() == 0 || main_centroids.cols() == 0) {
    return Status::InvalidArgument("assign_delta_to_main_centroids_for_merge: empty main_centroids");
  }
  if (main_partition_sizes.size() != static_cast<size_t>(main_centroids.rows())) {
    return Status::InvalidArgument(
        "assign_delta_to_main_centroids_for_merge: main_partition_sizes size mismatch");
  }

AssignmentDiagnostics diag;
  diag.imbalance_before = ComputeImbalanceRatio(main_partition_sizes);

  const uint32_t nlist = static_cast<uint32_t>(main_centroids.rows());
  const bool use_balanced_append = options.assignment_mode == "balanced_append";
  const uint32_t top_r = std::max<uint32_t>(1, std::min<uint32_t>(options.assignment_top_r, nlist));
  const double main_rows = std::accumulate(main_partition_sizes.begin(),
                                           main_partition_sizes.end(),
                                           0.0);
  const double delta_rows = static_cast<double>(frozen_delta.records.size());
  const double avg_after = (main_rows + delta_rows) / std::max(1.0, static_cast<double>(nlist));
  const double hard_cap = avg_after * options.assignment_hard_cap_ratio;
  const double balance_denom = std::max(1.0, avg_after);
  const double eps = 1e-12;

  PartitionAssignments assigned(static_cast<size_t>(nlist));
  std::vector<uint32_t> projected_size = main_partition_sizes;
  uint32_t moved_count = 0;
  double ratio_sum = 0.0;
  double ratio_max = 1.0;

  for (const auto& rec : frozen_delta.records) {
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument(
          "assign_delta_to_main_centroids_for_merge: record dim mismatch with main centroids");
    }

    std::vector<std::pair<float, uint32_t>> dists;
    dists.reserve(static_cast<size_t>(nlist));
    for (uint32_t i = 0; i < nlist; ++i) {
      const float dist =
          (rec.x - main_centroids.row(static_cast<Eigen::Index>(i)).transpose()).squaredNorm();
      dists.emplace_back(dist, i);
    }
    std::partial_sort(dists.begin(),
                      dists.begin() + static_cast<std::ptrdiff_t>(top_r),
                      dists.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    const uint32_t nearest_partition = dists[0].second;
    const float nearest_dist = dists[0].first;
    uint32_t chosen_partition = nearest_partition;
    float chosen_dist = nearest_dist;

    if (use_balanced_append) {
      const float dist_cap = nearest_dist * static_cast<float>(options.assignment_gamma);
      const double nearest_penalty = std::max(
          0.0,
          (static_cast<double>(projected_size[static_cast<size_t>(nearest_partition)]) + 1.0) -
              avg_after) /
          balance_denom;
      const bool nearest_overloaded = nearest_penalty > 0.0;

      struct CandidateState {
        uint32_t part{0};
        float dist{0.0f};
        double penalty{0.0};
        double cost{0.0};
      };

      std::vector<CandidateState> legal_candidates;
      legal_candidates.reserve(top_r);
      std::vector<CandidateState> healthier_candidates;
      healthier_candidates.reserve(top_r);

      for (uint32_t k = 0; k < top_r; ++k) {
        const float dist = dists[static_cast<size_t>(k)].first;
        const uint32_t part = dists[static_cast<size_t>(k)].second;
        if (dist > dist_cap) {
          continue;
        }
        if (static_cast<double>(projected_size[static_cast<size_t>(part)]) >= hard_cap) {
          continue;
        }
        CandidateState cand;
        cand.part = part;
        cand.dist = dist;
        cand.penalty = std::max(
            0.0,
            (static_cast<double>(projected_size[static_cast<size_t>(part)]) + 1.0) - avg_after) /
                       balance_denom;
        cand.cost = static_cast<double>(dist) + options.assignment_lambda * cand.penalty;
        legal_candidates.push_back(cand);
        if (nearest_overloaded && cand.penalty + 1e-12 < nearest_penalty) {
          healthier_candidates.push_back(cand);
        }
      }

      const std::vector<CandidateState>& candidates =
          healthier_candidates.empty() ? legal_candidates : healthier_candidates;
      double best_cost = std::numeric_limits<double>::infinity();
      bool found_candidate = false;
      for (const auto& cand : candidates) {
        if (!found_candidate || cand.cost < best_cost) {
          found_candidate = true;
          best_cost = cand.cost;
          chosen_partition = cand.part;
          chosen_dist = cand.dist;
        }
      }
    }

    DeltaAssignment item;
    item.record = rec;
    item.record.ivf_id = chosen_partition;
    item.main_partition = chosen_partition;
    item.residual_dist = chosen_dist;
    assigned[static_cast<size_t>(chosen_partition)].push_back(std::move(item));
    projected_size[static_cast<size_t>(chosen_partition)]++;

    if (chosen_partition != nearest_partition) {
      moved_count++;
    }
    const double ratio = nearest_dist > eps ? static_cast<double>(chosen_dist) / nearest_dist : 1.0;
    ratio_sum += ratio;
    ratio_max = std::max(ratio_max, ratio);
  }

  if (!frozen_delta.records.empty()) {
    const double denom = static_cast<double>(frozen_delta.records.size());
    diag.moved_delta_ratio = static_cast<double>(moved_count) / denom;
    diag.avg_assignment_dist_ratio = ratio_sum / denom;
    diag.max_assignment_dist_ratio = ratio_max;
  }
  diag.imbalance_after = ComputeImbalanceRatio(projected_size);
  if (diag_out != nullptr) {
    *diag_out = diag;
  }
  return assigned;
}

Status AppendVectorRecordFromStore(DocId doc_id,
                                   const VersionSet& versions,
                                   uint32_t ivf_id,
                                   const MatrixRM& vector_store,
                                   AlignedVector<VectorRecord>* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("AppendVectorRecordFromStore: null output");
  }
  if (static_cast<Eigen::Index>(doc_id) >= vector_store.rows()) {
    return Status::InvalidArgument("AppendVectorRecordFromStore: doc_id out of vector store range");
  }
  VectorRecord rec;
  rec.doc_id = doc_id;
  rec.dim = static_cast<uint32_t>(vector_store.cols());
  rec.versions = versions;
  rec.ivf_id = ivf_id;
  rec.x = vector_store.row(static_cast<Eigen::Index>(doc_id)).transpose();
  out->push_back(std::move(rec));
  return Status::OK();
}


Status AppendVectorRecordFromAccessor(DocId doc_id,
                                      const VersionSet& versions,
                                      uint32_t ivf_id,
                                      const VectorAccessor& vector_accessor,
                                      AlignedVector<VectorRecord>* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("AppendVectorRecordFromAccessor: null output");
  }
  auto vec_res = vector_accessor.GetVector(doc_id);
  if (!vec_res.ok()) {
    return vec_res.status();
  }
  VectorRecord rec;
  rec.doc_id = doc_id;
  rec.dim = vector_accessor.dim();
  rec.versions = versions;
  rec.ivf_id = ivf_id;
  rec.x = std::move(vec_res.value());
  if (static_cast<uint32_t>(rec.x.size()) != rec.dim) {
    return Status::InvalidArgument("AppendVectorRecordFromAccessor: dimension mismatch");
  }
  out->push_back(std::move(rec));
  return Status::OK();
}

Result<AlignedVector<VectorRecord>> FetchPartitionRecordsForMerge(
    const std::shared_ptr<IVFIndex>& ivf,
    const VersionSet& versions,
    uint32_t partition_id,
    const MatrixRM* vector_store) {
  if (!ivf) {
    return Status::InvalidArgument("FetchPartitionRecordsForMerge: ivf is null");
  }
  if (vector_store == nullptr) {
    return ivf->GetPartitionRecords(versions, partition_id);
  }

  auto doc_ids_res = ivf->GetPartitionDocIds(versions, partition_id);
  if (!doc_ids_res.ok()) {
    return doc_ids_res.status();
  }
  AlignedVector<VectorRecord> records;
  records.reserve(doc_ids_res.value().size());
  for (DocId doc_id : doc_ids_res.value()) {
    Status append = AppendVectorRecordFromStore(
        doc_id, versions, partition_id, *vector_store, &records);
    if (!append.ok()) {
      return append;
    }
  }
  return records;
}


Result<AlignedVector<VectorRecord>> FetchPartitionRecordsForMergeAccessor(
    const std::shared_ptr<IVFIndex>& ivf,
    const VersionSet& versions,
    uint32_t partition_id,
    const VectorAccessor* vector_accessor) {
  if (!ivf) {
    return Status::InvalidArgument("FetchPartitionRecordsForMergeAccessor: ivf is null");
  }
  if (vector_accessor == nullptr) {
    return ivf->GetPartitionRecords(versions, partition_id);
  }
  auto doc_ids_res = ivf->GetPartitionDocIds(versions, partition_id);
  if (!doc_ids_res.ok()) {
    return doc_ids_res.status();
  }
  AlignedVector<VectorRecord> records;
  records.reserve(doc_ids_res.value().size());
  for (DocId doc_id : doc_ids_res.value()) {
    Status append = AppendVectorRecordFromAccessor(
        doc_id, versions, partition_id, *vector_accessor, &records);
    if (!append.ok()) {
      return append;
    }
  }
  return records;
}

std::vector<uint32_t> TopRNeighborPartitions(uint32_t seed_partition,
                                             Eigen::Ref<const MatrixRM> main_centroids,
                                             uint32_t top_r) {
  const uint32_t nlist = static_cast<uint32_t>(main_centroids.rows());
  if (nlist == 0 || seed_partition >= nlist) {
    return {};
  }
  const uint32_t keep = std::max<uint32_t>(1, std::min<uint32_t>(top_r, nlist));
  const Eigen::VectorXf seed = main_centroids.row(static_cast<Eigen::Index>(seed_partition)).transpose();
  std::vector<std::pair<float, uint32_t>> dists;
  dists.reserve(static_cast<size_t>(nlist));
  for (uint32_t p = 0; p < nlist; ++p) {
    const float dist = (seed - main_centroids.row(static_cast<Eigen::Index>(p)).transpose()).squaredNorm();
    dists.emplace_back(dist, p);
  }
  std::partial_sort(dists.begin(),
                    dists.begin() + static_cast<std::ptrdiff_t>(keep),
                    dists.end(),
                    [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  std::vector<uint32_t> out;
  out.reserve(keep);
  for (uint32_t i = 0; i < keep; ++i) {
    out.push_back(dists[static_cast<size_t>(i)].second);
  }
  return out;
}

Result<std::vector<AlignedVector<VectorRecord>>> RepartitionNeighborhood(
    const std::vector<uint32_t>& neighborhood_partitions,
    const std::vector<AlignedVector<VectorRecord>>& neighborhood_main_records,
    const PartitionAssignments& assignments,
    Eigen::Ref<const MatrixRM> main_centroids,
    const MergeOptions& options) {
  if (neighborhood_partitions.empty()) {
    return Status::InvalidArgument("RepartitionNeighborhood: empty neighborhood");
  }
  if (neighborhood_partitions.size() != neighborhood_main_records.size()) {
    return Status::InvalidArgument(
        "RepartitionNeighborhood: partition/main-records size mismatch");
  }

  const uint32_t neighborhood_n = static_cast<uint32_t>(neighborhood_partitions.size());
  for (uint32_t part : neighborhood_partitions) {
    if (part >= static_cast<uint32_t>(assignments.size())) {
      return Status::InvalidArgument("RepartitionNeighborhood: partition out of assignments range");
    }
    if (part >= static_cast<uint32_t>(main_centroids.rows())) {
      return Status::InvalidArgument("RepartitionNeighborhood: partition out of centroid range");
    }
  }

  AlignedVector<VectorRecord> pooled_records;
  size_t reserve_count = 0;
  for (const auto& records : neighborhood_main_records) {
    reserve_count += records.size();
  }
  for (uint32_t part : neighborhood_partitions) {
    reserve_count += assignments[static_cast<size_t>(part)].size();
  }
  pooled_records.reserve(reserve_count);

  for (const auto& records : neighborhood_main_records) {
    for (const auto& rec : records) {
      pooled_records.push_back(rec);
    }
  }
  for (uint32_t part : neighborhood_partitions) {
    const auto& delta_bucket = assignments[static_cast<size_t>(part)];
    for (const auto& item : delta_bucket) {
      pooled_records.push_back(item.record);
    }
  }

  std::vector<AlignedVector<VectorRecord>> repartitioned(neighborhood_n);
  if (pooled_records.empty()) {
    return repartitioned;
  }

  const double avg_after =
      static_cast<double>(pooled_records.size()) / static_cast<double>(neighborhood_n);
  const double hard_cap = avg_after * options.assignment_hard_cap_ratio;
  const double balance_denom = std::max(1.0, avg_after);
  std::vector<uint32_t> projected_size(neighborhood_n, 0u);

  for (const auto& rec : pooled_records) {
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument("RepartitionNeighborhood: record dim mismatch");
    }
    std::vector<std::pair<float, uint32_t>> dists;
    dists.reserve(neighborhood_n);
    for (uint32_t local_idx = 0; local_idx < neighborhood_n; ++local_idx) {
      const uint32_t part = neighborhood_partitions[static_cast<size_t>(local_idx)];
      const float dist =
          (rec.x - main_centroids.row(static_cast<Eigen::Index>(part)).transpose()).squaredNorm();
      dists.emplace_back(dist, local_idx);
    }
    std::sort(dists.begin(),
              dists.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    const uint32_t nearest_local = dists[0].second;
    const float nearest_dist = dists[0].first;
    uint32_t chosen_local = nearest_local;

    const float dist_cap = nearest_dist * static_cast<float>(options.assignment_gamma);
    const double nearest_penalty = std::max(
        0.0,
        (static_cast<double>(projected_size[static_cast<size_t>(nearest_local)]) + 1.0) -
            avg_after) /
                                   balance_denom;
    const bool nearest_overloaded = nearest_penalty > 0.0;

    struct CandidateState {
      uint32_t local_idx{0};
      float dist{0.0f};
      double penalty{0.0};
      double cost{0.0};
    };

    std::vector<CandidateState> legal_candidates;
    legal_candidates.reserve(neighborhood_n);
    std::vector<CandidateState> healthier_candidates;
    healthier_candidates.reserve(neighborhood_n);

    for (const auto& dist_item : dists) {
      const float dist = dist_item.first;
      const uint32_t local_idx = dist_item.second;
      if (dist > dist_cap) {
        continue;
      }
      if (static_cast<double>(projected_size[static_cast<size_t>(local_idx)]) >= hard_cap) {
        continue;
      }
      CandidateState cand;
      cand.local_idx = local_idx;
      cand.dist = dist;
      cand.penalty = std::max(
          0.0,
          (static_cast<double>(projected_size[static_cast<size_t>(local_idx)]) + 1.0) -
              avg_after) /
                     balance_denom;
      cand.cost = static_cast<double>(dist) + options.assignment_lambda * cand.penalty;
      legal_candidates.push_back(cand);
      if (nearest_overloaded && cand.penalty + 1e-12 < nearest_penalty) {
        healthier_candidates.push_back(cand);
      }
    }

    const std::vector<CandidateState>& candidates =
        healthier_candidates.empty() ? legal_candidates : healthier_candidates;
    double best_cost = std::numeric_limits<double>::infinity();
    bool found_candidate = false;
    for (const auto& cand : candidates) {
      if (!found_candidate || cand.cost < best_cost) {
        found_candidate = true;
        best_cost = cand.cost;
        chosen_local = cand.local_idx;
      }
    }

    VectorRecord out = rec;
    out.ivf_id = neighborhood_partitions[static_cast<size_t>(chosen_local)];
    repartitioned[static_cast<size_t>(chosen_local)].push_back(std::move(out));
    projected_size[static_cast<size_t>(chosen_local)]++;
  }

  for (auto& bucket : repartitioned) {
    std::sort(bucket.begin(), bucket.end(), [](const VectorRecord& lhs, const VectorRecord& rhs) {
      return lhs.doc_id < rhs.doc_id;
    });
  }
  return repartitioned;
}

}  // namespace

Result<PartitionPatch> PreparePartitionPatchImpl(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MatrixRM* base_vectors,
    const MergeOptions& options);

Result<MergeReport> MergeFrozenDeltaIntoMainImpl(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MatrixRM* base_vectors,
                                                 const MergeOptions& options);

Result<PartitionPatch> PreparePartitionPatchImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options);

Result<MergeReport> MergeFrozenDeltaIntoMainImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options);

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

Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions,
                                 const MatrixRM& base_vectors) {
  if (!delta_ivf) {
    return Status::InvalidArgument("freeze_delta: delta_ivf is null");
  }
  auto doc_ids_res = delta_ivf->SnapshotDocIds(delta_versions);
  if (!doc_ids_res.ok()) {
    return doc_ids_res.status();
  }
  FrozenDelta out;
  out.records.reserve(doc_ids_res.value().size());
  for (DocId doc_id : doc_ids_res.value()) {
    Status append = AppendVectorRecordFromStore(doc_id, delta_versions, 0, base_vectors, &out.records);
    if (!append.ok()) {
      return append;
    }
  }
  return out;
}


Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions,
                                 const VectorAccessor& vector_accessor) {
  if (!delta_ivf) {
    return Status::InvalidArgument("freeze_delta(accessor): delta_ivf is null");
  }
  auto doc_ids_res = delta_ivf->SnapshotDocIds(delta_versions);
  if (!doc_ids_res.ok()) {
    return doc_ids_res.status();
  }
  FrozenDelta out;
  out.records.reserve(doc_ids_res.value().size());
  for (DocId doc_id : doc_ids_res.value()) {
    Status append = AppendVectorRecordFromAccessor(
        doc_id, delta_versions, 0, vector_accessor, &out.records);
    if (!append.ok()) {
      return append;
    }
  }
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
  return PreparePartitionPatchImpl(
      main_ivf, main_versions, assignments, score_result, nullptr, options);
}

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MatrixRM& base_vectors,
    const MergeOptions& options) {
  return PreparePartitionPatchImpl(
      main_ivf, main_versions, assignments, score_result, &base_vectors, options);
}

Result<PartitionPatch> PreparePartitionPatchImpl(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MatrixRM* base_vectors,
    const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("prepare_partition_patch: main_ivf is null");
  }

  if (options.assignment_mode == "balanced_append") {
    auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
    if (!centroids_res.ok()) {
      return centroids_res.status();
    }
    if (assignments.size() != static_cast<size_t>(centroids_res.value().rows())) {
      return Status::InvalidArgument(
          "prepare_partition_patch: assignments size mismatch with centroids");
    }

    std::vector<uint32_t> seeds;
    seeds.reserve(assignments.size());
    for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
      if (!assignments[static_cast<size_t>(p)].empty()) {
        seeds.push_back(p);
      }
    }
    std::sort(seeds.begin(),
              seeds.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                return assignments[static_cast<size_t>(lhs)].size() >
                       assignments[static_cast<size_t>(rhs)].size();
              });

    PartitionPatch patch;
    std::vector<uint8_t> claimed(assignments.size(), 0u);
    for (uint32_t seed : seeds) {
      if (claimed[static_cast<size_t>(seed)] != 0u) {
        continue;
      }
      std::vector<uint32_t> neighborhood =
          TopRNeighborPartitions(seed, centroids_res.value(), options.assignment_top_r);
      std::vector<uint32_t> active_neighborhood;
      active_neighborhood.reserve(neighborhood.size());
      for (uint32_t part : neighborhood) {
        if (claimed[static_cast<size_t>(part)] == 0u) {
          active_neighborhood.push_back(part);
        }
      }
      if (active_neighborhood.empty()) {
        continue;
      }

      std::vector<AlignedVector<VectorRecord>> neighborhood_main_records;
      neighborhood_main_records.reserve(active_neighborhood.size());
      for (uint32_t part : active_neighborhood) {
        auto main_records_res =
            FetchPartitionRecordsForMerge(main_ivf, main_versions, part, base_vectors);
        if (!main_records_res.ok()) {
          return main_records_res.status();
        }
        neighborhood_main_records.push_back(std::move(main_records_res.value()));
      }

      auto repartition_res = RepartitionNeighborhood(active_neighborhood,
                                                     neighborhood_main_records,
                                                     assignments,
                                                     centroids_res.value(),
                                                     options);
      if (!repartition_res.ok()) {
        return repartition_res.status();
      }
      std::vector<AlignedVector<VectorRecord>> repartitioned = std::move(repartition_res.value());
      for (size_t i = 0; i < active_neighborhood.size(); ++i) {
        patch.partition_ids.push_back(active_neighborhood[i]);
        patch.replacement_records.push_back(std::move(repartitioned[i]));
        claimed[static_cast<size_t>(active_neighborhood[i])] = 1u;
      }
    }
    return patch;
  }

  std::vector<uint8_t> recluster_flags(assignments.size(), 0);
  for (const auto& d : score_result.decisions) {
    if (d.partition_id >= assignments.size()) {
      return Status::InvalidArgument("prepare_partition_patch: partition_id out of range");
    }
    recluster_flags[static_cast<size_t>(d.partition_id)] = d.use_recluster ? 1u : 0u;
  }

  PartitionPatch patch;
  std::vector<uint32_t> active_partitions;
  active_partitions.reserve(assignments.size());
  for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
    if (!assignments[static_cast<size_t>(p)].empty()) {
      active_partitions.push_back(p);
    }
  }
  patch.partition_ids = active_partitions;
  patch.replacement_records.resize(active_partitions.size());
  if (active_partitions.empty()) {
    return patch;
  }

  std::atomic<bool> failed{false};
  std::mutex err_mu;
  Status first_error = Status::OK();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(active_partitions.size()); ++i) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    const uint32_t p = active_partitions[static_cast<size_t>(i)];
    const auto& delta_bucket = assignments[static_cast<size_t>(p)];
    auto main_records_res =
        FetchPartitionRecordsForMerge(main_ivf, main_versions, p, base_vectors);
    if (!main_records_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = main_records_res.status();
      }
      continue;
    }
    Result<AlignedVector<VectorRecord>> merged_res =
        recluster_flags[static_cast<size_t>(p)] != 0
            ? merge_partition_recluster(p, main_records_res.value(), delta_bucket, options)
            : merge_partition_append(p, main_records_res.value(), delta_bucket);
    if (!merged_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = merged_res.status();
      }
      continue;
    }
    patch.replacement_records[static_cast<size_t>(i)] = std::move(merged_res.value());
  }
  if (failed.load(std::memory_order_relaxed)) {
    return first_error;
  }
  return patch;
}

Result<PartitionPatch> PreparePartitionPatchImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("prepare_partition_patch: main_ivf is null");
  }

  if (options.assignment_mode == "balanced_append") {
    auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
    if (!centroids_res.ok()) {
      return centroids_res.status();
    }
    if (assignments.size() != static_cast<size_t>(centroids_res.value().rows())) {
      return Status::InvalidArgument(
          "prepare_partition_patch: assignments size mismatch with centroids");
    }

    std::vector<uint32_t> seeds;
    seeds.reserve(assignments.size());
    for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
      if (!assignments[static_cast<size_t>(p)].empty()) {
        seeds.push_back(p);
      }
    }
    std::sort(seeds.begin(),
              seeds.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                return assignments[static_cast<size_t>(lhs)].size() >
                       assignments[static_cast<size_t>(rhs)].size();
              });

    PartitionPatch patch;
    std::vector<uint8_t> claimed(assignments.size(), 0u);
    for (uint32_t seed : seeds) {
      if (claimed[static_cast<size_t>(seed)] != 0u) {
        continue;
      }
      std::vector<uint32_t> neighborhood =
          TopRNeighborPartitions(seed, centroids_res.value(), options.assignment_top_r);
      std::vector<uint32_t> active_neighborhood;
      active_neighborhood.reserve(neighborhood.size());
      for (uint32_t part : neighborhood) {
        if (claimed[static_cast<size_t>(part)] == 0u) {
          active_neighborhood.push_back(part);
        }
      }
      if (active_neighborhood.empty()) {
        continue;
      }

      std::vector<AlignedVector<VectorRecord>> neighborhood_main_records;
      neighborhood_main_records.reserve(active_neighborhood.size());
      for (uint32_t part : active_neighborhood) {
        auto main_records_res =
            FetchPartitionRecordsForMergeAccessor(main_ivf, main_versions, part, vector_accessor);
        if (!main_records_res.ok()) {
          return main_records_res.status();
        }
        neighborhood_main_records.push_back(std::move(main_records_res.value()));
      }

      auto repartition_res = RepartitionNeighborhood(active_neighborhood,
                                                     neighborhood_main_records,
                                                     assignments,
                                                     centroids_res.value(),
                                                     options);
      if (!repartition_res.ok()) {
        return repartition_res.status();
      }
      std::vector<AlignedVector<VectorRecord>> repartitioned = std::move(repartition_res.value());
      for (size_t i = 0; i < active_neighborhood.size(); ++i) {
        patch.partition_ids.push_back(active_neighborhood[i]);
        patch.replacement_records.push_back(std::move(repartitioned[i]));
        claimed[static_cast<size_t>(active_neighborhood[i])] = 1u;
      }
    }
    return patch;
  }

  std::vector<uint8_t> recluster_flags(assignments.size(), 0);
  for (const auto& d : score_result.decisions) {
    if (d.partition_id >= assignments.size()) {
      return Status::InvalidArgument("prepare_partition_patch: partition_id out of range");
    }
    recluster_flags[static_cast<size_t>(d.partition_id)] = d.use_recluster ? 1u : 0u;
  }

  PartitionPatch patch;
  std::vector<uint32_t> active_partitions;
  active_partitions.reserve(assignments.size());
  for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
    if (!assignments[static_cast<size_t>(p)].empty()) {
      active_partitions.push_back(p);
    }
  }
  patch.partition_ids = active_partitions;
  patch.replacement_records.resize(active_partitions.size());
  if (active_partitions.empty()) {
    return patch;
  }

  std::atomic<bool> failed{false};
  std::mutex err_mu;
  Status first_error = Status::OK();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(active_partitions.size()); ++i) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    const uint32_t p = active_partitions[static_cast<size_t>(i)];
    const auto& delta_bucket = assignments[static_cast<size_t>(p)];
    auto main_records_res =
        FetchPartitionRecordsForMergeAccessor(main_ivf, main_versions, p, vector_accessor);
    if (!main_records_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = main_records_res.status();
      }
      continue;
    }
    Result<AlignedVector<VectorRecord>> merged_res =
        recluster_flags[static_cast<size_t>(p)] != 0
            ? merge_partition_recluster(p, main_records_res.value(), delta_bucket, options)
            : merge_partition_append(p, main_records_res.value(), delta_bucket);
    if (!merged_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = merged_res.status();
      }
      continue;
    }
    patch.replacement_records[static_cast<size_t>(i)] = std::move(merged_res.value());
  }
  if (failed.load(std::memory_order_relaxed)) {
    return first_error;
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
  return MergeFrozenDeltaIntoMainImpl(
      main_ivf, main_versions, frozen_delta, nullptr, options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MatrixRM& base_vectors,
                                                 const MergeOptions& options) {
  return MergeFrozenDeltaIntoMainImpl(
      main_ivf, main_versions, frozen_delta, &base_vectors, options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const VectorAccessor& vector_accessor,
                                                 const MergeOptions& options) {
  return MergeFrozenDeltaIntoMainImplAccessor(
      main_ivf, main_versions, frozen_delta, &vector_accessor, options);
}

Result<MergeReport> MergeFrozenDeltaIntoMainImpl(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MatrixRM* base_vectors,
                                                 const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("merge_frozen_delta_into_main: main_ivf is null");
  }
  Timer merge_compute_timer;
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    return centroids_res.status();
  }
  auto sizes_res = main_ivf->GetPartitionSizes(main_versions);
  if (!sizes_res.ok()) {
    return sizes_res.status();
  }
  AssignmentDiagnostics assignment_diag;
  auto assign_res = assign_delta_to_main_centroids_for_merge(
      frozen_delta, centroids_res.value(), sizes_res.value(), options, &assignment_diag);
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
  auto patch_res = PreparePartitionPatchImpl(
      main_ivf, main_versions, assign_res.value(), score_res.value(), base_vectors, options);
  if (!patch_res.ok()) {
    return patch_res.status();
  }
  const double merge_compute_ms = merge_compute_timer.ElapsedMillis();
  Status commit = commit_partition_patch(main_ivf, main_versions, patch_res.value());
  if (!commit.ok()) {
    return commit;
  }
  auto codebook_ms_res = main_ivf->GetLastPatchPQReencodeMs(main_versions);
  if (!codebook_ms_res.ok()) {
    return codebook_ms_res.status();
  }

  MergeReport report;
  report.frozen_records = static_cast<uint32_t>(frozen_delta.records.size());
  report.patch_partitions = static_cast<uint32_t>(patch_res.value().partition_ids.size());
  report.append_partitions = static_cast<uint32_t>(score_res.value().append_partitions.size());
  report.recluster_partitions = static_cast<uint32_t>(score_res.value().recluster_partitions.size());
  report.merge_compute_ms = merge_compute_ms;
  report.codebook_rebuild_ms = codebook_ms_res.value();
  report.moved_delta_ratio = assignment_diag.moved_delta_ratio;
  report.avg_assignment_dist_ratio = assignment_diag.avg_assignment_dist_ratio;
  report.max_assignment_dist_ratio = assignment_diag.max_assignment_dist_ratio;
  report.imbalance_before = assignment_diag.imbalance_before;
  report.imbalance_after = assignment_diag.imbalance_after;
  std::vector<double> score_values;
  std::vector<double> residual_values;
  std::vector<double> growth_values;
  score_values.reserve(report.patch_partitions);
  residual_values.reserve(report.patch_partitions);
  growth_values.reserve(report.patch_partitions);
  const auto& stats = stats_res.value();
  const auto& decisions = score_res.value().decisions;
  const size_t n = std::min(stats.size(), decisions.size());
  for (size_t i = 0; i < n; ++i) {
    if (stats[i].insert_count == 0) {
      continue;
    }
    score_values.push_back(decisions[i].score);
    residual_values.push_back(stats[i].avg_residual_dist);
    growth_values.push_back(stats[i].growth_ratio);
  }
  report.score_summary = SummarizeDistribution(std::move(score_values));
  report.residual_summary = SummarizeDistribution(std::move(residual_values));
  report.growth_summary = SummarizeDistribution(std::move(growth_values));
  report.stats = std::move(stats_res.value());
  report.scoring = std::move(score_res.value());
  return report;
}

Result<MergeReport> MergeFrozenDeltaIntoMainImplAccessor(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const VectorAccessor* vector_accessor,
    const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("merge_frozen_delta_into_main: main_ivf is null");
  }
  Timer merge_compute_timer;
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    return centroids_res.status();
  }
  auto sizes_res = main_ivf->GetPartitionSizes(main_versions);
  if (!sizes_res.ok()) {
    return sizes_res.status();
  }
  AssignmentDiagnostics assignment_diag;
  auto assign_res = assign_delta_to_main_centroids_for_merge(
      frozen_delta, centroids_res.value(), sizes_res.value(), options, &assignment_diag);
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
  auto patch_res = PreparePartitionPatchImplAccessor(
      main_ivf, main_versions, assign_res.value(), score_res.value(), vector_accessor, options);
  if (!patch_res.ok()) {
    return patch_res.status();
  }
  const double merge_compute_ms = merge_compute_timer.ElapsedMillis();
  Status commit = commit_partition_patch(main_ivf, main_versions, patch_res.value());
  if (!commit.ok()) {
    return commit;
  }
  auto codebook_ms_res = main_ivf->GetLastPatchPQReencodeMs(main_versions);
  if (!codebook_ms_res.ok()) {
    return codebook_ms_res.status();
  }

  MergeReport report;
  report.frozen_records = static_cast<uint32_t>(frozen_delta.records.size());
  report.patch_partitions = static_cast<uint32_t>(patch_res.value().partition_ids.size());
  report.append_partitions = static_cast<uint32_t>(score_res.value().append_partitions.size());
  report.recluster_partitions = static_cast<uint32_t>(score_res.value().recluster_partitions.size());
  report.merge_compute_ms = merge_compute_ms;
  report.codebook_rebuild_ms = codebook_ms_res.value();
  report.moved_delta_ratio = assignment_diag.moved_delta_ratio;
  report.avg_assignment_dist_ratio = assignment_diag.avg_assignment_dist_ratio;
  report.max_assignment_dist_ratio = assignment_diag.max_assignment_dist_ratio;
  report.imbalance_before = assignment_diag.imbalance_before;
  report.imbalance_after = assignment_diag.imbalance_after;
  std::vector<double> score_values;
  std::vector<double> residual_values;
  std::vector<double> growth_values;
  score_values.reserve(report.patch_partitions);
  residual_values.reserve(report.patch_partitions);
  growth_values.reserve(report.patch_partitions);
  const auto& stats = stats_res.value();
  const auto& decisions = score_res.value().decisions;
  const size_t n = std::min(stats.size(), decisions.size());
  for (size_t i = 0; i < n; ++i) {
    if (stats[i].insert_count == 0) {
      continue;
    }
    score_values.push_back(decisions[i].score);
    residual_values.push_back(stats[i].avg_residual_dist);
    growth_values.push_back(stats[i].growth_ratio);
  }
  report.score_summary = SummarizeDistribution(std::move(score_values));
  report.residual_summary = SummarizeDistribution(std::move(residual_values));
  report.growth_summary = SummarizeDistribution(std::move(growth_values));
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

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const MatrixRM& base_vectors,
                                                 const MergeOptions& options) {
  auto frozen_res = freeze_delta(delta_ivf, delta_versions, base_vectors);
  if (!frozen_res.ok()) {
    return frozen_res.status();
  }
  return merge_frozen_delta_into_main(
      main_ivf, main_versions, frozen_res.value(), base_vectors, options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const VectorAccessor& vector_accessor,
                                                 const MergeOptions& options) {
  auto frozen_res = freeze_delta(delta_ivf, delta_versions, vector_accessor);
  if (!frozen_res.ok()) {
    return frozen_res.status();
  }
  return merge_frozen_delta_into_main(
      main_ivf, main_versions, frozen_res.value(), vector_accessor, options);
}

}  // namespace ann