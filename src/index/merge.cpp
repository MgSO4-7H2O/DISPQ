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

constexpr size_t kMergeRepartitionChunkRows = 65536;

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

struct AssignmentTopCandidate {
  float dist{0.0f};
  uint32_t part{0};
};

struct AssignmentCandidateState {
  uint32_t part{0};
  float dist{0.0f};
  double penalty{0.0};
  double cost{0.0};
};

struct AssignmentScratch {
  std::vector<AssignmentTopCandidate> top_candidates;
};

bool IsAssignmentCandidateLess(const AssignmentTopCandidate& lhs,
                               const AssignmentTopCandidate& rhs) {
  if (lhs.dist != rhs.dist) {
    return lhs.dist < rhs.dist;
  }
  return lhs.part < rhs.part;
}

void InsertAssignmentTopCandidate(AssignmentScratch* scratch,
                                  AssignmentTopCandidate candidate,
                                  uint32_t top_r) {
  auto& top_candidates = scratch->top_candidates;
  if (top_candidates.size() == static_cast<size_t>(top_r) &&
      !IsAssignmentCandidateLess(candidate, top_candidates.back())) {
    return;
  }
  const auto insert_pos = std::lower_bound(
      top_candidates.begin(), top_candidates.end(), candidate,
      [](const AssignmentTopCandidate& lhs, const AssignmentTopCandidate& rhs) {
        return IsAssignmentCandidateLess(lhs, rhs);
      });
  top_candidates.insert(insert_pos, candidate);
  if (top_candidates.size() > static_cast<size_t>(top_r)) {
    top_candidates.pop_back();
  }
}

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
    AssignmentDiagnostics* diag_out,
    MergeProfiling* profiling) {
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
  }

  constexpr size_t kMaxAssignmentChunkBytes = 64ULL * 1024ULL * 1024ULL;
  constexpr size_t kMaxAssignmentChunkRecords = 16384;
  const size_t bytes_per_record =
      static_cast<size_t>(nlist) * sizeof(float) +
      static_cast<size_t>(top_r) * sizeof(AssignmentTopCandidate) +
      sizeof(uint32_t) + sizeof(float);
  const size_t chunk_records = std::max<size_t>(
      1,
      std::min(kMaxAssignmentChunkRecords,
               kMaxAssignmentChunkBytes / std::max<size_t>(1, bytes_per_record)));
  std::vector<float> chunk_distances;
  std::vector<AssignmentTopCandidate> chunk_top_candidates;
  std::vector<uint32_t> chunk_chosen_partitions;
  std::vector<float> chunk_chosen_distances;
  if (!frozen_delta.records.empty()) {
    chunk_distances.resize(chunk_records * static_cast<size_t>(nlist));
    chunk_top_candidates.resize(chunk_records * static_cast<size_t>(top_r));
    chunk_chosen_partitions.resize(chunk_records);
    chunk_chosen_distances.resize(chunk_records);
  }

  double distance_us = 0.0;
  double top_r_us = 0.0;
  double balance_us = 0.0;
  double materialize_us = 0.0;
  uint32_t chunk_count_total = 0;
  Timer assignment_timer;

  for (size_t chunk_begin = 0; chunk_begin < frozen_delta.records.size();
       chunk_begin += chunk_records) {
    const size_t chunk_count =
        std::min(chunk_records, frozen_delta.records.size() - chunk_begin);
    ++chunk_count_total;

    Timer distance_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t local_index = 0; local_index < static_cast<int64_t>(chunk_count);
         ++local_index) {
      const auto& rec =
          frozen_delta.records[chunk_begin + static_cast<size_t>(local_index)];
      float* distances = chunk_distances.data() +
                         static_cast<size_t>(local_index) * static_cast<size_t>(nlist);
      for (uint32_t i = 0; i < nlist; ++i) {
        distances[static_cast<size_t>(i)] =
            (rec.x - main_centroids.row(static_cast<Eigen::Index>(i)).transpose())
                .squaredNorm();
      }
    }
    distance_us += distance_timer.ElapsedMicros();

    Timer top_r_timer;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      AssignmentScratch scratch;
      scratch.top_candidates.reserve(static_cast<size_t>(top_r));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int64_t local_index = 0; local_index < static_cast<int64_t>(chunk_count);
           ++local_index) {
        const float* distances = chunk_distances.data() +
                                 static_cast<size_t>(local_index) *
                                     static_cast<size_t>(nlist);
        scratch.top_candidates.clear();
        for (uint32_t i = 0; i < nlist; ++i) {
          InsertAssignmentTopCandidate(
              &scratch,
              AssignmentTopCandidate{distances[static_cast<size_t>(i)], i},
              top_r);
        }
        std::copy(scratch.top_candidates.begin(), scratch.top_candidates.end(),
                  chunk_top_candidates.begin() +
                      static_cast<size_t>(local_index) * static_cast<size_t>(top_r));
      }
    }
    top_r_us += top_r_timer.ElapsedMicros();

    Timer balance_timer;
    for (size_t local_index = 0; local_index < chunk_count; ++local_index) {
      const AssignmentTopCandidate* top_candidates =
          chunk_top_candidates.data() + local_index * static_cast<size_t>(top_r);
      const uint32_t nearest_partition = top_candidates[0].part;
      const float nearest_dist = top_candidates[0].dist;
      uint32_t chosen_partition = nearest_partition;
      float chosen_dist = nearest_dist;

      if (use_balanced_append) {
        const float dist_cap = nearest_dist * static_cast<float>(options.assignment_gamma);
        const double nearest_penalty = std::max(
            0.0,
            (static_cast<double>(projected_size[static_cast<size_t>(nearest_partition)]) +
             1.0 - avg_after) /
                balance_denom);
        const bool nearest_overloaded = nearest_penalty > 0.0;
        bool found_legal_candidate = false;
        bool found_healthier_candidate = false;
        AssignmentCandidateState best_legal_candidate;
        AssignmentCandidateState best_healthier_candidate;

        for (uint32_t k = 0; k < top_r; ++k) {
          const float dist = top_candidates[static_cast<size_t>(k)].dist;
          const uint32_t part = top_candidates[static_cast<size_t>(k)].part;
          if (dist > dist_cap ||
              static_cast<double>(projected_size[static_cast<size_t>(part)]) >= hard_cap) {
            continue;
          }
          AssignmentCandidateState candidate;
          candidate.part = part;
          candidate.dist = dist;
          candidate.penalty = std::max(
              0.0,
              (static_cast<double>(projected_size[static_cast<size_t>(part)]) + 1.0 -
               avg_after) /
                  balance_denom);
          candidate.cost =
              static_cast<double>(dist) + options.assignment_lambda * candidate.penalty;
          if (!found_legal_candidate || candidate.cost < best_legal_candidate.cost) {
            found_legal_candidate = true;
            best_legal_candidate = candidate;
          }
          if (nearest_overloaded && candidate.penalty + 1e-12 < nearest_penalty &&
              (!found_healthier_candidate ||
               candidate.cost < best_healthier_candidate.cost)) {
            found_healthier_candidate = true;
            best_healthier_candidate = candidate;
          }
        }

        const AssignmentCandidateState* selected = nullptr;
        if (found_healthier_candidate) {
          selected = &best_healthier_candidate;
        } else if (found_legal_candidate) {
          selected = &best_legal_candidate;
        }
        if (selected != nullptr) {
          chosen_partition = selected->part;
          chosen_dist = selected->dist;
        }
      }

      chunk_chosen_partitions[local_index] = chosen_partition;
      chunk_chosen_distances[local_index] = chosen_dist;
      projected_size[static_cast<size_t>(chosen_partition)]++;
      if (chosen_partition != nearest_partition) {
        ++moved_count;
      }
      const double ratio =
          nearest_dist > eps ? static_cast<double>(chosen_dist) / nearest_dist : 1.0;
      ratio_sum += ratio;
      ratio_max = std::max(ratio_max, ratio);
    }
    balance_us += balance_timer.ElapsedMicros();

    Timer materialize_timer;
    for (size_t local_index = 0; local_index < chunk_count; ++local_index) {
      const uint32_t chosen_partition = chunk_chosen_partitions[local_index];
      DeltaAssignment item;
      item.frozen_index = static_cast<uint32_t>(chunk_begin + local_index);
      item.main_partition = chosen_partition;
      item.residual_dist = chunk_chosen_distances[local_index];
      assigned[static_cast<size_t>(chosen_partition)].push_back(std::move(item));
    }
    materialize_us += materialize_timer.ElapsedMicros();
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
  if (profiling != nullptr) {
    profiling->merge_delta_to_main_assignment_us = assignment_timer.ElapsedMicros();
    profiling->merge_assignment_distance_us = distance_us;
    profiling->merge_assignment_top_r_us = top_r_us;
    profiling->merge_assignment_balance_us = balance_us;
    profiling->merge_assignment_materialize_us = materialize_us;
    profiling->assignment_distance_evaluations =
        static_cast<uint64_t>(frozen_delta.records.size()) * static_cast<uint64_t>(nlist);
    profiling->assignment_workspace_bytes =
        frozen_delta.records.empty()
            ? 0
            : static_cast<uint64_t>(chunk_records) * static_cast<uint64_t>(bytes_per_record);
    profiling->assignment_chunk_records = static_cast<uint32_t>(std::min<size_t>(
        chunk_records, static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    profiling->assignment_chunk_count = chunk_count_total;
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


Result<AlignedVector<VectorRecord>> MaterializeVectorRecordsFromAccessor(
    const std::vector<DocId>& doc_ids,
    const VersionSet& versions,
    uint32_t ivf_id,
    const VectorAccessor& vector_accessor) {
  MatrixRM rows;
  Status materialize = vector_accessor.Materialize(doc_ids, &rows);
  if (!materialize.ok()) {
    return materialize;
  }
  const uint32_t dim = vector_accessor.dim();
  if (rows.rows() != static_cast<Eigen::Index>(doc_ids.size()) ||
      rows.cols() != static_cast<Eigen::Index>(dim)) {
    return Status::InvalidArgument(
        "MaterializeVectorRecordsFromAccessor: materialized shape mismatch");
  }

  AlignedVector<VectorRecord> records(doc_ids.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && doc_ids.size() > 1)
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(doc_ids.size()); ++i) {
    VectorRecord rec;
    rec.doc_id = doc_ids[static_cast<size_t>(i)];
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = ivf_id;
    rec.x = rows.row(static_cast<Eigen::Index>(i)).transpose();
    records[static_cast<size_t>(i)] = std::move(rec);
  }
  return records;
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
  return MaterializeVectorRecordsFromAccessor(
      doc_ids_res.value(), versions, partition_id, *vector_accessor);
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
    std::vector<AlignedVector<VectorRecord>> neighborhood_main_records,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
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

  for (auto& records : neighborhood_main_records) {
    for (auto& rec : records) {
      pooled_records.push_back(std::move(rec));
    }
  }
  for (uint32_t part : neighborhood_partitions) {
    const auto& delta_bucket = assignments[static_cast<size_t>(part)];
    for (const auto& item : delta_bucket) {
      const size_t frozen_index = static_cast<size_t>(item.frozen_index);
      if (consumable_frozen_delta != nullptr) {
        pooled_records.push_back(
            std::move(consumable_frozen_delta->records[frozen_index]));
      } else {
        pooled_records.push_back(frozen_delta.records[frozen_index]);
      }
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

  struct CandidateState {
    uint32_t local_idx{0};
    double cost{0.0};
  };
  std::vector<std::pair<float, uint32_t>> dists;
  dists.reserve(neighborhood_n);

  for (auto& rec : pooled_records) {
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument("RepartitionNeighborhood: record dim mismatch");
    }
    dists.clear();
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

    bool found_legal_candidate = false;
    bool found_healthier_candidate = false;
    CandidateState best_legal_candidate;
    CandidateState best_healthier_candidate;

    for (const auto& dist_item : dists) {
      const float dist = dist_item.first;
      const uint32_t local_idx = dist_item.second;
      if (dist > dist_cap) {
        continue;
      }
      if (static_cast<double>(projected_size[static_cast<size_t>(local_idx)]) >= hard_cap) {
        continue;
      }
      const double penalty = std::max(
          0.0,
          (static_cast<double>(projected_size[static_cast<size_t>(local_idx)]) + 1.0) -
              avg_after) /
                     balance_denom;
      const CandidateState cand{
          local_idx, static_cast<double>(dist) + options.assignment_lambda * penalty};
      if (!found_legal_candidate || cand.cost < best_legal_candidate.cost) {
        found_legal_candidate = true;
        best_legal_candidate = cand;
      }
      if (nearest_overloaded && penalty + 1e-12 < nearest_penalty) {
        if (!found_healthier_candidate || cand.cost < best_healthier_candidate.cost) {
          found_healthier_candidate = true;
          best_healthier_candidate = cand;
        }
      }
    }

    if (found_healthier_candidate) {
      chosen_local = best_healthier_candidate.local_idx;
    } else if (found_legal_candidate) {
      chosen_local = best_legal_candidate.local_idx;
    }

    rec.ivf_id = neighborhood_partitions[static_cast<size_t>(chosen_local)];
    repartitioned[static_cast<size_t>(chosen_local)].push_back(std::move(rec));
    projected_size[static_cast<size_t>(chosen_local)]++;
  }

  for (auto& bucket : repartitioned) {
    std::sort(bucket.begin(), bucket.end(), [](const VectorRecord& lhs, const VectorRecord& rhs) {
      return lhs.doc_id < rhs.doc_id;
    });
  }
  return repartitioned;
}

Result<std::vector<std::vector<CompactRecord>>> RepartitionNeighborhoodCompact(
    const std::vector<uint32_t>& neighborhood_partitions,
    std::vector<AlignedVector<VectorRecord>> neighborhood_main_records,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    Eigen::Ref<const MatrixRM> main_centroids,
    const MergeOptions& options,
    MergeProfiling* profiling) {
  if (neighborhood_partitions.empty()) {
    return Status::InvalidArgument("RepartitionNeighborhoodCompact: empty neighborhood");
  }
  if (neighborhood_partitions.size() != neighborhood_main_records.size()) {
    return Status::InvalidArgument(
        "RepartitionNeighborhoodCompact: partition/main-records size mismatch");
  }

  const uint32_t neighborhood_n = static_cast<uint32_t>(neighborhood_partitions.size());
  for (uint32_t part : neighborhood_partitions) {
    if (part >= static_cast<uint32_t>(assignments.size())) {
      return Status::InvalidArgument(
          "RepartitionNeighborhoodCompact: partition out of assignments range");
    }
    if (part >= static_cast<uint32_t>(main_centroids.rows())) {
      return Status::InvalidArgument(
          "RepartitionNeighborhoodCompact: partition out of centroid range");
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

  for (auto& records : neighborhood_main_records) {
    for (auto& rec : records) {
      pooled_records.push_back(std::move(rec));
    }
  }
  for (uint32_t part : neighborhood_partitions) {
    const auto& delta_bucket = assignments[static_cast<size_t>(part)];
    for (const auto& item : delta_bucket) {
      const size_t frozen_index = static_cast<size_t>(item.frozen_index);
      if (consumable_frozen_delta != nullptr) {
        pooled_records.push_back(
            std::move(consumable_frozen_delta->records[frozen_index]));
      } else {
        pooled_records.push_back(frozen_delta.records[frozen_index]);
      }
    }
  }

  std::vector<std::vector<CompactRecord>> repartitioned(neighborhood_n);
  if (pooled_records.empty()) {
    return repartitioned;
  }
  for (const auto& rec : pooled_records) {
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument("RepartitionNeighborhoodCompact: record dim mismatch");
    }
  }

  const double avg_after =
      static_cast<double>(pooled_records.size()) / static_cast<double>(neighborhood_n);
  const double hard_cap = avg_after * options.assignment_hard_cap_ratio;
  const double balance_denom = std::max(1.0, avg_after);
  std::vector<uint32_t> projected_size(neighborhood_n, 0u);

  struct CandidateState {
    uint32_t local_idx{0};
    double cost{0.0};
  };
  std::vector<std::pair<float, uint32_t>> dists;
  dists.reserve(neighborhood_n);
  const size_t max_chunk_rows =
      std::min(kMergeRepartitionChunkRows, pooled_records.size());
  std::vector<float> distances(max_chunk_rows * neighborhood_n);
  std::vector<uint32_t> chosen_partitions(max_chunk_rows, 0u);

  for (size_t begin = 0; begin < pooled_records.size();
       begin += kMergeRepartitionChunkRows) {
    const size_t chunk_rows =
        std::min(kMergeRepartitionChunkRows, pooled_records.size() - begin);
    Timer distance_timer;
    auto compute_distances = [&](size_t row) {
      const VectorRecord& rec = pooled_records[begin + row];
      for (uint32_t local_idx = 0; local_idx < neighborhood_n; ++local_idx) {
        const uint32_t part = neighborhood_partitions[static_cast<size_t>(local_idx)];
        distances[row * neighborhood_n + local_idx] =
            (rec.x - main_centroids.row(static_cast<Eigen::Index>(part)).transpose())
                .squaredNorm();
      }
    };
#ifdef _OPENMP
    if (!omp_in_parallel() && chunk_rows > 1) {
#pragma omp parallel for schedule(static)
      for (int64_t row = 0; row < static_cast<int64_t>(chunk_rows); ++row) {
        compute_distances(static_cast<size_t>(row));
      }
    } else
#endif
    {
      for (size_t row = 0; row < chunk_rows; ++row) {
        compute_distances(row);
      }
    }
    if (profiling != nullptr) {
      profiling->prepare_distance_us += distance_timer.ElapsedMicros();
    }

    Timer balance_timer;
    for (size_t row = 0; row < chunk_rows; ++row) {
      dists.clear();
      for (uint32_t local_idx = 0; local_idx < neighborhood_n; ++local_idx) {
        dists.emplace_back(distances[row * neighborhood_n + local_idx], local_idx);
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

      bool found_legal_candidate = false;
      bool found_healthier_candidate = false;
      CandidateState best_legal_candidate;
      CandidateState best_healthier_candidate;
      for (const auto& dist_item : dists) {
        const float dist = dist_item.first;
        const uint32_t local_idx = dist_item.second;
        if (dist > dist_cap) {
          continue;
        }
        if (static_cast<double>(projected_size[static_cast<size_t>(local_idx)]) >= hard_cap) {
          continue;
        }
        const double penalty = std::max(
            0.0,
            (static_cast<double>(projected_size[static_cast<size_t>(local_idx)]) + 1.0) -
                avg_after) /
                balance_denom;
        const CandidateState cand{
            local_idx, static_cast<double>(dist) + options.assignment_lambda * penalty};
        if (!found_legal_candidate || cand.cost < best_legal_candidate.cost) {
          found_legal_candidate = true;
          best_legal_candidate = cand;
        }
        if (nearest_overloaded && penalty + 1e-12 < nearest_penalty) {
          if (!found_healthier_candidate || cand.cost < best_healthier_candidate.cost) {
            found_healthier_candidate = true;
            best_healthier_candidate = cand;
          }
        }
      }
      if (found_healthier_candidate) {
        chosen_local = best_healthier_candidate.local_idx;
      } else if (found_legal_candidate) {
        chosen_local = best_legal_candidate.local_idx;
      }
      chosen_partitions[row] = chosen_local;
      projected_size[static_cast<size_t>(chosen_local)]++;
    }
    if (profiling != nullptr) {
      profiling->prepare_balance_us += balance_timer.ElapsedMicros();
    }

    Timer emit_timer;
    for (size_t row = 0; row < chunk_rows; ++row) {
      const VectorRecord& rec = pooled_records[begin + row];
      repartitioned[static_cast<size_t>(chosen_partitions[row])].push_back(
          CompactRecord{rec.doc_id, rec.versions});
    }
    if (profiling != nullptr) {
      profiling->compact_emit_us += emit_timer.ElapsedMicros();
    }
  }

  Timer sort_timer;
  for (auto& bucket : repartitioned) {
    std::sort(bucket.begin(), bucket.end(), [](const CompactRecord& lhs, const CompactRecord& rhs) {
      return lhs.doc_id < rhs.doc_id;
    });
  }
  if (profiling != nullptr) {
    profiling->compact_emit_us += sort_timer.ElapsedMicros();
  }
  return repartitioned;
}

uint64_t EstimateCompactPatchBytes(const CompactPartitionPatch& patch) {
  uint64_t bytes = sizeof(CompactPartitionPatch);
  bytes += static_cast<uint64_t>(patch.partition_ids.capacity()) * sizeof(uint32_t);
  bytes += static_cast<uint64_t>(patch.replacement_records.capacity()) *
           sizeof(std::vector<CompactRecord>);
  for (const auto& records : patch.replacement_records) {
    bytes += static_cast<uint64_t>(records.capacity()) * sizeof(CompactRecord);
  }
  return bytes;
}

}  // namespace

Result<PartitionPatch> PreparePartitionPatchImpl(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MatrixRM* base_vectors,
    const MergeOptions& options,
    MergeProfiling* profiling);

Result<MergeReport> MergeFrozenDeltaIntoMainImpl(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 FrozenDelta* consumable_frozen_delta,
                                                 const MatrixRM* base_vectors,
                                                 const MergeOptions& options);

Result<PartitionPatch> PreparePartitionPatchImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options,
    MergeProfiling* profiling);

Result<MergeReport> MergeFrozenDeltaIntoMainImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
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

  constexpr size_t kMaterializeChunkRows = 65536;
  const auto& doc_ids = doc_ids_res.value();
  FrozenDelta out;
  out.records.reserve(doc_ids.size());
  std::vector<DocId> chunk_doc_ids;
  chunk_doc_ids.reserve(std::min(kMaterializeChunkRows, doc_ids.size()));
  for (size_t begin = 0; begin < doc_ids.size(); begin += kMaterializeChunkRows) {
    const size_t end = std::min(begin + kMaterializeChunkRows, doc_ids.size());
    chunk_doc_ids.assign(
        doc_ids.begin() + static_cast<std::ptrdiff_t>(begin),
        doc_ids.begin() + static_cast<std::ptrdiff_t>(end));
    auto records_res = MaterializeVectorRecordsFromAccessor(
        chunk_doc_ids, delta_versions, 0, vector_accessor);
    if (!records_res.ok()) {
      return records_res.status();
    }
    for (auto& rec : records_res.value()) {
      out.records.push_back(std::move(rec));
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
  for (size_t frozen_index = 0; frozen_index < frozen_delta.records.size();
       ++frozen_index) {
    const auto& rec = frozen_delta.records[frozen_index];
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument(
          "assign_delta_to_main_centroids: record dim mismatch with main centroids");
    }
    float residual_dist = 0.0f;
    const uint32_t partition = NearestCentroid(rec.x, main_centroids, &residual_dist);
    DeltaAssignment item;
    item.frozen_index = static_cast<uint32_t>(frozen_index);
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

Result<AlignedVector<VectorRecord>> MergePartitionAppendImpl(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const AlignedVector<DeltaAssignment>& delta_partition_records) {
  AlignedVector<VectorRecord> merged;
  merged.reserve(main_partition_records.size() + delta_partition_records.size());
  for (const auto& rec : main_partition_records) {
    VectorRecord out = rec;
    out.ivf_id = partition_id;
    merged.push_back(std::move(out));
  }
  for (const auto& item : delta_partition_records) {
    const size_t frozen_index = static_cast<size_t>(item.frozen_index);
    VectorRecord out;
    if (consumable_frozen_delta != nullptr) {
      out = std::move(consumable_frozen_delta->records[frozen_index]);
    } else {
      out = frozen_delta.records[frozen_index];
    }
    out.ivf_id = partition_id;
    merged.push_back(std::move(out));
  }
  return merged;
}

Result<AlignedVector<VectorRecord>> MergePartitionReclusterImpl(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const AlignedVector<DeltaAssignment>& delta_partition_records,
    const MergeOptions& options) {
  auto append_res = MergePartitionAppendImpl(partition_id,
                                             main_partition_records,
                                             frozen_delta,
                                             consumable_frozen_delta,
                                             delta_partition_records);
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

Result<AlignedVector<VectorRecord>> merge_partition_append(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const FrozenDelta& frozen_delta,
    const AlignedVector<DeltaAssignment>& delta_partition_records) {
  return MergePartitionAppendImpl(
      partition_id, main_partition_records, frozen_delta, nullptr, delta_partition_records);
}

Result<AlignedVector<VectorRecord>> merge_partition_recluster(
    uint32_t partition_id,
    const AlignedVector<VectorRecord>& main_partition_records,
    const FrozenDelta& frozen_delta,
    const AlignedVector<DeltaAssignment>& delta_partition_records,
    const MergeOptions& options) {
  return MergePartitionReclusterImpl(partition_id,
                                     main_partition_records,
                                     frozen_delta,
                                     nullptr,
                                     delta_partition_records,
                                     options);
}

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
  const MergeOptions& options) {
  return PreparePartitionPatchImpl(
      main_ivf,
      main_versions,
      frozen_delta,
      nullptr,
      assignments,
      score_result,
      nullptr,
      options,
      nullptr);
}

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
  const MatrixRM& base_vectors,
  const MergeOptions& options) {
  return PreparePartitionPatchImpl(
      main_ivf,
      main_versions,
      frozen_delta,
      nullptr,
      assignments,
      score_result,
      &base_vectors,
      options,
      nullptr);
}

Result<PartitionPatch> PreparePartitionPatchImpl(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const MatrixRM* base_vectors,
    const MergeOptions& options,
    MergeProfiling* profiling) {
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
    if (profiling != nullptr) {
      profiling->seed_partitions += static_cast<uint64_t>(seeds.size());
    }

    PartitionPatch patch;
    std::vector<uint8_t> claimed(assignments.size(), 0u);
    for (uint32_t seed : seeds) {
      if (claimed[static_cast<size_t>(seed)] != 0u) {
        continue;
      }
      Timer neighbor_timer;
      std::vector<uint32_t> neighborhood =
          TopRNeighborPartitions(seed, centroids_res.value(), options.assignment_top_r);
      if (profiling != nullptr) {
        profiling->top_r_neighbor_us += neighbor_timer.ElapsedMicros();
      }
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
      if (profiling != nullptr) {
        ++profiling->neighborhoods;
      }

      std::vector<AlignedVector<VectorRecord>> neighborhood_main_records;
      neighborhood_main_records.reserve(active_neighborhood.size());
      for (uint32_t part : active_neighborhood) {
        Timer fetch_timer;
        auto main_records_res =
            FetchPartitionRecordsForMerge(main_ivf, main_versions, part, base_vectors);
        if (profiling != nullptr) {
          profiling->fetch_main_records_us += fetch_timer.ElapsedMicros();
        }
        if (!main_records_res.ok()) {
          return main_records_res.status();
        }
        if (profiling != nullptr) {
          profiling->main_records_loaded +=
              static_cast<uint64_t>(main_records_res.value().size());
        }
        neighborhood_main_records.push_back(std::move(main_records_res.value()));
      }

      if (profiling != nullptr) {
        uint64_t pooled = 0;
        for (size_t i = 0; i < active_neighborhood.size(); ++i) {
          pooled += static_cast<uint64_t>(neighborhood_main_records[i].size());
          pooled += static_cast<uint64_t>(
              assignments[static_cast<size_t>(active_neighborhood[i])].size());
        }
        profiling->pooled_records += pooled;
      }
      Timer repartition_timer;
      auto repartition_res = RepartitionNeighborhood(active_neighborhood,
                                                     std::move(neighborhood_main_records),
                                                     frozen_delta,
                                                     consumable_frozen_delta,
                                                     assignments,
                                                     centroids_res.value(),
                                                     options);
      if (profiling != nullptr) {
        profiling->repartition_us += repartition_timer.ElapsedMicros();
      }
      if (!repartition_res.ok()) {
        return repartition_res.status();
      }
      std::vector<AlignedVector<VectorRecord>> repartitioned = std::move(repartition_res.value());
      if (profiling != nullptr) {
        for (const auto& records : repartitioned) {
          profiling->repartitioned_records += static_cast<uint64_t>(records.size());
        }
      }
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
  if (profiling != nullptr) {
    profiling->seed_partitions += static_cast<uint64_t>(active_partitions.size());
    profiling->neighborhoods += static_cast<uint64_t>(active_partitions.size());
  }
  std::vector<double> fetch_profile_us(
      profiling != nullptr ? active_partitions.size() : 0, 0.0);
  std::vector<double> repartition_profile_us(
      profiling != nullptr ? active_partitions.size() : 0, 0.0);

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
    Timer fetch_timer;
    auto main_records_res =
        FetchPartitionRecordsForMerge(main_ivf, main_versions, p, base_vectors);
    if (profiling != nullptr) {
      fetch_profile_us[static_cast<size_t>(i)] = fetch_timer.ElapsedMicros();
    }
    if (!main_records_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = main_records_res.status();
      }
      continue;
    }
    Timer repartition_timer;
    Result<AlignedVector<VectorRecord>> merged_res =
        recluster_flags[static_cast<size_t>(p)] != 0
            ? MergePartitionReclusterImpl(p,
                                          main_records_res.value(),
                                          frozen_delta,
                                          consumable_frozen_delta,
                                          delta_bucket,
                                          options)
            : MergePartitionAppendImpl(p,
                                       main_records_res.value(),
                                       frozen_delta,
                                       consumable_frozen_delta,
                                       delta_bucket);
    if (!merged_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = merged_res.status();
      }
      continue;
    }
    if (profiling != nullptr) {
      repartition_profile_us[static_cast<size_t>(i)] = repartition_timer.ElapsedMicros();
    }
    patch.replacement_records[static_cast<size_t>(i)] = std::move(merged_res.value());
  }
  if (failed.load(std::memory_order_relaxed)) {
    return first_error;
  }
  if (profiling != nullptr) {
    for (size_t i = 0; i < active_partitions.size(); ++i) {
      const uint64_t replacement_size =
          static_cast<uint64_t>(patch.replacement_records[i].size());
      const uint64_t delta_size = static_cast<uint64_t>(
          assignments[static_cast<size_t>(active_partitions[i])].size());
      profiling->fetch_main_records_us += fetch_profile_us[i];
      profiling->repartition_us += repartition_profile_us[i];
      profiling->main_records_loaded += replacement_size - delta_size;
      profiling->pooled_records += replacement_size;
      profiling->repartitioned_records += replacement_size;
    }
  }
  return patch;
}

Result<PartitionPatch> PreparePartitionPatchImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options,
    MergeProfiling* profiling) {
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
    if (profiling != nullptr) {
      profiling->seed_partitions += static_cast<uint64_t>(seeds.size());
    }

    PartitionPatch patch;
    std::vector<uint8_t> claimed(assignments.size(), 0u);
    for (uint32_t seed : seeds) {
      if (claimed[static_cast<size_t>(seed)] != 0u) {
        continue;
      }
      Timer neighbor_timer;
      std::vector<uint32_t> neighborhood =
          TopRNeighborPartitions(seed, centroids_res.value(), options.assignment_top_r);
      if (profiling != nullptr) {
        profiling->top_r_neighbor_us += neighbor_timer.ElapsedMicros();
      }
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
      if (profiling != nullptr) {
        ++profiling->neighborhoods;
      }

      std::vector<AlignedVector<VectorRecord>> neighborhood_main_records;
      neighborhood_main_records.reserve(active_neighborhood.size());
      for (uint32_t part : active_neighborhood) {
        Timer fetch_timer;
        auto main_records_res =
            FetchPartitionRecordsForMergeAccessor(main_ivf, main_versions, part, vector_accessor);
        if (profiling != nullptr) {
          profiling->fetch_main_records_us += fetch_timer.ElapsedMicros();
        }
        if (!main_records_res.ok()) {
          return main_records_res.status();
        }
        if (profiling != nullptr) {
          profiling->main_records_loaded +=
              static_cast<uint64_t>(main_records_res.value().size());
        }
        neighborhood_main_records.push_back(std::move(main_records_res.value()));
      }

      if (profiling != nullptr) {
        uint64_t pooled = 0;
        for (size_t i = 0; i < active_neighborhood.size(); ++i) {
          pooled += static_cast<uint64_t>(neighborhood_main_records[i].size());
          pooled += static_cast<uint64_t>(
              assignments[static_cast<size_t>(active_neighborhood[i])].size());
        }
        profiling->pooled_records += pooled;
      }
      Timer repartition_timer;
      auto repartition_res = RepartitionNeighborhood(active_neighborhood,
                                                     std::move(neighborhood_main_records),
                                                     frozen_delta,
                                                     consumable_frozen_delta,
                                                     assignments,
                                                     centroids_res.value(),
                                                     options);
      if (profiling != nullptr) {
        profiling->repartition_us += repartition_timer.ElapsedMicros();
      }
      if (!repartition_res.ok()) {
        return repartition_res.status();
      }
      std::vector<AlignedVector<VectorRecord>> repartitioned = std::move(repartition_res.value());
      if (profiling != nullptr) {
        for (const auto& records : repartitioned) {
          profiling->repartitioned_records += static_cast<uint64_t>(records.size());
        }
      }
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
  if (profiling != nullptr) {
    profiling->seed_partitions += static_cast<uint64_t>(active_partitions.size());
    profiling->neighborhoods += static_cast<uint64_t>(active_partitions.size());
  }
  std::vector<double> fetch_profile_us(
      profiling != nullptr ? active_partitions.size() : 0, 0.0);
  std::vector<double> repartition_profile_us(
      profiling != nullptr ? active_partitions.size() : 0, 0.0);

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
    Timer fetch_timer;
    auto main_records_res =
        FetchPartitionRecordsForMergeAccessor(main_ivf, main_versions, p, vector_accessor);
    if (profiling != nullptr) {
      fetch_profile_us[static_cast<size_t>(i)] = fetch_timer.ElapsedMicros();
    }
    if (!main_records_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = main_records_res.status();
      }
      continue;
    }
    Timer repartition_timer;
    Result<AlignedVector<VectorRecord>> merged_res =
        recluster_flags[static_cast<size_t>(p)] != 0
            ? MergePartitionReclusterImpl(p,
                                          main_records_res.value(),
                                          frozen_delta,
                                          consumable_frozen_delta,
                                          delta_bucket,
                                          options)
            : MergePartitionAppendImpl(p,
                                       main_records_res.value(),
                                       frozen_delta,
                                       consumable_frozen_delta,
                                       delta_bucket);
    if (!merged_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = merged_res.status();
      }
      continue;
    }
    if (profiling != nullptr) {
      repartition_profile_us[static_cast<size_t>(i)] = repartition_timer.ElapsedMicros();
    }
    patch.replacement_records[static_cast<size_t>(i)] = std::move(merged_res.value());
  }
  if (failed.load(std::memory_order_relaxed)) {
    return first_error;
  }
  if (profiling != nullptr) {
    for (size_t i = 0; i < active_partitions.size(); ++i) {
      const uint64_t replacement_size =
          static_cast<uint64_t>(patch.replacement_records[i].size());
      const uint64_t delta_size = static_cast<uint64_t>(
          assignments[static_cast<size_t>(active_partitions[i])].size());
      profiling->fetch_main_records_us += fetch_profile_us[i];
      profiling->repartition_us += repartition_profile_us[i];
      profiling->main_records_loaded += replacement_size - delta_size;
      profiling->pooled_records += replacement_size;
      profiling->repartitioned_records += replacement_size;
    }
  }
  return patch;
}

Result<CompactPartitionPatch> PrepareCompactPartitionPatchImplAccessor(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options,
    MergeProfiling* profiling) {
  if (!main_ivf) {
    return Status::InvalidArgument("prepare_compact_partition_patch: main_ivf is null");
  }
  if (vector_accessor == nullptr) {
    return Status::InvalidArgument(
        "prepare_compact_partition_patch: vector_accessor is null");
  }

  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    return centroids_res.status();
  }
  if (assignments.size() != static_cast<size_t>(centroids_res.value().rows())) {
    return Status::InvalidArgument(
        "prepare_compact_partition_patch: assignments size mismatch with centroids");
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
  if (profiling != nullptr) {
    profiling->seed_partitions += static_cast<uint64_t>(seeds.size());
  }

  CompactPartitionPatch patch;
  std::vector<uint8_t> claimed(assignments.size(), 0u);
  for (uint32_t seed : seeds) {
    if (claimed[static_cast<size_t>(seed)] != 0u) {
      continue;
    }
    Timer neighbor_timer;
    std::vector<uint32_t> neighborhood =
        TopRNeighborPartitions(seed, centroids_res.value(), options.assignment_top_r);
    if (profiling != nullptr) {
      profiling->top_r_neighbor_us += neighbor_timer.ElapsedMicros();
    }
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
    if (profiling != nullptr) {
      ++profiling->neighborhoods;
    }

    std::vector<AlignedVector<VectorRecord>> neighborhood_main_records;
    neighborhood_main_records.reserve(active_neighborhood.size());
    for (uint32_t part : active_neighborhood) {
      Timer fetch_timer;
      auto main_records_res =
          FetchPartitionRecordsForMergeAccessor(main_ivf, main_versions, part, vector_accessor);
      const double fetch_us = fetch_timer.ElapsedMicros();
      if (profiling != nullptr) {
        profiling->fetch_main_records_us += fetch_us;
        profiling->prepare_fetch_us += fetch_us;
      }
      if (!main_records_res.ok()) {
        return main_records_res.status();
      }
      if (profiling != nullptr) {
        profiling->main_records_loaded +=
            static_cast<uint64_t>(main_records_res.value().size());
      }
      neighborhood_main_records.push_back(std::move(main_records_res.value()));
    }

    if (profiling != nullptr) {
      uint64_t pooled = 0;
      for (size_t i = 0; i < active_neighborhood.size(); ++i) {
        pooled += static_cast<uint64_t>(neighborhood_main_records[i].size());
        pooled += static_cast<uint64_t>(
            assignments[static_cast<size_t>(active_neighborhood[i])].size());
      }
      profiling->pooled_records += pooled;
    }
    Timer repartition_timer;
    auto repartition_res = RepartitionNeighborhoodCompact(active_neighborhood,
                                                           std::move(neighborhood_main_records),
                                                           frozen_delta,
                                                           consumable_frozen_delta,
                                                           assignments,
                                                           centroids_res.value(),
                                                           options,
                                                           profiling);
    if (profiling != nullptr) {
      profiling->repartition_us += repartition_timer.ElapsedMicros();
    }
    if (!repartition_res.ok()) {
      return repartition_res.status();
    }
    std::vector<std::vector<CompactRecord>> repartitioned =
        std::move(repartition_res.value());
    if (profiling != nullptr) {
      for (const auto& records : repartitioned) {
        profiling->repartitioned_records += static_cast<uint64_t>(records.size());
      }
    }
    for (size_t i = 0; i < active_neighborhood.size(); ++i) {
      patch.partition_ids.push_back(active_neighborhood[i]);
      patch.replacement_records.push_back(std::move(repartitioned[i]));
      claimed[static_cast<size_t>(active_neighborhood[i])] = 1u;
    }
  }

  if (profiling != nullptr) {
    for (const auto& records : patch.replacement_records) {
      profiling->compact_patch_records += static_cast<uint64_t>(records.size());
    }
    profiling->compact_patch_estimated_bytes = EstimateCompactPatchBytes(patch);
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
      main_ivf, main_versions, frozen_delta, nullptr, nullptr, options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
  const MatrixRM& base_vectors,
  const MergeOptions& options) {
  return MergeFrozenDeltaIntoMainImpl(
      main_ivf, main_versions, frozen_delta, nullptr, &base_vectors, options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
  const VectorAccessor& vector_accessor,
  const MergeOptions& options) {
  return MergeFrozenDeltaIntoMainImplAccessor(
      main_ivf, main_versions, frozen_delta, nullptr, &vector_accessor, options);
}

Result<MergeReport> MergeFrozenDeltaIntoMainImpl(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 FrozenDelta* consumable_frozen_delta,
                                                 const MatrixRM* base_vectors,
                                                 const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("merge_frozen_delta_into_main: main_ivf is null");
  }
  MergeProfiling profiling;
  profiling.frozen_records = static_cast<uint64_t>(frozen_delta.records.size());
  Timer merge_compute_timer;
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    return centroids_res.status();
  }
  profiling.effective_nlist = static_cast<uint32_t>(centroids_res.value().rows());
  auto sizes_res = main_ivf->GetPartitionSizes(main_versions);
  if (!sizes_res.ok()) {
    return sizes_res.status();
  }
  AssignmentDiagnostics assignment_diag;
  auto assign_res = assign_delta_to_main_centroids_for_merge(
      frozen_delta,
      centroids_res.value(),
      sizes_res.value(),
      options,
      &assignment_diag,
      &profiling);
  if (!assign_res.ok()) {
    return assign_res.status();
  }
  Timer stats_timer;
  auto stats_res = compute_partition_stats(assign_res.value(), sizes_res.value());
  if (!stats_res.ok()) {
    return stats_res.status();
  }
  profiling.stats_us = stats_timer.ElapsedMicros();
  Timer scoring_timer;
  auto score_res = score_partitions(
      stats_res.value(), options.alpha, options.beta, options.recluster_threshold);
  if (!score_res.ok()) {
    return score_res.status();
  }
  profiling.scoring_us = scoring_timer.ElapsedMicros();
  Timer patch_prepare_timer;
  auto patch_res = PreparePartitionPatchImpl(
      main_ivf,
      main_versions,
      frozen_delta,
      consumable_frozen_delta,
      assign_res.value(),
      score_res.value(),
      base_vectors,
      options,
      &profiling);
  if (!patch_res.ok()) {
    return patch_res.status();
  }
  profiling.patch_prepare_us = patch_prepare_timer.ElapsedMicros();
  for (const auto& records : patch_res.value().replacement_records) {
    profiling.patch_records += static_cast<uint64_t>(records.size());
  }
  const double merge_compute_ms = merge_compute_timer.ElapsedMillis();
  Timer commit_timer;
  Status commit = commit_partition_patch(main_ivf, main_versions, patch_res.value());
  if (!commit.ok()) {
    return commit;
  }
  profiling.commit_us = commit_timer.ElapsedMicros();
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
  profiling.pq_code_assignment_us = codebook_ms_res.value() * 1000.0;
  report.profiling = profiling;
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
                                                 FrozenDelta* consumable_frozen_delta,
                                                 const VectorAccessor* vector_accessor,
    const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("merge_frozen_delta_into_main: main_ivf is null");
  }
  MergeProfiling profiling;
  profiling.frozen_records = static_cast<uint64_t>(frozen_delta.records.size());
  Timer merge_compute_timer;
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    return centroids_res.status();
  }
  profiling.effective_nlist = static_cast<uint32_t>(centroids_res.value().rows());
  auto sizes_res = main_ivf->GetPartitionSizes(main_versions);
  if (!sizes_res.ok()) {
    return sizes_res.status();
  }
  AssignmentDiagnostics assignment_diag;
  auto assign_res = assign_delta_to_main_centroids_for_merge(
      frozen_delta,
      centroids_res.value(),
      sizes_res.value(),
      options,
      &assignment_diag,
      &profiling);
  if (!assign_res.ok()) {
    return assign_res.status();
  }
  Timer stats_timer;
  auto stats_res = compute_partition_stats(assign_res.value(), sizes_res.value());
  if (!stats_res.ok()) {
    return stats_res.status();
  }
  profiling.stats_us = stats_timer.ElapsedMicros();
  Timer scoring_timer;
  auto score_res = score_partitions(
      stats_res.value(), options.alpha, options.beta, options.recluster_threshold);
  if (!score_res.ok()) {
    return score_res.status();
  }
  profiling.scoring_us = scoring_timer.ElapsedMicros();
  bool use_compact_patch = false;
  {
    auto pq_state_res = main_ivf->GetPQRuntimeState(main_versions);
    if (!pq_state_res.ok()) {
      return pq_state_res.status();
    }
    use_compact_patch = vector_accessor != nullptr && pq_state_res.value().use_pq &&
                        options.assignment_mode == "balanced_append";
  }

  PartitionPatch raw_patch;
  CompactPartitionPatch compact_patch;
  Timer patch_prepare_timer;
  if (use_compact_patch) {
    auto patch_res = PrepareCompactPartitionPatchImplAccessor(main_ivf,
                                                               main_versions,
                                                               frozen_delta,
                                                               consumable_frozen_delta,
                                                               assign_res.value(),
                                                               vector_accessor,
                                                               options,
                                                               &profiling);
    if (!patch_res.ok()) {
      return patch_res.status();
    }
    compact_patch = std::move(patch_res.value());
    profiling.patch_records = profiling.compact_patch_records;
  } else {
    auto patch_res = PreparePartitionPatchImplAccessor(main_ivf,
                                                        main_versions,
                                                        frozen_delta,
                                                        consumable_frozen_delta,
                                                        assign_res.value(),
                                                        score_res.value(),
                                                        vector_accessor,
                                                        options,
                                                        &profiling);
    if (!patch_res.ok()) {
      return patch_res.status();
    }
    raw_patch = std::move(patch_res.value());
    for (const auto& records : raw_patch.replacement_records) {
      profiling.patch_records += static_cast<uint64_t>(records.size());
    }
  }
  profiling.patch_prepare_us = patch_prepare_timer.ElapsedMicros();
  profiling.prepare_total_us = profiling.patch_prepare_us;
  if (!use_compact_patch) {
    profiling.prepare_fetch_us = profiling.fetch_main_records_us;
  }
  const double merge_compute_ms = merge_compute_timer.ElapsedMillis();
  Timer commit_timer;
  Status commit = Status::OK();
  if (use_compact_patch) {
    CompactPatchCommitProfiling commit_profile;
    commit = main_ivf->CommitCompactPartitionPatch(
        main_versions, compact_patch, *vector_accessor, &commit_profile);
    profiling.commit_total_us = commit_profile.commit_total_us;
    profiling.commit_validation_us = commit_profile.commit_validation_us;
    profiling.commit_reuse_classify_us = commit_profile.commit_reuse_classify_us;
    profiling.commit_materialize_us = commit_profile.commit_materialize_us;
    profiling.commit_materialized_rows = commit_profile.commit_materialized_rows;
    profiling.commit_materialized_bytes = commit_profile.commit_materialized_bytes;
    profiling.commit_materialize_batches = commit_profile.commit_materialize_batches;
    profiling.commit_max_materialize_rows = commit_profile.commit_max_materialize_rows;
    profiling.commit_pq_encode_us = commit_profile.commit_pq_encode_us;
    profiling.pq_codes_reused = commit_profile.pq_codes_reused;
    profiling.pq_codes_reencoded = commit_profile.pq_codes_reencoded;
    profiling.commit_apply_us = commit_profile.commit_apply_us;
    profiling.pq_list_flatten_us = commit_profile.pq_list_flatten_us;
    profiling.docmap_rebuild_us = commit_profile.docmap_rebuild_us;
    profiling.commit_lock_hold_us = commit_profile.commit_lock_hold_us;
  } else {
    commit = commit_partition_patch(main_ivf, main_versions, raw_patch);
  }
  if (!commit.ok()) {
    return commit;
  }
  profiling.commit_us = commit_timer.ElapsedMicros();
  if (!use_compact_patch) {
    profiling.commit_total_us = profiling.commit_us;
  }
  auto codebook_ms_res = main_ivf->GetLastPatchPQReencodeMs(main_versions);
  if (!codebook_ms_res.ok()) {
    return codebook_ms_res.status();
  }

  MergeReport report;
  report.frozen_records = static_cast<uint32_t>(frozen_delta.records.size());
  report.patch_partitions = static_cast<uint32_t>(
      use_compact_patch ? compact_patch.partition_ids.size() : raw_patch.partition_ids.size());
  report.append_partitions = static_cast<uint32_t>(score_res.value().append_partitions.size());
  report.recluster_partitions = static_cast<uint32_t>(score_res.value().recluster_partitions.size());
  report.merge_compute_ms = merge_compute_ms;
  report.codebook_rebuild_ms = codebook_ms_res.value();
  profiling.pq_code_assignment_us = codebook_ms_res.value() * 1000.0;
  report.profiling = profiling;
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
  return MergeFrozenDeltaIntoMainImpl(main_ivf,
                                      main_versions,
                                      frozen_res.value(),
                                      &frozen_res.value(),
                                      nullptr,
                                      options);
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
  return MergeFrozenDeltaIntoMainImpl(main_ivf,
                                      main_versions,
                                      frozen_res.value(),
                                      &frozen_res.value(),
                                      &base_vectors,
                                      options);
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
  return MergeFrozenDeltaIntoMainImplAccessor(main_ivf,
                                              main_versions,
                                              frozen_res.value(),
                                              &frozen_res.value(),
                                              &vector_accessor,
                                              options);
}

}  // namespace ann
