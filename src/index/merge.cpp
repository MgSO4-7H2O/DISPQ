#include "index/merge.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iterator>
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

using ProfilingClock = std::chrono::steady_clock;

struct VectorAccessorMaterializationStats {
  uint64_t calls{0};
  uint64_t rows{0};
  uint64_t bytes{0};
  uint32_t max_rows{0};
};

void AccumulateVectorAccessorMaterialization(
    const VectorAccessorMaterializationStats& source,
    MergeProfiling* destination) {
  if (destination == nullptr) {
    return;
  }
  destination->vector_accessor_materialize_calls += source.calls;
  destination->vector_accessor_materialized_rows += source.rows;
  destination->vector_accessor_materialized_bytes += source.bytes;
  destination->vector_accessor_max_materialize_rows = std::max(
      destination->vector_accessor_max_materialize_rows, source.max_rows);
}

void AccumulateNeighborhoodProfiling(const MergeProfiling& source,
                                     MergeProfiling* destination) {
  if (destination == nullptr) {
    return;
  }
  destination->fetch_main_records_us += source.fetch_main_records_us;
  destination->main_records_loaded += source.main_records_loaded;
  destination->vector_accessor_materialize_calls += source.vector_accessor_materialize_calls;
  destination->vector_accessor_materialized_rows += source.vector_accessor_materialized_rows;
  destination->vector_accessor_materialized_bytes += source.vector_accessor_materialized_bytes;
  destination->vector_accessor_max_materialize_rows = std::max(
      destination->vector_accessor_max_materialize_rows,
      source.vector_accessor_max_materialize_rows);
  destination->pooled_records += source.pooled_records;
  destination->repartitioned_records += source.repartitioned_records;
  destination->repartition_pool_us += source.repartition_pool_us;
  destination->repartition_distance_us += source.repartition_distance_us;
  destination->repartition_candidate_selection_us +=
      source.repartition_candidate_selection_us;
  destination->repartition_sort_us += source.repartition_sort_us;
  destination->patch_dense_vector_bytes += source.patch_dense_vector_bytes;
  destination->patch_retained_vector_bytes += source.patch_retained_vector_bytes;
  destination->patch_elided_vector_bytes += source.patch_elided_vector_bytes;
  destination->patch_final_pq_code_bytes += source.patch_final_pq_code_bytes;
  destination->prepare_pq_codes_reused += source.prepare_pq_codes_reused;
  destination->prepare_pq_codes_reencoded += source.prepare_pq_codes_reencoded;
  destination->prepare_pq_encode_us += source.prepare_pq_encode_us;
}

bool SamePQEncodingContext(const PQEncodingContext& lhs,
                           const PQEncodingContext& rhs) {
  return lhs.index_version == rhs.index_version &&
         lhs.mutation_generation == rhs.mutation_generation &&
         lhs.use_pq == rhs.use_pq && lhs.pq_residual == rhs.pq_residual &&
         lhs.M == rhs.M && lhs.nbits == rhs.nbits && lhs.Ks == rhs.Ks &&
         lhs.dsub == rhs.dsub;
}

struct PQFinalizationResult {
  std::optional<PreparedPQPayload> prepared_pq;
  uint64_t dense_vector_bytes{0};
  uint64_t retained_vector_bytes{0};
  uint64_t final_pq_code_bytes{0};
};

Result<PQFinalizationResult> FinalizePartitionPQCodes(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const std::vector<uint32_t>& partition_ids,
    std::vector<AlignedVector<VectorRecord>>* replacement_records) {
  if (replacement_records == nullptr ||
      partition_ids.size() != replacement_records->size()) {
    return Status::InvalidArgument(
        "FinalizePartitionPQCodes: partition/record shape mismatch");
  }

  PQFinalizationResult result;
  for (const auto& records : *replacement_records) {
    for (const auto& rec : records) {
      result.dense_vector_bytes +=
          static_cast<uint64_t>(rec.x.size()) * sizeof(float);
    }
  }
  auto prepared_res = main_ivf->PreparePartitionPQCodes(
      main_versions, partition_ids, *replacement_records);
  if (!prepared_res.ok()) {
    return prepared_res.status();
  }
  result.prepared_pq = std::move(prepared_res.value());
  if (result.prepared_pq.has_value()) {
    for (const auto& partition : result.prepared_pq->partition_codes) {
      result.final_pq_code_bytes +=
          static_cast<uint64_t>(partition.final_codes.size());
    }
    // Each final code is now owned by the prepared payload. Swapping with an
    // empty vector releases Eigen's per-record backing allocation immediately.
    for (auto& records : *replacement_records) {
      for (auto& rec : records) {
        Eigen::VectorXf empty;
        rec.x.swap(empty);
      }
    }
  }
  for (const auto& records : *replacement_records) {
    for (const auto& rec : records) {
      result.retained_vector_bytes +=
          static_cast<uint64_t>(rec.x.size()) * sizeof(float);
    }
  }
  return result;
}

Status AppendPreparedPQPayload(std::optional<PreparedPQPayload> batch,
                               size_t partition_reserve_hint,
                               PartitionPatch* patch) {
  if (!batch.has_value()) {
    if (patch->prepared_pq.has_value()) {
      return Status::InvalidArgument(
          "PreparePartitionPatch: mixed PQ and dense preparation state");
    }
    return Status::OK();
  }
  if (!patch->prepared_pq.has_value()) {
    PreparedPQPayload combined;
    combined.context = batch->context;
    combined.partition_codes.reserve(partition_reserve_hint);
    patch->prepared_pq = std::move(combined);
  } else if (!SamePQEncodingContext(patch->prepared_pq->context,
                                    batch->context)) {
    return Status::InvalidArgument(
        "PreparePartitionPatch: IVF changed between PQ preparation batches");
  }
  patch->prepared_pq->partition_codes.insert(
      patch->prepared_pq->partition_codes.end(),
      std::make_move_iterator(batch->partition_codes.begin()),
      std::make_move_iterator(batch->partition_codes.end()));
  patch->prepared_pq->codes_reused += batch->codes_reused;
  patch->prepared_pq->codes_reencoded += batch->codes_reencoded;
  patch->prepared_pq->encode_us += batch->encode_us;
  patch->prepared_pq->copy_or_reuse_us += batch->copy_or_reuse_us;
  return Status::OK();
}

void AccumulatePQFinalizationProfiling(const PQFinalizationResult& result,
                                       MergeProfiling* profiling) {
  if (profiling == nullptr) {
    return;
  }
  profiling->patch_dense_vector_bytes += result.dense_vector_bytes;
  profiling->patch_retained_vector_bytes += result.retained_vector_bytes;
  profiling->patch_elided_vector_bytes +=
      result.dense_vector_bytes - result.retained_vector_bytes;
  profiling->patch_final_pq_code_bytes += result.final_pq_code_bytes;
  if (result.prepared_pq.has_value()) {
    profiling->prepare_pq_codes_reused += result.prepared_pq->codes_reused;
    profiling->prepare_pq_codes_reencoded += result.prepared_pq->codes_reencoded;
    profiling->prepare_pq_encode_us += result.prepared_pq->encode_us;
  }
}

double ElapsedProfilingMicros(ProfilingClock::time_point start) {
  return std::chrono::duration<double, std::micro>(ProfilingClock::now() - start).count();
}

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

constexpr uint32_t kFastAssignmentTopR = 32;

struct AssignmentCandidateState {
  uint32_t part{0};
  float dist{0.0f};
  double penalty{0.0};
  double cost{0.0};
};

struct AssignmentScratch {
  std::array<AssignmentTopCandidate, kFastAssignmentTopR> fixed_candidates{};
  std::vector<AssignmentTopCandidate> fallback_candidates;
  uint32_t fixed_size{0};

  void Reserve(uint32_t top_r) {
    if (top_r > kFastAssignmentTopR) {
      fallback_candidates.reserve(top_r);
    }
  }

  void Clear(uint32_t top_r) {
    if (top_r <= kFastAssignmentTopR) {
      fixed_size = 0;
    } else {
      fallback_candidates.clear();
    }
  }

  const AssignmentTopCandidate* Data(uint32_t top_r) const {
    return top_r <= kFastAssignmentTopR ? fixed_candidates.data()
                                       : fallback_candidates.data();
  }
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
  if (top_r <= kFastAssignmentTopR) {
    auto& top_candidates = scratch->fixed_candidates;
    uint32_t& size = scratch->fixed_size;
    if (size == top_r &&
        !IsAssignmentCandidateLess(candidate, top_candidates[top_r - 1])) {
      return;
    }

    uint32_t insert_pos = 0;
    while (insert_pos < size &&
           IsAssignmentCandidateLess(top_candidates[insert_pos], candidate)) {
      ++insert_pos;
    }
    const uint32_t last = std::min(size, top_r - 1);
    for (uint32_t i = last; i > insert_pos; --i) {
      top_candidates[i] = top_candidates[i - 1];
    }
    top_candidates[insert_pos] = candidate;
    if (size < top_r) {
      ++size;
    }
    return;
  }

  auto& top_candidates = scratch->fallback_candidates;
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
  double distance_us = 0.0;
  double top_r_us = 0.0;
  double balance_us = 0.0;
  double materialize_us = 0.0;
  const auto assignment_start = ProfilingClock::now();

  for (const auto& rec : frozen_delta.records) {
    if (rec.x.size() != main_centroids.cols()) {
      return Status::InvalidArgument(
          "assign_delta_to_main_centroids_for_merge: record dim mismatch with main centroids");
    }
  }

  constexpr size_t kMaxAssignmentChunkBytes = 64ULL * 1024ULL * 1024ULL;
  Eigen::VectorXf centroid_norms(main_centroids.rows());
  for (uint32_t i = 0; i < nlist; ++i) {
    centroid_norms[static_cast<Eigen::Index>(i)] =
        main_centroids.row(static_cast<Eigen::Index>(i)).squaredNorm();
  }
  const size_t bytes_per_record =
      static_cast<size_t>(nlist) * sizeof(float) +
      static_cast<size_t>(top_r) * sizeof(AssignmentTopCandidate) +
      static_cast<size_t>(main_centroids.cols()) * sizeof(float) + sizeof(float);
  const size_t chunk_records = std::max<size_t>(
      1,
      std::min<size_t>(16384,
                       kMaxAssignmentChunkBytes / std::max<size_t>(1, bytes_per_record)));
  std::vector<float> chunk_distances;
  std::vector<AssignmentTopCandidate> chunk_top_candidates;
  MatrixRM chunk_vectors;
  Eigen::VectorXf chunk_row_norms;
  if (!frozen_delta.records.empty()) {
    chunk_distances.resize(chunk_records * static_cast<size_t>(nlist));
    chunk_top_candidates.resize(chunk_records * static_cast<size_t>(top_r));
    chunk_vectors.resize(static_cast<Eigen::Index>(chunk_records), main_centroids.cols());
    chunk_row_norms.resize(static_cast<Eigen::Index>(chunk_records));
  }
  for (size_t chunk_begin = 0; chunk_begin < frozen_delta.records.size();
       chunk_begin += chunk_records) {
    const size_t chunk_count =
        std::min(chunk_records, frozen_delta.records.size() - chunk_begin);

    const auto distance_start = ProfilingClock::now();
    for (size_t local_index = 0; local_index < chunk_count; ++local_index) {
      const auto& rec = frozen_delta.records[chunk_begin + static_cast<size_t>(local_index)];
      chunk_vectors.row(static_cast<Eigen::Index>(local_index)) = rec.x.transpose();
      chunk_row_norms[static_cast<Eigen::Index>(local_index)] =
          chunk_vectors.row(static_cast<Eigen::Index>(local_index)).squaredNorm();
    }
    Eigen::Map<MatrixRM> chunk_distance_matrix(
        chunk_distances.data(),
        static_cast<Eigen::Index>(chunk_records),
        static_cast<Eigen::Index>(nlist));
    auto active_distances =
        chunk_distance_matrix.topRows(static_cast<Eigen::Index>(chunk_count));
    active_distances.noalias() =
        chunk_vectors.topRows(static_cast<Eigen::Index>(chunk_count)) *
        main_centroids.transpose();
    active_distances *= -2.0f;
    active_distances.colwise() +=
        chunk_row_norms.head(static_cast<Eigen::Index>(chunk_count));
    active_distances.rowwise() += centroid_norms.transpose();
    distance_us += ElapsedProfilingMicros(distance_start);

    const auto top_r_start = ProfilingClock::now();
    #pragma omp parallel
    {
      AssignmentScratch scratch;
      scratch.Reserve(top_r);
      #pragma omp for schedule(static)
      for (int64_t local_index = 0; local_index < static_cast<int64_t>(chunk_count);
           ++local_index) {
        const float* distances = chunk_distances.data() +
                                 static_cast<size_t>(local_index) * static_cast<size_t>(nlist);
        scratch.Clear(top_r);
        for (uint32_t i = 0; i < nlist; ++i) {
          InsertAssignmentTopCandidate(
              &scratch,
              AssignmentTopCandidate{distances[static_cast<size_t>(i)], i},
              top_r);
        }
        std::copy_n(scratch.Data(top_r), top_r,
                    chunk_top_candidates.begin() +
                        static_cast<size_t>(local_index) * static_cast<size_t>(top_r));
      }
    }
    top_r_us += ElapsedProfilingMicros(top_r_start);

    for (size_t local_index = 0; local_index < chunk_count; ++local_index) {
      const AssignmentTopCandidate* top_candidates =
          chunk_top_candidates.data() + local_index * static_cast<size_t>(top_r);

      const uint32_t nearest_partition = top_candidates[0].part;
      const float nearest_dist = top_candidates[0].dist;
      uint32_t chosen_partition = nearest_partition;
      float chosen_dist = nearest_dist;

      const auto balance_start = ProfilingClock::now();
      if (use_balanced_append) {
        const float dist_cap = nearest_dist * static_cast<float>(options.assignment_gamma);
        const double nearest_penalty = std::max(
            0.0,
            (static_cast<double>(projected_size[static_cast<size_t>(nearest_partition)]) + 1.0) -
                avg_after) /
            balance_denom;
        const bool nearest_overloaded = nearest_penalty > 0.0;

        bool found_legal_candidate = false;
        bool found_healthier_candidate = false;
        AssignmentCandidateState best_legal_candidate;
        AssignmentCandidateState best_healthier_candidate;
        for (uint32_t k = 0; k < top_r; ++k) {
          const float dist = top_candidates[static_cast<size_t>(k)].dist;
          const uint32_t part = top_candidates[static_cast<size_t>(k)].part;
          if (dist > dist_cap) {
            continue;
          }
          if (static_cast<double>(projected_size[static_cast<size_t>(part)]) >= hard_cap) {
            continue;
          }
          AssignmentCandidateState cand;
          cand.part = part;
          cand.dist = dist;
          cand.penalty = std::max(
              0.0,
              (static_cast<double>(projected_size[static_cast<size_t>(part)]) + 1.0) - avg_after) /
                         balance_denom;
          cand.cost = static_cast<double>(dist) + options.assignment_lambda * cand.penalty;

          if (!found_legal_candidate || cand.cost < best_legal_candidate.cost) {
            found_legal_candidate = true;
            best_legal_candidate = cand;
          }
          if (nearest_overloaded && cand.penalty + 1e-12 < nearest_penalty) {
            if (!found_healthier_candidate || cand.cost < best_healthier_candidate.cost) {
              found_healthier_candidate = true;
              best_healthier_candidate = cand;
            }
          }
        }

        const AssignmentCandidateState* chosen_candidate = nullptr;
        if (found_healthier_candidate) {
          chosen_candidate = &best_healthier_candidate;
        } else if (found_legal_candidate) {
          chosen_candidate = &best_legal_candidate;
        }
        if (chosen_candidate != nullptr) {
          chosen_partition = chosen_candidate->part;
          chosen_dist = chosen_candidate->dist;
        }
      }
      balance_us += ElapsedProfilingMicros(balance_start);

      const auto materialize_start = ProfilingClock::now();
      DeltaAssignment item;
      item.frozen_index = static_cast<uint32_t>(chunk_begin + local_index);
      item.main_partition = chosen_partition;
      item.residual_dist = chosen_dist;
      assigned[static_cast<size_t>(chosen_partition)].push_back(std::move(item));
      projected_size[static_cast<size_t>(chosen_partition)]++;
      materialize_us += ElapsedProfilingMicros(materialize_start);

      if (chosen_partition != nearest_partition) {
        moved_count++;
      }
      const double ratio =
          nearest_dist > eps ? static_cast<double>(chosen_dist) / nearest_dist : 1.0;
      ratio_sum += ratio;
      ratio_max = std::max(ratio_max, ratio);
    }
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
    profiling->merge_delta_to_main_assignment_us =
        ElapsedProfilingMicros(assignment_start);
    profiling->merge_assignment_distance_us = distance_us;
    profiling->merge_assignment_top_r_us = top_r_us;
    profiling->merge_assignment_balance_us = balance_us;
    profiling->merge_assignment_materialize_us = materialize_us;
    profiling->assignment_descriptor_records =
        static_cast<uint64_t>(frozen_delta.records.size());
    profiling->assignment_full_vector_copy_bytes = 0;
  }
  return assigned;
}

Result<AlignedVector<VectorRecord>> MaterializeVectorRecordsFromAccessor(
    const std::vector<DocId>& doc_ids,
    const VersionSet& versions,
    uint32_t ivf_id,
    const VectorAccessor& vector_accessor,
    VectorAccessorMaterializationStats* profiling) {
  const uint32_t dim = vector_accessor.dim();
  AlignedVector<VectorRecord> records;
  records.reserve(doc_ids.size());
  std::vector<DocId> chunk_doc_ids;
  chunk_doc_ids.reserve(std::min(kVectorAccessorMaterializeChunkRows, doc_ids.size()));
  MatrixRM rows;
  for (size_t begin = 0; begin < doc_ids.size();
       begin += kVectorAccessorMaterializeChunkRows) {
    const size_t end =
        std::min(begin + kVectorAccessorMaterializeChunkRows, doc_ids.size());
    chunk_doc_ids.assign(
        doc_ids.begin() + static_cast<std::ptrdiff_t>(begin),
        doc_ids.begin() + static_cast<std::ptrdiff_t>(end));
    Status materialize = vector_accessor.Materialize(chunk_doc_ids, &rows);
    if (!materialize.ok()) {
      return materialize;
    }
    if (rows.rows() != static_cast<Eigen::Index>(chunk_doc_ids.size()) ||
        rows.cols() != static_cast<Eigen::Index>(dim)) {
      return Status::InvalidArgument(
          "MaterializeVectorRecordsFromAccessor: materialized shape mismatch");
    }
    if (profiling != nullptr) {
      ++profiling->calls;
      profiling->rows += static_cast<uint64_t>(rows.rows());
      profiling->bytes += static_cast<uint64_t>(rows.size()) * sizeof(float);
      profiling->max_rows = std::max<uint32_t>(
          profiling->max_rows, static_cast<uint32_t>(rows.rows()));
    }
    for (size_t i = 0; i < chunk_doc_ids.size(); ++i) {
      VectorRecord rec;
      rec.doc_id = chunk_doc_ids[i];
      rec.dim = dim;
      rec.versions = versions;
      rec.ivf_id = ivf_id;
      rec.x = rows.row(static_cast<Eigen::Index>(i)).transpose();
      records.push_back(std::move(rec));
    }
  }
  return records;
}

Result<AlignedVector<VectorRecord>> FetchPartitionRecordsForMerge(
    const std::shared_ptr<IVFIndex>& ivf,
    const VersionSet& versions,
    uint32_t partition_id,
    const VectorAccessor* vector_accessor,
    VectorAccessorMaterializationStats* profiling) {
  if (!ivf) {
    return Status::InvalidArgument("FetchPartitionRecordsForMerge: ivf is null");
  }
  if (vector_accessor == nullptr) {
    return ivf->GetPartitionRecords(versions, partition_id);
  }

  auto doc_ids_res = ivf->GetPartitionDocIds(versions, partition_id);
  if (!doc_ids_res.ok()) {
    return doc_ids_res.status();
  }
  return MaterializeVectorRecordsFromAccessor(
      doc_ids_res.value(), versions, partition_id, *vector_accessor, profiling);
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
    const MergeOptions& options,
    MergeProfiling* profiling) {
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

  const auto pool_start = profiling != nullptr
                              ? ProfilingClock::now()
                              : ProfilingClock::time_point{};
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
  if (profiling != nullptr) {
    profiling->repartition_pool_us += ElapsedProfilingMicros(pool_start);
    profiling->pooled_records += static_cast<uint64_t>(pooled_records.size());
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
    const auto distance_start = profiling != nullptr
                                    ? ProfilingClock::now()
                                    : ProfilingClock::time_point{};
    dists.clear();
    for (uint32_t local_idx = 0; local_idx < neighborhood_n; ++local_idx) {
      const uint32_t part = neighborhood_partitions[static_cast<size_t>(local_idx)];
      const float dist =
          (rec.x - main_centroids.row(static_cast<Eigen::Index>(part)).transpose()).squaredNorm();
      dists.emplace_back(dist, local_idx);
    }
    if (profiling != nullptr) {
      profiling->repartition_distance_us += ElapsedProfilingMicros(distance_start);
    }
    const auto local_sort_start = profiling != nullptr
                                      ? ProfilingClock::now()
                                      : ProfilingClock::time_point{};
    std::sort(dists.begin(),
              dists.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    if (profiling != nullptr) {
      profiling->repartition_sort_us += ElapsedProfilingMicros(local_sort_start);
    }

    const auto candidate_start = profiling != nullptr
                                     ? ProfilingClock::now()
                                     : ProfilingClock::time_point{};
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
    if (profiling != nullptr) {
      profiling->repartition_candidate_selection_us +=
          ElapsedProfilingMicros(candidate_start);
    }

    rec.ivf_id = neighborhood_partitions[static_cast<size_t>(chosen_local)];
    repartitioned[static_cast<size_t>(chosen_local)].push_back(std::move(rec));
    projected_size[static_cast<size_t>(chosen_local)]++;
  }

  for (auto& bucket : repartitioned) {
    const auto final_sort_start = profiling != nullptr
                                      ? ProfilingClock::now()
                                      : ProfilingClock::time_point{};
    std::sort(bucket.begin(), bucket.end(), [](const VectorRecord& lhs, const VectorRecord& rhs) {
      return lhs.doc_id < rhs.doc_id;
    });
    if (profiling != nullptr) {
      profiling->repartition_sort_us += ElapsedProfilingMicros(final_sort_start);
    }
  }
  if (profiling != nullptr) {
    for (const auto& bucket : repartitioned) {
      profiling->repartitioned_records += static_cast<uint64_t>(bucket.size());
    }
  }
  return repartitioned;
}

}  // namespace

Result<PartitionPatch> PreparePartitionPatchImpl(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    FrozenDelta* consumable_frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor* vector_accessor,
    const MergeOptions& options,
    MergeProfiling* profiling);

Result<MergeReport> MergeFrozenDeltaIntoMainImpl(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 FrozenDelta* consumable_frozen_delta,
                                                 const VectorAccessor* vector_accessor,
                                                 const VectorAccessorMaterializationStats*
                                                     initial_materialization,
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

Result<FrozenDelta> FreezeDeltaFromAccessor(
    const std::shared_ptr<IVFIndex>& delta_ivf,
    const VersionSet& delta_versions,
    const VectorAccessor& vector_accessor,
    VectorAccessorMaterializationStats* profiling) {
  if (!delta_ivf) {
    return Status::InvalidArgument("freeze_delta(accessor): delta_ivf is null");
  }
  auto doc_ids_res = delta_ivf->SnapshotDocIds(delta_versions);
  if (!doc_ids_res.ok()) {
    return doc_ids_res.status();
  }
  FrozenDelta out;
  auto records_res = MaterializeVectorRecordsFromAccessor(
      doc_ids_res.value(), delta_versions, 0, vector_accessor, profiling);
  if (!records_res.ok()) {
    return records_res.status();
  }
  out.records = std::move(records_res.value());
  return out;
}

Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions,
                                 const MatrixRM& base_vectors) {
  MatrixVectorAccessor vector_accessor(base_vectors);
  return FreezeDeltaFromAccessor(delta_ivf, delta_versions, vector_accessor, nullptr);
}

Result<FrozenDelta> freeze_delta(const std::shared_ptr<IVFIndex>& delta_ivf,
                                 const VersionSet& delta_versions,
                                 const VectorAccessor& vector_accessor) {
  return FreezeDeltaFromAccessor(delta_ivf, delta_versions, vector_accessor, nullptr);
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
  MatrixVectorAccessor vector_accessor(base_vectors);
  return prepare_partition_patch(main_ivf,
                                 main_versions,
                                 frozen_delta,
                                 assignments,
                                 score_result,
                                 vector_accessor,
                                 options);
}

Result<PartitionPatch> prepare_partition_patch(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const PartitionAssignments& assignments,
    const PartitionScoreResult& score_result,
    const VectorAccessor& vector_accessor,
    const MergeOptions& options) {
  return PreparePartitionPatchImpl(
      main_ivf,
      main_versions,
      frozen_delta,
      nullptr,
      assignments,
      score_result,
      &vector_accessor,
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
    const VectorAccessor* vector_accessor,
    const MergeOptions& options,
    MergeProfiling* profiling) {
  if (!main_ivf) {
    return Status::InvalidArgument("prepare_partition_patch: main_ivf is null");
  }
  std::mutex profiling_mu;

  if (options.assignment_mode == "balanced_append") {
    auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
    if (!centroids_res.ok()) {
      return centroids_res.status();
    }
    if (assignments.size() != static_cast<size_t>(centroids_res.value().rows())) {
      return Status::InvalidArgument(
          "prepare_partition_patch: assignments size mismatch with centroids");
    }

    const auto plan_start = profiling != nullptr
                                ? ProfilingClock::now()
                                : ProfilingClock::time_point{};
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
      profiling->seed_partitions = static_cast<uint64_t>(seeds.size());
    }

    struct MergeNeighborhoodPlan {
      uint32_t seed_partition{0};
      std::vector<uint32_t> partition_ids;
    };
    struct MergeNeighborhoodResult {
      std::vector<uint32_t> partition_ids;
      std::vector<AlignedVector<VectorRecord>> replacement_records;
      std::optional<PreparedPQPayload> prepared_pq;
      MergeProfiling profiling;
    };

    std::vector<MergeNeighborhoodPlan> plans;
    std::vector<uint8_t> claimed(assignments.size(), 0u);
    for (uint32_t seed : seeds) {
      if (claimed[static_cast<size_t>(seed)] != 0u) {
        continue;
      }
      const auto neighbor_start = profiling != nullptr
                                      ? ProfilingClock::now()
                                      : ProfilingClock::time_point{};
      std::vector<uint32_t> neighborhood =
          TopRNeighborPartitions(seed, centroids_res.value(), options.assignment_top_r);
      if (profiling != nullptr) {
        profiling->top_r_neighbor_us += ElapsedProfilingMicros(neighbor_start);
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
      for (uint32_t part : active_neighborhood) {
        claimed[static_cast<size_t>(part)] = 1u;
      }
      plans.push_back(MergeNeighborhoodPlan{seed, std::move(active_neighborhood)});
    }
    if (profiling != nullptr) {
      profiling->neighborhoods += static_cast<uint64_t>(plans.size());
      profiling->neighborhood_plan_us += ElapsedProfilingMicros(plan_start);
    }

    std::vector<MergeNeighborhoodResult> results(plans.size());
    auto execute_neighborhood = [&](size_t plan_index) -> Status {
      const MergeNeighborhoodPlan& plan = plans[plan_index];
      MergeNeighborhoodResult& result = results[plan_index];
      result.partition_ids = plan.partition_ids;
      MergeProfiling* local_profiling = profiling != nullptr ? &result.profiling : nullptr;
      std::vector<AlignedVector<VectorRecord>> neighborhood_main_records;
      neighborhood_main_records.reserve(plan.partition_ids.size());
      for (uint32_t part : plan.partition_ids) {
        VectorAccessorMaterializationStats materialization_profile;
        const auto fetch_start = local_profiling != nullptr
                                     ? ProfilingClock::now()
                                     : ProfilingClock::time_point{};
        auto main_records_res =
            FetchPartitionRecordsForMerge(main_ivf,
                                          main_versions,
                                          part,
                                          vector_accessor,
                                          &materialization_profile);
        if (local_profiling != nullptr) {
          local_profiling->fetch_main_records_us += ElapsedProfilingMicros(fetch_start);
          AccumulateVectorAccessorMaterialization(materialization_profile, local_profiling);
        }
        if (!main_records_res.ok()) {
          return main_records_res.status();
        }
        if (local_profiling != nullptr) {
          local_profiling->main_records_loaded +=
              static_cast<uint64_t>(main_records_res.value().size());
        }
        neighborhood_main_records.push_back(std::move(main_records_res.value()));
      }

      auto repartition_res = RepartitionNeighborhood(plan.partition_ids,
                                                     std::move(neighborhood_main_records),
                                                     frozen_delta,
                                                     consumable_frozen_delta,
                                                     assignments,
                                                     centroids_res.value(),
                                                     options,
                                                     local_profiling);
      if (!repartition_res.ok()) {
        return repartition_res.status();
      }
      std::vector<AlignedVector<VectorRecord>> repartitioned = std::move(repartition_res.value());
      auto finalization_res = FinalizePartitionPQCodes(
          main_ivf, main_versions, plan.partition_ids, &repartitioned);
      if (!finalization_res.ok()) {
        return finalization_res.status();
      }
      AccumulatePQFinalizationProfiling(finalization_res.value(), local_profiling);
      result.prepared_pq = std::move(finalization_res.value().prepared_pq);
      result.replacement_records = std::move(repartitioned);
      return Status::OK();
    };

    const auto execute_start = profiling != nullptr
                                   ? ProfilingClock::now()
                                   : ProfilingClock::time_point{};
    if (consumable_frozen_delta == nullptr) {
      std::atomic<bool> failed{false};
      std::mutex error_mutex;
      Status first_error = Status::OK();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
      for (int64_t i = 0; i < static_cast<int64_t>(plans.size()); ++i) {
        if (failed.load(std::memory_order_relaxed)) {
          continue;
        }
        Status status = execute_neighborhood(static_cast<size_t>(i));
        if (!status.ok()) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!failed.load(std::memory_order_relaxed)) {
            first_error = std::move(status);
            failed.store(true, std::memory_order_relaxed);
          }
        }
      }
      if (failed.load(std::memory_order_relaxed)) {
        return first_error;
      }
    } else {
      for (size_t i = 0; i < plans.size(); ++i) {
        Status status = execute_neighborhood(i);
        if (!status.ok()) {
          return status;
        }
      }
    }
    if (profiling != nullptr) {
      profiling->neighborhood_execute_wall_us += ElapsedProfilingMicros(execute_start);
    }

    PartitionPatch patch;
    const auto combine_start = profiling != nullptr
                                   ? ProfilingClock::now()
                                   : ProfilingClock::time_point{};
    for (MergeNeighborhoodResult& result : results) {
      if (profiling != nullptr) {
        AccumulateNeighborhoodProfiling(result.profiling, profiling);
      }
      Status append_prepared = AppendPreparedPQPayload(
          std::move(result.prepared_pq), assignments.size(), &patch);
      if (!append_prepared.ok()) {
        return append_prepared;
      }
      for (size_t i = 0; i < result.partition_ids.size(); ++i) {
        patch.partition_ids.push_back(result.partition_ids[i]);
        patch.replacement_records.push_back(std::move(result.replacement_records[i]));
      }
    }
    if (profiling != nullptr) {
      profiling->neighborhood_combine_us += ElapsedProfilingMicros(combine_start);
    }
    if (profiling != nullptr) {
      profiling->patch_records = 0;
      for (const auto& records : patch.replacement_records) {
        profiling->patch_records += static_cast<uint64_t>(records.size());
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
  std::vector<std::optional<PreparedPQPayload>> prepared_batches(active_partitions.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(active_partitions.size()); ++i) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    const uint32_t p = active_partitions[static_cast<size_t>(i)];
    const auto& delta_bucket = assignments[static_cast<size_t>(p)];
    VectorAccessorMaterializationStats materialization_profile;
    const auto fetch_start = profiling != nullptr
                                 ? ProfilingClock::now()
                                 : ProfilingClock::time_point{};
    auto main_records_res =
        FetchPartitionRecordsForMerge(main_ivf,
                                      main_versions,
                                      p,
                                      vector_accessor,
                                      &materialization_profile);
    if (profiling != nullptr) {
      std::lock_guard<std::mutex> lock(profiling_mu);
      profiling->fetch_main_records_us += ElapsedProfilingMicros(fetch_start);
      AccumulateVectorAccessorMaterialization(materialization_profile, profiling);
    }
    if (!main_records_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = main_records_res.status();
      }
      continue;
    }
    if (profiling != nullptr) {
      std::lock_guard<std::mutex> lock(profiling_mu);
      profiling->main_records_loaded +=
          static_cast<uint64_t>(main_records_res.value().size());
    }
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
    std::vector<uint32_t> batch_partition_ids{p};
    std::vector<AlignedVector<VectorRecord>> batch_records;
    batch_records.push_back(std::move(merged_res.value()));
    auto finalization_res = FinalizePartitionPQCodes(
        main_ivf, main_versions, batch_partition_ids, &batch_records);
    if (!finalization_res.ok()) {
      std::lock_guard<std::mutex> lock(err_mu);
      if (!failed.exchange(true)) {
        first_error = finalization_res.status();
      }
      continue;
    }
    if (profiling != nullptr) {
      std::lock_guard<std::mutex> lock(profiling_mu);
      AccumulatePQFinalizationProfiling(finalization_res.value(), profiling);
    }
    prepared_batches[static_cast<size_t>(i)] =
        std::move(finalization_res.value().prepared_pq);
    patch.replacement_records[static_cast<size_t>(i)] =
        std::move(batch_records.front());
  }
  if (failed.load(std::memory_order_relaxed)) {
    return first_error;
  }
  for (auto& prepared_batch : prepared_batches) {
    Status append_prepared = AppendPreparedPQPayload(
        std::move(prepared_batch), active_partitions.size(), &patch);
    if (!append_prepared.ok()) {
      return append_prepared;
    }
  }
  if (profiling != nullptr) {
    profiling->patch_records = 0;
    for (const auto& records : patch.replacement_records) {
      profiling->patch_records += static_cast<uint64_t>(records.size());
    }
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
      main_ivf, main_versions, frozen_delta, nullptr, nullptr, nullptr, options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 const MatrixRM& base_vectors,
                                                 const MergeOptions& options) {
  MatrixVectorAccessor vector_accessor(base_vectors);
  return merge_frozen_delta_into_main(
      main_ivf, main_versions, frozen_delta, vector_accessor, options);
}

Result<MergeReport> merge_frozen_delta_into_main(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const FrozenDelta& frozen_delta,
    const VectorAccessor& vector_accessor,
    const MergeOptions& options) {
  return MergeFrozenDeltaIntoMainImpl(
      main_ivf,
      main_versions,
      frozen_delta,
      nullptr,
      &vector_accessor,
      nullptr,
      options);
}

Result<MergeReport> MergeFrozenDeltaIntoMainImpl(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const FrozenDelta& frozen_delta,
                                                 FrozenDelta* consumable_frozen_delta,
                                                 const VectorAccessor* vector_accessor,
                                                 const VectorAccessorMaterializationStats*
                                                     initial_materialization,
                                                 const MergeOptions& options) {
  if (!main_ivf) {
    return Status::InvalidArgument("merge_frozen_delta_into_main: main_ivf is null");
  }
  MergeProfiling profiling;
  profiling.frozen_records = static_cast<uint64_t>(frozen_delta.records.size());
  if (initial_materialization != nullptr) {
    AccumulateVectorAccessorMaterialization(*initial_materialization, &profiling);
  }
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
  // Assignment, diagnostics, stats, and scoring are complete. Prepare is the
  // first phase allowed to consume a locally owned Frozen payload.
  auto patch_res = PreparePartitionPatchImpl(
      main_ivf,
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
  if (consumable_frozen_delta != nullptr) {
    profiling.frozen_payload_records_moved =
        static_cast<uint64_t>(frozen_delta.records.size());
  }
  profiling.patch_prepare_us = patch_prepare_timer.ElapsedMicros();
  const double merge_compute_ms = merge_compute_timer.ElapsedMillis();
  profiling.compute_done_at = ProfilingClock::now();
  Timer commit_timer;
  Status commit = commit_partition_patch(main_ivf, main_versions, patch_res.value());
  if (!commit.ok()) {
    return commit;
  }
  profiling.commit_us = commit_timer.ElapsedMicros();
  profiling.patch_records = 0;
  for (const auto& records : patch_res.value().replacement_records) {
    profiling.patch_records += static_cast<uint64_t>(records.size());
  }
  auto codebook_ms_res = main_ivf->GetLastPatchPQReencodeMs(main_versions);
  if (!codebook_ms_res.ok()) {
    return codebook_ms_res.status();
  }
  auto patch_profile_res = main_ivf->GetLastPatchProfiling(main_versions);
  if (!patch_profile_res.ok()) {
    return patch_profile_res.status();
  }
  const PatchProfiling& patch_profile = patch_profile_res.value();
  profiling.prepare_publish_us = patch_profile.prepare_publish_us;
  profiling.commit_validation_us = patch_profile.commit_validation_us;
  profiling.commit_docmap_us = patch_profile.commit_docmap_us;
  profiling.commit_partition_swap_us = patch_profile.commit_partition_swap_us;
  profiling.commit_wait_start_at = patch_profile.commit_wait_start_at;
  profiling.commit_lock_acquired_at = patch_profile.commit_lock_acquired_at;
  profiling.commit_done_at = patch_profile.commit_done_at;
  profiling.pq_codes_reused = patch_profile.pq_codes_reused;
  profiling.pq_codes_reencoded = patch_profile.pq_codes_reencoded;
  profiling.pq_code_assignment_us = patch_profile.pq_code_assignment_us;
  profiling.pq_code_copy_or_reuse_us = patch_profile.pq_code_copy_or_reuse_us;
  profiling.pq_list_flatten_us = patch_profile.pq_list_flatten_us;

  MergeReport report;
  report.frozen_records = static_cast<uint32_t>(frozen_delta.records.size());
  report.patch_partitions = static_cast<uint32_t>(patch_res.value().partition_ids.size());
  report.append_partitions = static_cast<uint32_t>(score_res.value().append_partitions.size());
  report.recluster_partitions = static_cast<uint32_t>(score_res.value().recluster_partitions.size());
  report.merge_compute_ms = merge_compute_ms;
  report.codebook_rebuild_ms = codebook_ms_res.value();
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
                                      nullptr,
                                      options);
}

Result<MergeReport> merge_frozen_delta_into_main(const std::shared_ptr<IVFIndex>& main_ivf,
                                                 const VersionSet& main_versions,
                                                 const std::shared_ptr<IVFIndex>& delta_ivf,
                                                 const VersionSet& delta_versions,
                                                 const MatrixRM& base_vectors,
                                                 const MergeOptions& options) {
  MatrixVectorAccessor vector_accessor(base_vectors);
  return merge_frozen_delta_into_main(main_ivf,
                                      main_versions,
                                      delta_ivf,
                                      delta_versions,
                                      vector_accessor,
                                      options);
}

Result<MergeReport> merge_frozen_delta_into_main(
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    const std::shared_ptr<IVFIndex>& delta_ivf,
    const VersionSet& delta_versions,
    const VectorAccessor& vector_accessor,
    const MergeOptions& options) {
  VectorAccessorMaterializationStats materialization_profile;
  auto frozen_res = FreezeDeltaFromAccessor(
      delta_ivf, delta_versions, vector_accessor, &materialization_profile);
  if (!frozen_res.ok()) {
    return frozen_res.status();
  }
  return MergeFrozenDeltaIntoMainImpl(main_ivf,
                                      main_versions,
                                      frozen_res.value(),
                                      &frozen_res.value(),
                                      &vector_accessor,
                                      &materialization_profile,
                                      options);
}

}  // namespace ann
