#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/config.h"
#include "common/status.h"
#include "eval/eval_counter.h"
#include "eval/eval_type.h"
#include "search/hybrid_search.h"

namespace ann::eval_memory {
class MemoryTraceRecorder;
struct MemoryComponent;
}

namespace ann::eval {

std::filesystem::path BuildDefaultResultPath(const std::string& config_path,
                                             const std::string& base_path);
struct RunEvalOutputPaths {
  std::filesystem::path results_dir;
  std::filesystem::path result_path;
  std::filesystem::path memory_trace_path;
};
RunEvalOutputPaths BuildRunEvalOutputPaths(
    const std::string& config_path,
    const std::optional<std::string>& base_dataset_path,
    const std::string& dataset_label);
Status WriteQueryRecallCurve(
    const std::filesystem::path& result_path,
    const std::vector<std::pair<uint32_t, run_eval::EvalMetrics>>& curve);

struct RunEvalResultData {
  const std::string& dataset_label;
  int64_t timestamp;
  const Config& config;
  const SearchParams& params;
  const run_eval::EvalMetrics& final_metrics;
  const std::vector<run_eval::SnapshotRecord>& snapshots;
  const std::vector<run_eval::MergeEventRecord>& merge_events;
  const std::vector<run_eval::GlobalRebuildEventRecord>& global_rebuild_events;
  const RunEvalCounterValues& counters;
  const run_eval::SeriesStats& recall_summary;
  const run_eval::SeriesStats& qps_summary;
  const run_eval::SeriesStats& latency_summary;
  const run_eval::SeriesStats& e2e_latency_summary;
  const run_eval::SeriesStats& update_throughput_summary;
  const run_eval::SeriesStats& amortized_update_throughput_summary;
  double total_maintenance_ms;
  uint64_t summary_gt_new_total;
  uint64_t summary_gt_old_total;
  uint64_t summary_hit_new_total;
  uint64_t summary_hit_old_total;
  double summary_recall_new;
  double summary_recall_old;
  double summary_gt_new_ratio;
  bool used_prebuilt_index;
  double init_load_ms;
  double init_rebuild_ms;
  double init_main_build_ms;
  double init_main_add_ms;
  double init_total_wall_ms;
  double init_whitening_ms;
  double init_whitening_transform_ms;
  const IngestProfiling& initial_main_ingest_profile;
  const IngestProfiling& initial_delta_seed_ingest_profile;
  const std::optional<IVFBuildProfiling>& initial_build_profile;
  uint32_t main_rows_initial;
  uint32_t main_rows_current;
  uint32_t rows_after_main;
  uint32_t delta_train_rows;
  uint32_t delta_ivf_nlist;
  uint32_t merge_trigger_rows;
  uint32_t stream_start_idx;
  uint32_t total_stream_rows;
  uint32_t snapshot_span;
  uint32_t sliding_window_rows;
  uint32_t delta_kmeans_iterations;
  const std::optional<run_eval::EvalMetrics>& pre_stream_metrics;
  bool online_pq_enabled;
  double online_avg_nqe;
  double online_avg_qe_ratio;
  double online_avg_drift;
  bool debug_output_enabled;
  bool ground_truth_available;
  double full_run_wall_ms;
};

Status WriteRunEvalResultJson(const std::filesystem::path& path,
                              const RunEvalResultData& data);

void PrintEval(uint64_t eval_id,
               const std::string& stage,
               uint32_t active_rows,
               uint32_t topk,
               const run_eval::EvalMetrics& metrics);

void PrintSnapshot(const run_eval::SnapshotRecord& snapshot,
                   uint32_t topk,
                   bool ground_truth_available);

void PrintMsEval(uint32_t search_id,
                 uint32_t active_rows,
                 uint32_t topk,
                 const run_eval_ms::SnapshotLite& snapshot);

void PrintMsSnapshot(uint32_t topk,
                     uint32_t active_rows,
                     uint32_t snapshot_rows,
                     double throughput,
                     double maintenance_ms,
                     double amortized_throughput,
                     const run_eval_ms::SnapshotLite& snapshot);

void PrintRunEvalSummary(const Config& config,
                         const run_eval::EvalMetrics& metrics,
                         const OnlinePQRollup& online_pq_rollup,
                         uint32_t nprobe,
                         double rebuild_ms_total,
                         double total_update_ms,
                         double total_merge_compute_ms,
                         double total_global_rebuild_ms,
                         double total_delta_ingest_assignment_us,
                         uint64_t total_delta_ingest_assignment_records,
                         uint32_t global_rebuild_count,
                         double online_avg_nqe,
                         double online_avg_qe_ratio,
                         double online_avg_drift,
                         bool ground_truth_available);

void PrintMsDiagnostics(const Config& config,
                        const run_eval_ms::SnapshotLite& snapshot);

void PrintMergeProfile(const MergeProfiling& profile);
void PrintGlobalRebuildBuildProfile(const IVFBuildProfiling& profile);
void PrintGlobalRebuildAddProfile(const IngestProfiling& profile);
void PrintGlobalRebuildProfileSummary(double setup_ms,
                                     double whitening_ms,
                                     double base_retarget_ms,
                                     double query_transform_ms,
                                     double main_build_ms,
                                     double main_add_ms,
                                     double delta_build_ms,
                                     double delta_add_ms,
                                     double delta_finalize_ms,
                                     double delta_seed_ms,
                                     double norms_recompute_ms,
                                     double publication_ms,
                                     double malloc_trim_ms,
                                     double memory_trace_ms,
                                     double profiling_collect_ms,
                                     double accounted_stage_sum_ms,
                                     double wall_total_ms,
                                     double unaccounted_ms);

Status WriteMemoryTraceJson(
    eval_memory::MemoryTraceRecorder& recorder,
    const std::string& path,
    const std::vector<eval_memory::MemoryComponent>& metadata);

void WriteSummaryJson(const std::string& path,
                     const Config& config,
                     const run_eval_ms::Workload& workload,
                     const std::vector<run_eval_ms::SnapshotLite>& snapshots,
                     double init_ms,
                     double update_ms,
                     double update_whitening_ms,
                     double update_insert_ms,
                     double update_record_build_ms,
                     double update_insert_encode_ms,
                     double update_insert_commit_ms,
                     double update_onlinepq_maintenance_ms,
                     double update_delete_ms,
                     double update_codebook_update_ms,
                     double update_reencode_ms,
                     double delta_ingest_assignment_us,
                     uint64_t delta_ingest_assignment_records,
                     double merge_ms,
                     double merge_compute_ms,
                     double global_rebuild_ms,
                     uint32_t merge_count,
                     uint32_t global_rebuild_count,
                     const std::vector<uint32_t>& merge_nodes,
                     const std::vector<uint32_t>& global_rebuild_nodes,
                     uint32_t final_seen_rows,
                     double full_run_wall_ms,
                     const run_eval_ms::SummaryOutput& summary);

}  // namespace ann::eval
