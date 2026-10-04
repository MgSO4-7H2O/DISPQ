#include "eval/eval_output.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "eval_memory.h"

namespace ann::eval {

std::filesystem::path BuildDefaultResultPath(const std::string& config_path,
                                             const std::string& base_path) {
  namespace fs = std::filesystem;
  const std::string config_name = fs::path(config_path).stem().string();
  std::string dataset_name = fs::path(base_path).parent_path().filename().string();
  if (dataset_name.empty()) dataset_name = fs::path(base_path).stem().string();
  std::string metric_name = "default";
  fs::path p(base_path);
  std::vector<std::string> parts;
  for (const auto& part : p) parts.push_back(part.string());
  for (size_t i = 0; i < parts.size(); ++i) {
    if (parts[i] == "data" && i + 2 < parts.size()) {
      dataset_name = parts[i + 1];
      std::string metric;
      for (size_t j = i + 2; j + 1 < parts.size(); ++j) {
        if (!metric.empty()) metric += "_";
        metric += parts[j];
      }
      if (!metric.empty()) metric_name = metric;
      break;
    }
  }
  fs::path dir = fs::path("result") / dataset_name / metric_name / config_name;
  std::error_code ec;
  fs::create_directories(dir, ec);
  return dir / "online_eval.json";
}

RunEvalOutputPaths BuildRunEvalOutputPaths(
    const std::string& config_path,
    const std::optional<std::string>& base_dataset_path,
    const std::string& dataset_label) {
  std::string config_name = std::filesystem::path(config_path).stem().string();
  std::string dataset_name = dataset_label;
  std::string metric_name = "default";
  if (base_dataset_path) {
    std::filesystem::path path(*base_dataset_path);
    std::vector<std::string> parts;
    for (const auto& part : path) parts.push_back(part.string());
    for (size_t i = 0; i < parts.size(); ++i) {
      if (parts[i] == "data" && i + 2 < parts.size()) {
        dataset_name = parts[i + 1];
        std::string metric;
        for (size_t j = i + 2; j + 1 < parts.size(); ++j) {
          if (!metric.empty()) metric += "_";
          metric += parts[j];
        }
        if (!metric.empty()) metric_name = metric;
        break;
      }
    }
  }
  RunEvalOutputPaths paths;
  paths.results_dir = std::filesystem::path("result") / dataset_name / metric_name / config_name;
  std::error_code ec;
  std::filesystem::create_directories(paths.results_dir, ec);
  paths.result_path = paths.results_dir / "online_eval.json";
  paths.memory_trace_path = paths.results_dir / "memory_trace.json";
  return paths;
}

Status WriteQueryRecallCurve(
    const std::filesystem::path& result_path,
    const std::vector<std::pair<uint32_t, run_eval::EvalMetrics>>& curve) {
  if (curve.empty()) return Status::OK();
  const std::filesystem::path curve_path =
      result_path.parent_path() / "query_recall_curve.json";
  std::ofstream curve_ofs(curve_path);
  if (!curve_ofs) return Status::IOError("Failed to write " + curve_path.string());
  curve_ofs << "{\n  \"nprobe_sweep\": [";
  for (size_t i = 0; i < curve.size(); ++i) {
    curve_ofs << curve[i].first << (i + 1 < curve.size() ? ", " : "");
  }
  curve_ofs << "],\n  \"points\": [\n";
  for (size_t i = 0; i < curve.size(); ++i) {
    const auto& point = curve[i];
    curve_ofs << "    {\"nprobe\": " << point.first
              << ", \"recall\": " << point.second.recall
              << ", \"avg_query_ms\": " << point.second.avg_query_ms
              << ", \"p50_query_ms\": " << point.second.search_p50
              << ", \"p99_query_ms\": " << point.second.search_p99
              << ", \"qps\": " << point.second.query_qps
              << ", \"scanned_avg\": " << point.second.scanned_avg << "}"
              << (i + 1 < curve.size() ? "," : "") << "\n";
  }
  curve_ofs << "  ]\n}\n";
  return Status::OK();
}

void PrintEval(uint64_t eval_id,
               const std::string& stage,
               uint32_t active_rows,
               uint32_t topk,
               const run_eval::EvalMetrics& metrics) {
  std::cout << "[EVAL] #" << eval_id << " stage=" << stage << ", base_rows=" << active_rows
            << ", recall@" << topk << "="
            << (metrics.recall_available ? std::to_string(metrics.recall) : "unavailable")
            << ", latency_ms=" << metrics.avg_query_ms << ", qps=" << metrics.query_qps
            << std::endl;
}

void PrintSnapshot(const run_eval::SnapshotRecord& snapshot,
                   uint32_t topk,
                   bool ground_truth_available) {
  std::cout << "[SNAPSHOT] base_rows=" << snapshot.base_rows
            << ", snapshot_rows=" << snapshot.snapshot_rows
            << ", recall@" << topk << "="
            << (ground_truth_available ? std::to_string(snapshot.recall) : "unavailable")
            << ", latency_ms=" << snapshot.latency_ms
            << ", qps=" << snapshot.query_qps
            << ", throughput=" << snapshot.update_throughput_vecps
            << ", maintenance_ms=" << snapshot.snapshot_total_ms
            << ", amortized_update_throughput="
            << snapshot.amortized_update_throughput_vecps << std::endl;
}

void PrintMsEval(uint32_t search_id,
                 uint32_t active_rows,
                 uint32_t topk,
                 const run_eval_ms::SnapshotLite& snapshot) {
  std::cout << "[EVAL] #" << search_id
            << " stage=ms_workload"
            << ", base_rows=" << active_rows
            << ", recall@" << topk << "=" << snapshot.recall
            << ", latency_ms=" << snapshot.avg_query_ms
            << ", end_to_end_overhead_ms=" << snapshot.end_to_end_overhead_ms
            << ", search_p50_ms=" << snapshot.search_p50
            << ", search_p99_ms=" << snapshot.search_p99
            << ", total_p50_ms=" << snapshot.total_p50
            << ", total_p99_ms=" << snapshot.total_p99
            << ", qps=" << snapshot.qps
            << ", scanned_avg=" << snapshot.avg_scanned
            << ", scanned_p50=" << snapshot.scanned_p50
            << ", scanned_p99=" << snapshot.scanned_p99
            << ", scanned_max=" << snapshot.scanned_max
            << ", avg_pq_lut_build_us=" << snapshot.avg_pq_lut_build_us
            << ", avg_pq_adc_scan_us=" << snapshot.avg_pq_adc_scan_us
            << std::endl;
}

void PrintMsSnapshot(uint32_t topk,
                     uint32_t active_rows,
                     uint32_t snapshot_rows,
                     double throughput,
                     double maintenance_ms,
                     double amortized_throughput,
                     const run_eval_ms::SnapshotLite& snapshot) {
  std::cout << "[SNAPSHOT] base_rows=" << active_rows
            << ", snapshot_rows=" << snapshot_rows
            << ", recall@" << topk << "=" << snapshot.recall
            << ", latency_ms=" << snapshot.avg_query_ms
            << ", qps=" << snapshot.qps
            << ", throughput=" << throughput
            << ", maintenance_ms=" << maintenance_ms
            << ", amortized_update_throughput=" << amortized_throughput
            << std::endl;
}

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
                        bool ground_truth_available) {
  std::cout << "[ONLINE EVAL] ";
  if (metrics.recall_available) {
    std::cout << "Recall@" << config.topk << " = " << metrics.recall;
  } else {
    std::cout << "Recall disabled (ground truth unavailable)";
  }
  std::cout << " (nprobe=" << nprobe
            << ", exact_rerank=" << std::boolalpha << config.exact_rerank_enable
            << ", rerank_candidates_per_route=" << config.exact_rerank_candidates_per_route
            << ")" << std::endl;
  std::cout << "Latency(no_merge_wall)=" << metrics.avg_query_ms << "ms; "
            << "End-to-end overhead=" << metrics.end_to_end_overhead_ms << "ms; "
            << "Search p50=" << metrics.search_p50 << "ms, p99=" << metrics.search_p99
            << "ms; "
            << "Total p50=" << metrics.total_p50 << "ms, p99=" << metrics.total_p99
            << "ms; "
            << "Build/Rebuild=" << rebuild_ms_total << "ms; "
            << "Update maintenance total=" << metrics.update_total_ms
            << "ms (apply=" << total_update_ms
            << ", merge_compute=" << total_merge_compute_ms
            << ", global_rebuild=" << total_global_rebuild_ms
            << "), per_vec=" << metrics.update_per_vector_ms << "ms; "
            << "Scanned avg=" << metrics.scanned_avg << ", p50=" << metrics.scanned_p50
            << ", p99=" << metrics.scanned_p99 << ", max=" << metrics.scanned_max
            << "; Query QPS=" << metrics.query_qps
            << ", Update throughput=" << metrics.update_throughput_vecps << " vec/s"
            << std::endl;
  std::cout << "[STREAM_PROFILE_TOTAL] delta_ingest_assignment_us="
            << total_delta_ingest_assignment_us
            << ", records=" << total_delta_ingest_assignment_records << std::endl;
  std::cout << "[GLOBAL REBUILD] enabled=" << std::boolalpha << config.enable_global_rebuild
            << ", count=" << global_rebuild_count << "/" << config.global_rebuild_max_count
            << ", trigger_main_imbalance=" << config.global_rebuild_main_imbalance_ratio
            << ", trigger_main_rows=" << config.global_rebuild_force_main_rows
            << ", cooldown_rows=" << config.global_rebuild_cooldown_rows << std::endl;
  std::cout << "[ONLINE PQ] enabled=" << std::boolalpha
            << (config.online_pq_enable && config.pq_enable && config.pq_residual)
            << ", mode=" << config.online_pq_update_scheme
            << ", batches=" << online_pq_rollup.batches
            << ", warmup_batches=" << online_pq_rollup.warmup_batches
            << ", triggered=" << online_pq_rollup.triggered
            << ", updated=" << online_pq_rollup.updated
            << ", reencoded=" << online_pq_rollup.reencoded
            << ", avg_nqe=" << online_avg_nqe
            << ", warmup_left=" << online_pq_rollup.last_warmup_batches_left
            << ", last_qe_ratio=" << online_pq_rollup.last_qe_ratio
            << ", avg_qe_ratio=" << online_avg_qe_ratio
            << ", avg_codebook_drift=" << online_avg_drift << std::endl;
  std::cout << "[QUERY ROUTING] main_queries=" << metrics.main_route_queries
            << ", frozen_delta_queries=" << metrics.frozen_delta_route_queries
            << ", active_delta_queries=" << metrics.active_delta_route_queries << std::endl;
  if (ground_truth_available && config.enable_miss_diag) {
    std::cout << "[MISS DIAG] gt_probed_rate=" << metrics.gt_probed_rate
              << ", recall_on_probed_gt=" << metrics.recall_on_probed_gt
              << ", exact_recall_on_probed_candidates="
              << metrics.exact_recall_on_probed_candidates
              << ", miss_not_probed=" << metrics.miss_not_probed
              << ", miss_probed_filtered_by_pq=" << metrics.miss_probed_filtered_by_pq
              << ", avg_pq_rank_loss=" << metrics.avg_pq_rank_loss << std::endl;
  } else if (!ground_truth_available) {
    std::cout << "[MISS DIAG] unavailable (ground truth disabled)" << std::endl;
  } else {
    std::cout << "[MISS DIAG] disabled by config(enable_miss_diag=false)" << std::endl;
  }
  if (config.enable_rerank_source_diag) {
    std::cout << "[RERANK SOURCE DIAG] main_total=" << metrics.rerank_topk_main_total
              << ", delta_total=" << metrics.rerank_topk_delta_total
              << ", main_ratio=" << metrics.rerank_topk_main_ratio
              << ", delta_ratio=" << metrics.rerank_topk_delta_ratio
              << ", main_avg_topk=" << metrics.rerank_topk_main_avg
              << ", delta_avg_topk=" << metrics.rerank_topk_delta_avg << std::endl;
  } else {
    std::cout << "[RERANK SOURCE DIAG] disabled by config(enable_rerank_source_diag=false)"
              << std::endl;
  }
}

void PrintMsDiagnostics(const Config& config,
                        const run_eval_ms::SnapshotLite& snapshot) {
  if (config.enable_miss_diag) {
    std::cout << "[MISS DIAG] gt_probed_rate=" << snapshot.gt_probed_rate
              << ", recall_on_probed_gt=" << snapshot.recall_on_probed_gt
              << ", exact_recall_on_probed_candidates="
              << snapshot.exact_recall_on_probed_candidates
              << ", miss_not_probed=" << snapshot.miss_not_probed
              << ", miss_probed_filtered_by_pq=" << snapshot.miss_probed_filtered_by_pq
              << ", avg_pq_rank_loss=" << snapshot.avg_pq_rank_loss << std::endl;
  }
  if (config.enable_rerank_source_diag) {
    std::cout << "[RERANK SOURCE DIAG] main_total=" << snapshot.rerank_topk_main_total
              << ", delta_total=" << snapshot.rerank_topk_delta_total
              << ", main_ratio=" << snapshot.rerank_topk_main_ratio
              << ", delta_ratio=" << snapshot.rerank_topk_delta_ratio
              << ", main_avg_topk=" << snapshot.rerank_topk_main_avg
              << ", delta_avg_topk=" << snapshot.rerank_topk_delta_avg << std::endl;
  }
}

void PrintMergeProfile(const MergeProfiling& profile) {
  std::cout << "[MERGE_PROFILE] effective_nlist=" << profile.effective_nlist
            << ", frozen_records=" << profile.frozen_records
            << ", assignment_descriptor_records="
            << profile.assignment_descriptor_records
            << ", assignment_full_vector_copy_bytes="
            << profile.assignment_full_vector_copy_bytes
            << ", frozen_payload_records_moved="
            << profile.frozen_payload_records_moved
            << ", vector_accessor_materialize_calls="
            << profile.vector_accessor_materialize_calls
            << ", vector_accessor_materialized_rows="
            << profile.vector_accessor_materialized_rows
            << ", vector_accessor_materialized_bytes="
            << profile.vector_accessor_materialized_bytes
            << ", vector_accessor_max_materialize_rows="
            << profile.vector_accessor_max_materialize_rows
            << ", seed_partitions=" << profile.seed_partitions
            << ", neighborhoods=" << profile.neighborhoods
            << ", main_records_loaded=" << profile.main_records_loaded
            << ", pooled_records=" << profile.pooled_records
            << ", repartitioned_records=" << profile.repartitioned_records
            << ", patch_records=" << profile.patch_records
            << ", patch_dense_vector_bytes=" << profile.patch_dense_vector_bytes
            << ", patch_retained_vector_bytes=" << profile.patch_retained_vector_bytes
            << ", patch_elided_vector_bytes=" << profile.patch_elided_vector_bytes
            << ", patch_final_pq_code_bytes=" << profile.patch_final_pq_code_bytes
            << ", pq_codes_reused=" << profile.pq_codes_reused
            << ", pq_codes_reencoded=" << profile.pq_codes_reencoded
            << ", prepare_pq_codes_reused=" << profile.prepare_pq_codes_reused
            << ", prepare_pq_codes_reencoded=" << profile.prepare_pq_codes_reencoded
            << ", merge_delta_to_main_assignment_us="
            << profile.merge_delta_to_main_assignment_us
            << ", merge_assignment_distance_us=" << profile.merge_assignment_distance_us
            << ", merge_assignment_top_r_us=" << profile.merge_assignment_top_r_us
            << ", merge_assignment_balance_us=" << profile.merge_assignment_balance_us
            << ", merge_assignment_materialize_us="
            << profile.merge_assignment_materialize_us
            << ", stats_us=" << profile.stats_us
            << ", scoring_us=" << profile.scoring_us
            << ", top_r_neighbor_us=" << profile.top_r_neighbor_us
            << ", fetch_main_records_us=" << profile.fetch_main_records_us
            << ", repartition_pool_us=" << profile.repartition_pool_us
            << ", repartition_distance_us=" << profile.repartition_distance_us
            << ", repartition_candidate_selection_us="
            << profile.repartition_candidate_selection_us
            << ", repartition_sort_us=" << profile.repartition_sort_us
            << ", patch_prepare_us=" << profile.patch_prepare_us
            << ", prepare_pq_encode_us=" << profile.prepare_pq_encode_us
            << ", commit_us=" << profile.commit_us
            << ", pq_code_assignment_us=" << profile.pq_code_assignment_us
            << ", pq_code_copy_or_reuse_us=" << profile.pq_code_copy_or_reuse_us
            << ", pq_list_flatten_us=" << profile.pq_list_flatten_us << std::endl;
}

void PrintGlobalRebuildBuildProfile(const IVFBuildProfiling& profile) {
  std::cout << "[GLOBAL_REBUILD_BUILD_PROFILE]"
            << " build_coarse_centroid_init_us=" << profile.build_coarse_centroid_init_us
            << ", build_coarse_kmeans_us=" << profile.build_coarse_kmeans_us
            << ", build_coarse_kmeans_assignment_us="
            << profile.build_coarse_kmeans_assignment_us
            << ", build_coarse_kmeans_update_us=" << profile.build_coarse_kmeans_update_us
            << ", build_coarse_kmeans_rows=" << profile.build_coarse_kmeans_rows
            << ", build_coarse_kmeans_k=" << profile.build_coarse_kmeans_k
            << ", build_coarse_kmeans_dim=" << profile.build_coarse_kmeans_dim
            << ", build_coarse_kmeans_iterations=" << profile.build_coarse_kmeans_iterations
            << ", build_pq_routing_assignment_us="
            << profile.build_pq_routing_assignment_us
            << ", build_pq_routing_assignment_rows="
            << profile.build_pq_routing_assignment_rows
            << ", build_pq_subspace_materialize_us="
            << profile.build_pq_subspace_materialize_us
            << ", build_pq_subspace_materialized_rows="
            << profile.build_pq_subspace_materialized_rows
            << ", build_pq_subspace_materialized_bytes="
            << profile.build_pq_subspace_materialized_bytes
            << ", build_pq_centroid_init_us=" << profile.build_pq_centroid_init_us
            << ", build_pq_kmeans_assignment_us="
            << profile.build_pq_kmeans_assignment_us
            << ", build_pq_kmeans_update_us=" << profile.build_pq_kmeans_update_us
            << ", build_pq_kmeans_us=" << profile.build_pq_kmeans_us
            << ", build_pq_training_total_us=" << profile.build_pq_training_total_us
            << ", build_pq_codebook_soa_us=" << profile.build_pq_codebook_soa_us
            << ", build_pq_precomputed_table_us=" << profile.build_pq_precomputed_table_us
            << ", build_publication_us=" << profile.build_publication_us
            << ", build_pq_max_live_subspaces=" << profile.build_pq_max_live_subspaces
            << ", build_pq_full_residual_bytes=" << profile.build_pq_full_residual_bytes
            << ", build_pq_training_concurrency=" << profile.build_pq_training_concurrency
            << std::endl;
}

void PrintGlobalRebuildAddProfile(const IngestProfiling& profile) {
  std::cout << "[GLOBAL_REBUILD_ADD_PROFILE]"
            << " records=" << profile.records
            << ", record_construction_us=" << profile.record_construction_us
            << ", validation_us=" << profile.validation_us
            << ", assignment_us=" << profile.assignment_us
            << ", encode_us=" << profile.encode_us
            << ", deferred_pq_stats_us=" << profile.deferred_pq_stats_us
            << ", commit_us=" << profile.commit_us << std::endl;
}

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
                                      double unaccounted_ms) {
  std::cout << "[GLOBAL_REBUILD_PROFILE_SUMMARY]"
            << " setup_ms=" << setup_ms
            << ", whitening_fit_ms=" << whitening_ms
            << ", base_retarget_ms=" << base_retarget_ms
            << ", query_transform_ms=" << query_transform_ms
            << ", main_build_ms=" << main_build_ms
            << ", main_add_ms=" << main_add_ms
            << ", delta_build_ms=" << delta_build_ms
            << ", delta_add_ms=" << delta_add_ms
            << ", delta_finalize_ms=" << delta_finalize_ms
            << ", delta_seed_ms=" << delta_seed_ms
            << ", norms_recompute_ms=" << norms_recompute_ms
            << ", publication_ms=" << publication_ms
            << ", malloc_trim_ms=" << malloc_trim_ms
            << ", memory_trace_ms=" << memory_trace_ms
            << ", profiling_collect_ms=" << profiling_collect_ms
            << ", accounted_stage_sum_ms=" << accounted_stage_sum_ms
            << ", wall_total_ms=" << wall_total_ms
            << ", unaccounted_ms=" << unaccounted_ms << std::endl;
}

Status WriteMemoryTraceJson(
    eval_memory::MemoryTraceRecorder& recorder,
    const std::string& path,
    const std::vector<eval_memory::MemoryComponent>& metadata) {
  return recorder.WriteJson(path, metadata);
}

namespace {

uint64_t ReadProcStatusBytes(const std::string& key) {
  std::ifstream ifs("/proc/self/status");
  if (!ifs) return 0;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind(key, 0) != 0) continue;
    std::istringstream iss(line.substr(key.size()));
    uint64_t kb = 0;
    std::string unit;
    iss >> kb >> unit;
    return kb * 1024ull;
  }
  return 0;
}

uint32_t RuntimeMaxThreads() {
#ifdef _OPENMP
  return static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
#else
  return 1;
#endif
}

}  // namespace

using namespace run_eval_ms;

void WriteSummaryJson(const std::string& path,
                      const Config& config,
                      const Workload& workload,
                      const std::vector<SnapshotLite>& snapshots,
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
                      const SummaryOutput& summary) {
  std::ofstream ofs(path);

  if (!ofs) {
    std::cerr << "[WARN] Cannot write summary JSON: " << path << std::endl;
    return;
  }

  const auto& final_snapshot = summary.final_snapshot;
  const double final_recall = summary.final_recall;
  const double total_maintenance_ms = summary.total_maintenance_ms;
  const double amortized_update_throughput = summary.amortized_update_throughput;
  const double maintenance_ms_per_vector = summary.maintenance_ms_per_vector;
  const double final_imbalance_ratio = summary.final_imbalance_ratio;
  const double final_scan_ratio = summary.final_scan_ratio;
  const double final_scanned_per_topk = summary.final_scanned_per_topk;
  const uint64_t summary_gt_new_total = summary.summary_gt_new_total;
  const uint64_t summary_gt_old_total = summary.summary_gt_old_total;
  const uint64_t summary_hit_new_total = summary.summary_hit_new_total;
  const uint64_t summary_hit_old_total = summary.summary_hit_old_total;
  const double summary_recall_new = summary.summary_recall_new;
  const double summary_recall_old = summary.summary_recall_old;
  const double summary_gt_new_ratio = summary.summary_gt_new_ratio;

  ofs << "{\n";
  ofs << "  \"workload\": \"" << workload.path << "\",\n";
  ofs << "  \"dataset\": \"" << workload.dataset << "\",\n";
  ofs << "  \"final_seen_rows\": " << final_seen_rows << ",\n";
  ofs << "  \"topk\": " << config.topk << ",\n";
  ofs << "  \"nprobe\": " << config.nprobe << ",\n";
  ofs << "  \"use_whitening\": " << (config.use_whitening ? "true" : "false") << ",\n";
  ofs << "  \"use_cosine\": " << (config.use_cosine ? "true" : "false") << ",\n";
  ofs << "  \"omp_max_threads\": " << RuntimeMaxThreads() << ",\n";
  ofs << "  \"final_recall\": " << final_recall << ",\n";
  ofs << "  \"full_run_wall_ms\": " << full_run_wall_ms << ",\n";
  ofs << "  \"build_rebuild_ms\": " << summary.build_rebuild_ms << ",\n";
  ofs << "  \"recall_available\": " << (summary.recall_available ? "true" : "false") << ",\n";
  ofs << "  \"avg_query_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->avg_query_ms : 0.0) << ",\n";
  ofs << "  \"latency_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->avg_query_ms : 0.0) << ",\n";
  ofs << "  \"avg_whiten_ms\": 0,\n";
  ofs << "  \"whitening_p50\": 0,\n";
  ofs << "  \"whitening_p99\": 0,\n";
  ofs << "  \"end_to_end_overhead_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->end_to_end_overhead_ms : 0.0) << ",\n";
  ofs << "  \"query_qps\": " << (final_snapshot != nullptr ? final_snapshot->qps : 0.0)
      << ",\n";
  ofs << "  \"avg_pq_lut_build_us\": "
      << (final_snapshot != nullptr ? final_snapshot->avg_pq_lut_build_us : 0.0) << ",\n";
  ofs << "  \"avg_pq_adc_scan_us\": "
      << (final_snapshot != nullptr ? final_snapshot->avg_pq_adc_scan_us : 0.0) << ",\n";
  ofs << "  \"search_p50_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->search_p50 : 0.0) << ",\n";
  ofs << "  \"search_p50\": "
      << (final_snapshot != nullptr ? final_snapshot->search_p50 : 0.0) << ",\n";
  ofs << "  \"search_p99_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->search_p99 : 0.0) << ",\n";
  ofs << "  \"search_p99\": "
      << (final_snapshot != nullptr ? final_snapshot->search_p99 : 0.0) << ",\n";
  ofs << "  \"total_p50_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->total_p50 : 0.0) << ",\n";
  ofs << "  \"total_p50\": "
      << (final_snapshot != nullptr ? final_snapshot->total_p50 : 0.0) << ",\n";
  ofs << "  \"total_p99_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->total_p99 : 0.0) << ",\n";
  ofs << "  \"total_p99\": "
      << (final_snapshot != nullptr ? final_snapshot->total_p99 : 0.0) << ",\n";
  ofs << "  \"scanned_avg\": "
      << (final_snapshot != nullptr ? final_snapshot->avg_scanned : 0.0) << ",\n";
  ofs << "  \"scanned_p50\": "
      << (final_snapshot != nullptr ? final_snapshot->scanned_p50 : 0.0) << ",\n";
  ofs << "  \"scanned_p99\": "
      << (final_snapshot != nullptr ? final_snapshot->scanned_p99 : 0.0) << ",\n";
  ofs << "  \"scanned_max\": "
      << (final_snapshot != nullptr ? final_snapshot->scanned_max : 0.0) << ",\n";
  ofs << "  \"query_eval_ms\": "
      << (final_snapshot != nullptr ? final_snapshot->query_eval_ms : 0.0) << ",\n";
  ofs << "  \"query_count\": "
      << (final_snapshot != nullptr ? final_snapshot->query_count : 0) << ",\n";
  ofs << "  \"route_execution\": {\n";
  ofs << "    \"main_queries\": "
      << (final_snapshot != nullptr ? final_snapshot->main_route_queries : 0) << ",\n";
  ofs << "    \"frozen_delta_queries\": "
      << (final_snapshot != nullptr ? final_snapshot->frozen_delta_route_queries : 0) << ",\n";
  ofs << "    \"active_delta_queries\": "
      << (final_snapshot != nullptr ? final_snapshot->active_delta_route_queries : 0) << "\n";
  ofs << "  },\n";
  ofs << "  \"update_total_ms\": " << total_maintenance_ms << ",\n";
  ofs << "  \"update_apply_ms\": " << update_ms << ",\n";
  ofs << "  \"update_whitening_ms\": " << update_whitening_ms << ",\n";
  ofs << "  \"update_insert_ms\": " << update_insert_ms << ",\n";
  ofs << "  \"update_record_build_ms\": " << update_record_build_ms << ",\n";
  ofs << "  \"update_insert_encode_ms\": " << update_insert_encode_ms << ",\n";
  ofs << "  \"update_insert_commit_ms\": " << update_insert_commit_ms << ",\n";
  ofs << "  \"update_onlinepq_maintenance_ms\": " << update_onlinepq_maintenance_ms << ",\n";
  ofs << "  \"update_delete_ms\": " << update_delete_ms << ",\n";
  ofs << "  \"update_codebook_update_ms\": " << update_codebook_update_ms << ",\n";
  ofs << "  \"update_reencode_ms\": " << update_reencode_ms << ",\n";
  ofs << "  \"delta_ingest_assignment_us\": " << delta_ingest_assignment_us << ",\n";
  ofs << "  \"delta_ingest_assignment_records\": " << delta_ingest_assignment_records << ",\n";
  ofs << "  \"update_merge_compute_ms\": " << merge_compute_ms << ",\n";
  ofs << "  \"update_global_rebuild_ms\": " << global_rebuild_ms << ",\n";
  ofs << "  \"update_per_vector_ms\": " << summary.update_per_vector_ms << ",\n";
  ofs << "  \"update_throughput_vecps\": " << summary.update_throughput_vecps << ",\n";
  ofs << "  \"gt_probed_rate\": "
      << (final_snapshot != nullptr ? final_snapshot->gt_probed_rate : 0.0) << ",\n";
  ofs << "  \"recall_on_probed_gt\": "
      << (final_snapshot != nullptr ? final_snapshot->recall_on_probed_gt : 0.0) << ",\n";
  ofs << "  \"exact_recall_on_probed_candidates\": "
      << (final_snapshot != nullptr ? final_snapshot->exact_recall_on_probed_candidates : 0.0)
      << ",\n";
  ofs << "  \"avg_pq_rank_loss\": "
      << (final_snapshot != nullptr ? final_snapshot->avg_pq_rank_loss : 0.0) << ",\n";
  ofs << "  \"miss_not_probed\": "
      << (final_snapshot != nullptr ? final_snapshot->miss_not_probed : 0) << ",\n";
  ofs << "  \"miss_probed_filtered_by_pq\": "
      << (final_snapshot != nullptr ? final_snapshot->miss_probed_filtered_by_pq : 0) << ",\n";
  ofs << "  \"pq_rank_loss_count\": "
      << (final_snapshot != nullptr ? final_snapshot->pq_rank_loss_count : 0) << ",\n";
  ofs << "  \"rerank_topk_main_total\": "
      << (final_snapshot != nullptr ? final_snapshot->rerank_topk_main_total : 0) << ",\n";
  ofs << "  \"rerank_topk_delta_total\": "
      << (final_snapshot != nullptr ? final_snapshot->rerank_topk_delta_total : 0) << ",\n";
  ofs << "  \"rerank_topk_main_ratio\": "
      << (final_snapshot != nullptr ? final_snapshot->rerank_topk_main_ratio : 0.0) << ",\n";
  ofs << "  \"rerank_topk_delta_ratio\": "
      << (final_snapshot != nullptr ? final_snapshot->rerank_topk_delta_ratio : 0.0) << ",\n";
  ofs << "  \"rerank_topk_main_avg\": "
      << (final_snapshot != nullptr ? final_snapshot->rerank_topk_main_avg : 0.0) << ",\n";
  ofs << "  \"rerank_topk_delta_avg\": "
      << (final_snapshot != nullptr ? final_snapshot->rerank_topk_delta_avg : 0.0) << ",\n";
  ofs << "  \"miss_diag_available\": "
      << (config.enable_miss_diag ? "true" : "false") << ",\n";
  ofs << "  \"rerank_source_diag_available\": "
      << (config.enable_rerank_source_diag ? "true" : "false") << ",\n";
  ofs << "  \"init_ms\": " << init_ms << ",\n";
  ofs << "  \"update_ms\": " << update_ms << ",\n";
  ofs << "  \"merge_ms\": " << merge_ms << ",\n";
  ofs << "  \"global_rebuild_ms\": " << global_rebuild_ms << ",\n";
  ofs << "  \"merge_count\": " << merge_count << ",\n";
  ofs << "  \"global_rebuild_count\": " << global_rebuild_count << ",\n";
  ofs << "  \"merge_nodes\": [";
  for (size_t ni = 0; ni < merge_nodes.size(); ++ni) {
    ofs << merge_nodes[ni];
    if (ni + 1 < merge_nodes.size()) ofs << ", ";
  }
  ofs << "],\n";
  ofs << "  \"global_rebuild_nodes\": [";
  for (size_t ni = 0; ni < global_rebuild_nodes.size(); ++ni) {
    ofs << global_rebuild_nodes[ni];
    if (ni + 1 < global_rebuild_nodes.size()) ofs << ", ";
  }
  ofs << "],\n";
  ofs << "  \"merge_compute_ms\": " << merge_compute_ms << ",\n";
  ofs << "  \"total_maintenance_ms\": " << total_maintenance_ms << ",\n";
  ofs << "  \"maintenance_ms_per_vector\": " << maintenance_ms_per_vector << ",\n";
  ofs << "  \"imbalance_ratio\": " << final_imbalance_ratio << ",\n";
  ofs << "  \"scan_ratio\": " << final_scan_ratio << ",\n";
  ofs << "  \"scanned_per_topk\": " << final_scanned_per_topk << ",\n";
  ofs << "  \"amortized_update_throughput_vecps\": "
      << amortized_update_throughput << ",\n";
  ofs << "  \"gt_scope\": \"current_prefix\",\n";
  ofs << "  \"summary\": {\n";
  ofs << "    \"recall_avg\": " << summary.recall_stats.avg << ",\n";
  ofs << "    \"recall_p5\": " << summary.recall_stats.p5 << ",\n";
  ofs << "    \"recall_min\": " << summary.recall_stats.min << ",\n";
  ofs << "    \"recall_final\": " << final_recall << ",\n";
  ofs << "    \"recall_new\": " << summary_recall_new << ",\n";
  ofs << "    \"recall_old\": " << summary_recall_old << ",\n";
  ofs << "    \"gt_new_ratio\": " << summary_gt_new_ratio << ",\n";
  ofs << "    \"gt_new_total\": " << summary_gt_new_total << ",\n";
  ofs << "    \"gt_old_total\": " << summary_gt_old_total << ",\n";
  ofs << "    \"hit_new_total\": " << summary_hit_new_total << ",\n";
  ofs << "    \"hit_old_total\": " << summary_hit_old_total << ",\n";
  ofs << "    \"qps_avg\": " << summary.qps_stats.avg << ",\n";
  ofs << "    \"qps_p5\": " << summary.qps_stats.p5 << ",\n";
  ofs << "    \"latency_avg_ms\": " << summary.latency_stats.avg << ",\n";
  ofs << "    \"latency_p95_ms\": " << summary.latency_stats.p95 << ",\n";
  ofs << "    \"latency_p99_ms\": " << summary.latency_stats.p99 << ",\n";
  ofs << "    \"scanned_avg\": " << summary.scanned_stats.avg << ",\n";
  ofs << "    \"update_throughput_avg_vecps\": " << summary.update_throughput_stats.avg << ",\n";
  ofs << "    \"process_rss_bytes\": " << ReadProcStatusBytes("VmRSS:") << ",\n";
  ofs << "    \"process_peak_rss_bytes\": " << ReadProcStatusBytes("VmHWM:") << "\n";
  ofs << "  },\n";
  ofs << "  \"snapshots\": [\n";

  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto& s = snapshots[i];

    ofs << "    {\"search_id\": " << s.search_id
        << ", \"op_id\": " << s.op_id
        << ", \"active_rows\": " << s.active_rows
        << ", \"main_rows\": " << s.main_rows
        << ", \"active_delta_rows\": " << s.active_delta_rows
        << ", \"frozen_delta_rows\": " << s.frozen_delta_rows
        << ", \"round\": " << s.round
        << ", \"cluster\": " << s.cluster
        << ", \"recall\": " << s.recall
        << ", \"recall_new\": " << s.recall_new
        << ", \"recall_old\": " << s.recall_old
        << ", \"gt_new_ratio\": " << s.gt_new_ratio
        << ", \"gt_new_total\": " << s.gt_new_total
        << ", \"gt_old_total\": " << s.gt_old_total
        << ", \"hit_new_total\": " << s.hit_new_total
        << ", \"hit_old_total\": " << s.hit_old_total
        << ", \"avg_query_ms\": " << s.avg_query_ms
        << ", \"latency_ms\": " << s.avg_query_ms
        << ", \"avg_whiten_ms\": 0"
        << ", \"whitening_p50\": 0"
        << ", \"whitening_p99\": 0"
        << ", \"end_to_end_overhead_ms\": " << s.end_to_end_overhead_ms
        << ", \"avg_search_ms\": " << s.avg_search_ms
        << ", \"avg_pq_lut_build_us\": " << s.avg_pq_lut_build_us
        << ", \"avg_pq_adc_scan_us\": " << s.avg_pq_adc_scan_us
        << ", \"search_p50_ms\": " << s.search_p50
        << ", \"search_p50\": " << s.search_p50
        << ", \"search_p99_ms\": " << s.search_p99
        << ", \"search_p99\": " << s.search_p99
        << ", \"total_p50_ms\": " << s.total_p50
        << ", \"total_p50\": " << s.total_p50
        << ", \"total_p99_ms\": " << s.total_p99
        << ", \"total_p99\": " << s.total_p99
        << ", \"p50_query_ms\": " << s.p50_query_ms
        << ", \"p99_query_ms\": " << s.p99_query_ms
        << ", \"qps\": " << s.qps
        << ", \"avg_scanned\": " << s.avg_scanned
        << ", \"scanned_p50\": " << s.scanned_p50
        << ", \"scanned_p99\": " << s.scanned_p99
        << ", \"scanned_max\": " << s.scanned_max
        << ", \"scan_ratio\": " << s.scan_ratio
        << ", \"scanned_per_topk\": " << s.scanned_per_topk
        << ", \"query_eval_ms\": " << s.query_eval_ms
        << ", \"query_count\": " << s.query_count
        << ", \"main_route_queries\": " << s.main_route_queries
        << ", \"frozen_delta_route_queries\": " << s.frozen_delta_route_queries
        << ", \"active_delta_route_queries\": " << s.active_delta_route_queries
        << ", \"gt_probed_rate\": " << s.gt_probed_rate
        << ", \"recall_on_probed_gt\": " << s.recall_on_probed_gt
        << ", \"exact_recall_on_probed_candidates\": "
        << s.exact_recall_on_probed_candidates
        << ", \"avg_pq_rank_loss\": " << s.avg_pq_rank_loss
        << ", \"miss_not_probed\": " << s.miss_not_probed
        << ", \"miss_probed_filtered_by_pq\": " << s.miss_probed_filtered_by_pq
        << ", \"pq_rank_loss_count\": " << s.pq_rank_loss_count
        << ", \"rerank_topk_main_total\": " << s.rerank_topk_main_total
        << ", \"rerank_topk_delta_total\": " << s.rerank_topk_delta_total
        << ", \"rerank_topk_main_ratio\": " << s.rerank_topk_main_ratio
        << ", \"rerank_topk_delta_ratio\": " << s.rerank_topk_delta_ratio
        << ", \"rerank_topk_main_avg\": " << s.rerank_topk_main_avg
        << ", \"rerank_topk_delta_avg\": " << s.rerank_topk_delta_avg
        << ", \"imbalance_ratio\": " << s.imbalance_ratio
        << ", \"cumulative_update_ms\": " << s.cumulative_update_ms
        << ", \"cumulative_merge_ms\": " << s.cumulative_merge_ms
        << ", \"cumulative_global_rebuild_ms\": " << s.cumulative_global_rebuild_ms
        << ", \"snapshot_rows\": " << s.snapshot_rows
        << ", \"snapshot_maintenance_ms\": " << s.snapshot_maintenance_ms
        << ", \"maintenance_ms_per_vector\": " << s.maintenance_ms_per_vector
        << ", \"update_throughput_vecps\": " << s.update_throughput_vecps
        << ", \"amortized_update_throughput_vecps\": " << s.amortized_update_throughput_vecps
        << ", \"skipped_warmup\": "
        << (s.skipped_warmup ? "true" : "false")
        << ", \"worst_queries\": [";
    for (size_t wi = 0; wi < s.worst_queries.size(); ++wi) {
      ofs << "\"" << s.worst_queries[wi] << "\"";
      if (wi + 1 < s.worst_queries.size()) ofs << ", ";
    }
    ofs << "]}";

    if (i + 1 < snapshots.size()) ofs << ",";
    ofs << "\n";
  }

  ofs << "  ]\n";
  ofs << "}\n";
}
Status WriteRunEvalResultJson(const std::filesystem::path& path,
                              const RunEvalResultData& data) {
  std::ofstream ofs(path);
  if (!ofs) return Status::IOError("Failed to write results to " + path.string());
  const auto& dataset_label = data.dataset_label;
  const auto ts = data.timestamp;
  const auto& config = data.config;
  const auto& params = data.params;
  const auto& final_metrics = data.final_metrics;
  const auto& snapshots = data.snapshots;
  const auto& merge_events = data.merge_events;
  const auto& global_rebuild_events = data.global_rebuild_events;
  const auto& counters = data.counters;
  const auto& total_update_ms = counters.total_update_ms;
  const auto& total_update_whitening_ms = counters.total_update_whitening_ms;
  const auto& total_update_insert_ms = counters.total_update_insert_ms;
  const auto& total_update_record_build_ms = counters.total_update_record_build_ms;
  const auto& total_update_insert_encode_ms = counters.total_update_insert_encode_ms;
  const auto& total_update_insert_commit_ms = counters.total_update_insert_commit_ms;
  const auto& total_update_onlinepq_maintenance_ms = counters.total_update_onlinepq_maintenance_ms;
  const auto& total_update_delete_ms = counters.total_update_delete_ms;
  const auto& total_update_codebook_update_ms = counters.total_update_codebook_update_ms;
  const auto& total_update_reencode_ms = counters.total_update_reencode_ms;
  const auto& total_delta_ingest_assignment_us = counters.total_delta_ingest_assignment_us;
  const auto& total_delta_ingest_assignment_records = counters.total_delta_ingest_assignment_records;
  const auto& total_merge_compute_ms = counters.total_merge_compute_ms;
  const auto& total_global_rebuild_ms = counters.total_global_rebuild_ms;
  const auto& global_rebuild_count = counters.global_rebuild_count;
  const auto& rebuild_ms_total = counters.rebuild_ms_total;
  const auto& online_pq_rollup = counters.online_pq_rollup;
  const auto& recall_summary = data.recall_summary;
  const auto& qps_summary = data.qps_summary;
  const auto& latency_summary = data.latency_summary;
  const auto& e2e_latency_summary = data.e2e_latency_summary;
  const auto& update_throughput_summary = data.update_throughput_summary;
  const auto& amortized_update_throughput_summary = data.amortized_update_throughput_summary;
  const auto summary_gt_new_total = data.summary_gt_new_total;
  const auto summary_gt_old_total = data.summary_gt_old_total;
  const auto summary_hit_new_total = data.summary_hit_new_total;
  const auto summary_hit_old_total = data.summary_hit_old_total;
  const auto summary_recall_new = data.summary_recall_new;
  const auto summary_recall_old = data.summary_recall_old;
  const auto summary_gt_new_ratio = data.summary_gt_new_ratio;
  const auto used_prebuilt_index = data.used_prebuilt_index;
  const auto init_load_ms = data.init_load_ms;
  const auto init_rebuild_ms = data.init_rebuild_ms;
  const auto init_main_build_ms = data.init_main_build_ms;
  const auto init_main_add_ms = data.init_main_add_ms;
  const auto init_total_wall_ms = data.init_total_wall_ms;
  const auto init_whitening_ms = data.init_whitening_ms;
  const auto init_whitening_transform_ms = data.init_whitening_transform_ms;
  const auto& initial_main_ingest_profile = data.initial_main_ingest_profile;
  const auto& initial_delta_seed_ingest_profile = data.initial_delta_seed_ingest_profile;
  const auto& initial_build_profile = data.initial_build_profile;
  const auto main_rows_initial = data.main_rows_initial;
  const auto main_rows_current = data.main_rows_current;
  const auto rows_after_main = data.rows_after_main;
  const auto delta_train_rows = data.delta_train_rows;
  const auto delta_ivf_nlist = data.delta_ivf_nlist;
  const auto merge_trigger_rows = data.merge_trigger_rows;
  const auto stream_start_idx = data.stream_start_idx;
  const auto total_stream_rows = data.total_stream_rows;
  const auto snapshot_span = data.snapshot_span;
  const auto sliding_window_rows = data.sliding_window_rows;
  const auto kDeltaKMeansIterationsDefault = data.delta_kmeans_iterations;
  const auto& pre_stream_metrics = data.pre_stream_metrics;
  const auto online_pq_enabled = data.online_pq_enabled;
  const auto online_avg_nqe = data.online_avg_nqe;
  const auto online_avg_qe_ratio = data.online_avg_qe_ratio;
  const auto online_avg_drift = data.online_avg_drift;
  const auto debug_output_enabled = data.debug_output_enabled;
  const auto full_run_wall_ms = data.full_run_wall_ms;
  const auto ground_truth_available = data.ground_truth_available;
  const double total_maintenance_ms = data.total_maintenance_ms;
  if (!debug_output_enabled) {
    ofs << "{\n";
    ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
    ofs << "  \"timestamp\": " << ts << ",\n";
    ofs << "  \"params\": {\n";
    ofs << "    \"topk\": " << config.topk << ",\n";
    ofs << "    \"enable_dynamic_ground_truth\": "
        << (config.enable_dynamic_ground_truth ? "true" : "false") << ",\n";
    ofs << "    \"skip_query_ground_truth\": "
        << (config.skip_query_ground_truth ? "true" : "false") << ",\n";
    ofs << "    \"nprobe\": " << params.nprobe << ",\n";
    ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
    ofs << "    \"omp_max_threads\": " << RuntimeMaxThreads() << ",\n";
    ofs << "    \"pq_codebook_dimension_major\": "
        << (config.pq_codebook_dimension_major ? "true" : "false") << ",\n";
    ofs << "    \"pq_codes_subquantizer_major\": "
        << (config.pq_codes_subquantizer_major ? "true" : "false") << ",\n";
    ofs << "    \"use_whitening\": " << (config.use_whitening ? "true" : "false") << ",\n";
    ofs << "    \"use_cosine\": " << (config.use_cosine ? "true" : "false") << ",\n";
    ofs << "    \"main_exact_rerank_candidates\": " << config.main_exact_rerank_candidates
        << ",\n";
    ofs << "    \"active_exact_rerank_candidates\": " << config.active_exact_rerank_candidates
        << ",\n";
    ofs << "    \"frozen_exact_rerank_candidates\": " << config.frozen_exact_rerank_candidates
        << ",\n";
    ofs << "    \"merge_score_alpha\": " << config.merge_score_alpha << ",\n";
    ofs << "    \"merge_score_beta\": " << config.merge_score_beta << ",\n";
    ofs << "    \"merge_score_threshold\": " << config.merge_score_threshold << ",\n";
    ofs << "    \"enable_merge\": " << (config.enable_merge ? "true" : "false") << ",\n";
    ofs << "    \"snapshot_span\": " << snapshot_span << "\n";
    ofs << "  },\n";
    ofs << "  \"metrics\": {\n";
    ofs << "    \"recall_available\": "
        << (final_metrics.recall_available ? "true" : "false") << ",\n";
    ofs << "    \"recall@" << config.topk << "\": " << final_metrics.recall << ",\n";
    ofs << "    \"full_run_wall_ms\": " << full_run_wall_ms << ",\n";
    ofs << "    \"recall_new\": " << final_metrics.recall_new << ",\n";
    ofs << "    \"recall_old\": " << final_metrics.recall_old << ",\n";
    ofs << "    \"gt_new_ratio\": " << final_metrics.gt_new_ratio << ",\n";
    ofs << "    \"gt_new_total\": " << final_metrics.gt_new_total << ",\n";
    ofs << "    \"gt_old_total\": " << final_metrics.gt_old_total << ",\n";
    ofs << "    \"hit_new_total\": " << final_metrics.hit_new_total << ",\n";
    ofs << "    \"hit_old_total\": " << final_metrics.hit_old_total << ",\n";
    ofs << "    \"latency_ms\": " << final_metrics.avg_query_ms << ",\n";
    ofs << "    \"end_to_end_overhead_ms\": " << final_metrics.end_to_end_overhead_ms << ",\n";
    ofs << "    \"query_qps\": " << final_metrics.query_qps << ",\n";
    ofs << "    \"avg_pq_lut_build_us\": " << final_metrics.avg_pq_lut_build_us << ",\n";
    ofs << "    \"avg_pq_adc_scan_us\": " << final_metrics.avg_pq_adc_scan_us << ",\n";
    ofs << "    \"update_apply_ms\": " << total_update_ms << ",\n";
    ofs << "    \"update_whitening_ms\": " << total_update_whitening_ms << ",\n";
    ofs << "    \"update_insert_ms\": " << total_update_insert_ms << ",\n";
    ofs << "    \"update_record_build_ms\": " << total_update_record_build_ms << ",\n";
    ofs << "    \"update_insert_encode_ms\": " << total_update_insert_encode_ms << ",\n";
    ofs << "    \"update_insert_commit_ms\": " << total_update_insert_commit_ms << ",\n";
    ofs << "    \"update_onlinepq_maintenance_ms\": " << total_update_onlinepq_maintenance_ms << ",\n";
    ofs << "    \"update_delete_ms\": " << total_update_delete_ms << ",\n";
    ofs << "    \"update_codebook_update_ms\": " << total_update_codebook_update_ms << ",\n";
    ofs << "    \"update_reencode_ms\": " << total_update_reencode_ms << ",\n";
    ofs << "    \"delta_ingest_assignment_us\": "
        << total_delta_ingest_assignment_us << ",\n";
    ofs << "    \"delta_ingest_assignment_records\": "
        << total_delta_ingest_assignment_records << ",\n";
    ofs << "    \"update_merge_compute_ms\": " << total_merge_compute_ms << ",\n";
    ofs << "    \"update_global_rebuild_ms\": " << total_global_rebuild_ms << ",\n";
    ofs << "    \"initial_used_prebuilt_index\": "
        << (used_prebuilt_index ? "true" : "false") << ",\n";
    ofs << "    \"initial_index_load_ms\": " << init_load_ms << ",\n";
    ofs << "    \"initial_build_aligned_ms\": " << init_rebuild_ms << ",\n";
    ofs << "    \"initial_main_build_ms\": " << init_main_build_ms << ",\n";
    ofs << "    \"initial_main_add_ms\": " << init_main_add_ms << ",\n";
    ofs << "    \"initial_main_pq_encode_us\": "
        << initial_main_ingest_profile.encode_us << ",\n";
    ofs << "    \"initial_delta_seed_pq_encode_us\": "
        << initial_delta_seed_ingest_profile.encode_us << ",\n";
    ofs << "    \"initial_build_profile_available\": "
        << (initial_build_profile.has_value() ? "true" : "false") << ",\n";
    ofs << "    \"initial_build_coarse_kmeans_us\": "
        << (initial_build_profile ? initial_build_profile->build_coarse_kmeans_us : 0.0) << ",\n";
    ofs << "    \"initial_build_coarse_assignment_us\": "
        << (initial_build_profile ? initial_build_profile->build_coarse_kmeans_assignment_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_coarse_update_us\": "
        << (initial_build_profile ? initial_build_profile->build_coarse_kmeans_update_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_pq_training_us\": "
        << (initial_build_profile ? initial_build_profile->build_pq_training_total_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_pq_codebook_soa_us\": "
        << (initial_build_profile ? initial_build_profile->build_pq_codebook_soa_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_pq_precomputed_table_us\": "
        << (initial_build_profile ? initial_build_profile->build_pq_precomputed_table_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_pq_assignment_us\": "
        << (initial_build_profile ? initial_build_profile->build_pq_kmeans_assignment_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_pq_update_us\": "
        << (initial_build_profile ? initial_build_profile->build_pq_kmeans_update_us : 0.0)
        << ",\n";
    ofs << "    \"initial_build_publication_us\": "
        << (initial_build_profile ? initial_build_profile->build_publication_us : 0.0) << ",\n";
    ofs << "    \"initial_build_wall_ms\": " << init_total_wall_ms << ",\n";
    ofs << "    \"total_maintenance_ms\": " << total_maintenance_ms << ",\n";
    ofs << "    \"throughput\": " << final_metrics.update_throughput_vecps << "\n";
    ofs << "  },\n";
    ofs << "  \"summary\": {\n";
    ofs << "    \"recall_avg\": " << recall_summary.avg << ",\n";
    ofs << "    \"recall_p5\": " << recall_summary.p5 << ",\n";
    ofs << "    \"recall_min\": " << recall_summary.min << ",\n";
    ofs << "    \"recall_final\": " << final_metrics.recall << ",\n";
    ofs << "    \"recall_new\": " << summary_recall_new << ",\n";
    ofs << "    \"recall_old\": " << summary_recall_old << ",\n";
    ofs << "    \"gt_new_ratio\": " << summary_gt_new_ratio << ",\n";
    ofs << "    \"gt_new_total\": " << summary_gt_new_total << ",\n";
    ofs << "    \"gt_old_total\": " << summary_gt_old_total << ",\n";
    ofs << "    \"hit_new_total\": " << summary_hit_new_total << ",\n";
    ofs << "    \"hit_old_total\": " << summary_hit_old_total << ",\n";
    ofs << "    \"qps_avg\": " << qps_summary.avg << ",\n";
    ofs << "    \"qps_p5\": " << qps_summary.p5 << ",\n";
    ofs << "    \"latency_avg_ms\": " << latency_summary.avg << ",\n";
    ofs << "    \"latency_p95_ms\": " << latency_summary.p95 << ",\n";
    ofs << "    \"latency_p99_ms\": " << latency_summary.p99 << ",\n";
    ofs << "    \"e2e_latency_avg_ms\": " << e2e_latency_summary.avg << ",\n";
    ofs << "    \"update_throughput_avg_vecps\": "
        << update_throughput_summary.avg << ",\n";
    ofs << "    \"amortized_update_throughput_avg_vecps\": "
        << amortized_update_throughput_summary.avg << ",\n";
    ofs << "    \"merge_count\": " << merge_events.size() << ",\n";
    ofs << "    \"global_rebuild_count\": " << global_rebuild_count << ",\n";
    ofs << "    \"merge_nodes\": [";
    for (size_t ni = 0; ni < merge_events.size(); ++ni) {
      ofs << merge_events[ni].base_rows;
      if (ni + 1 < merge_events.size()) ofs << ", ";
    }
    ofs << "],\n";
    ofs << "    \"global_rebuild_nodes\": [";
    for (size_t ni = 0; ni < global_rebuild_events.size(); ++ni) {
      ofs << global_rebuild_events[ni].base_rows;
      if (ni + 1 < global_rebuild_events.size()) ofs << ", ";
    }
    ofs << "],\n";
    ofs << "    \"process_rss_bytes\": " << ReadProcStatusBytes("VmRSS:") << ",\n";
    ofs << "    \"process_peak_rss_bytes\": " << ReadProcStatusBytes("VmHWM:") << "\n";
    ofs << "  },\n";
    ofs << "  \"route_execution\": {\n";
    ofs << "    \"main_queries\": " << final_metrics.main_route_queries << ",\n";
    ofs << "    \"frozen_delta_queries\": " << final_metrics.frozen_delta_route_queries << ",\n";
    ofs << "    \"active_delta_queries\": " << final_metrics.active_delta_route_queries << "\n";
    ofs << "  },\n";
    ofs << "  \"global_rebuild_count\": " << global_rebuild_count << ",\n";
    ofs << "  \"merge_events\": [\n";
    for (size_t i = 0; i < merge_events.size(); ++i) {
      const auto& ev = merge_events[i];
      ofs << "    {\n";
      ofs << "      \"base_rows\": " << ev.base_rows << ",\n";
      ofs << "      \"frozen_rows\": " << ev.frozen_rows << ",\n";
      ofs << "      \"merge_compute_ms\": " << ev.merge_compute_ms << ",\n";
      ofs << "      \"assignment_distance_us\": "
          << ev.profiling.merge_assignment_distance_us << ",\n";
      ofs << "      \"assignment_top_r_us\": "
          << ev.profiling.merge_assignment_top_r_us << ",\n";
      ofs << "      \"assignment_balance_us\": "
          << ev.profiling.merge_assignment_balance_us << ",\n";
      ofs << "      \"fetch_main_records_us\": "
          << ev.profiling.fetch_main_records_us << ",\n";
      ofs << "      \"repartition_distance_us\": "
          << ev.profiling.repartition_distance_us << ",\n";
      ofs << "      \"repartition_sort_us\": " << ev.profiling.repartition_sort_us << ",\n";
      ofs << "      \"prepare_pq_encode_us\": "
          << ev.profiling.prepare_pq_encode_us << ",\n";
      ofs << "      \"commit_us\": " << ev.profiling.commit_us << ",\n";
      ofs << "      \"pq_code_copy_or_reuse_us\": "
          << ev.profiling.pq_code_copy_or_reuse_us << ",\n";
      ofs << "      \"pq_list_flatten_us\": " << ev.profiling.pq_list_flatten_us << ",\n";
      ofs << "      \"merge_ms\": " << ev.merge_ms << ",\n";
      ofs << "      \"main_imbalance_after_real\": " << ev.main_imbalance_after_real << "\n";
      ofs << "    }";
      if (i + 1 < merge_events.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "  ],\n";
    ofs << "  \"snapshots\": [\n";
    for (size_t i = 0; i < snapshots.size(); ++i) {
      const auto& snap = snapshots[i];
      ofs << "    {\n";
      ofs << "      \"snapshot_size\": " << snap.base_rows << ",\n";
      ofs << "      \"recall_available\": "
          << (ground_truth_available ? "true" : "false") << ",\n";
      ofs << "      \"recall@" << config.topk << "\": " << snap.recall << ",\n";
      ofs << "      \"recall_new\": " << snap.recall_new << ",\n";
      ofs << "      \"recall_old\": " << snap.recall_old << ",\n";
      ofs << "      \"gt_new_ratio\": " << snap.gt_new_ratio << ",\n";
      ofs << "      \"gt_new_total\": " << snap.gt_new_total << ",\n";
      ofs << "      \"gt_old_total\": " << snap.gt_old_total << ",\n";
      ofs << "      \"hit_new_total\": " << snap.hit_new_total << ",\n";
      ofs << "      \"hit_old_total\": " << snap.hit_old_total << ",\n";
      ofs << "      \"latency_ms\": " << snap.latency_ms << ",\n";
      ofs << "      \"end_to_end_overhead_ms\": " << snap.end_to_end_overhead_ms << ",\n";
      ofs << "      \"query_qps\": " << snap.query_qps << ",\n";
      ofs << "      \"update_ms\": " << snap.update_ms << ",\n";
      ofs << "      \"update_whitening_ms\": " << snap.update_whitening_ms << ",\n";
      ofs << "      \"update_insert_ms\": " << snap.update_insert_ms << ",\n";
      ofs << "      \"update_record_build_ms\": " << snap.update_record_build_ms << ",\n";
      ofs << "      \"update_insert_encode_ms\": " << snap.update_insert_encode_ms << ",\n";
      ofs << "      \"update_insert_commit_ms\": " << snap.update_insert_commit_ms << ",\n";
      ofs << "      \"update_onlinepq_maintenance_ms\": "
          << snap.update_onlinepq_maintenance_ms << ",\n";
      ofs << "      \"update_delete_ms\": " << snap.update_delete_ms << ",\n";
      ofs << "      \"update_codebook_update_ms\": "
          << snap.update_codebook_update_ms << ",\n";
      ofs << "      \"update_reencode_ms\": " << snap.update_reencode_ms << ",\n";
      ofs << "      \"merge_compute_ms\": " << snap.merge_compute_ms << ",\n";
      ofs << "      \"global_rebuild_ms\": " << snap.global_rebuild_ms << ",\n";
      ofs << "      \"snapshot_total_ms\": " << snap.snapshot_total_ms << ",\n";
      ofs << "      \"throughput\": " << snap.update_throughput_vecps << ",\n";
      ofs << "      \"amortized_update_throughput_vecps\": "
          << snap.amortized_update_throughput_vecps << "\n";
      ofs << "    }";
      if (i + 1 < snapshots.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";

  } else {
  ofs << "{\n";
  ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
  ofs << "  \"timestamp\": " << ts << ",\n";
  ofs << "  \"params\": {\n";
  ofs << "    \"topk\": " << config.topk << ",\n";
  ofs << "    \"nprobe\": " << params.nprobe << ",\n";
  ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
  ofs << "    \"omp_max_threads\": " << RuntimeMaxThreads() << ",\n";
  ofs << "    \"pq_codebook_dimension_major\": "
      << (config.pq_codebook_dimension_major ? "true" : "false") << ",\n";
  ofs << "    \"pq_codes_subquantizer_major\": "
      << (config.pq_codes_subquantizer_major ? "true" : "false") << ",\n";
  ofs << "    \"use_whitening\": " << (config.use_whitening ? "true" : "false") << ",\n";
  ofs << "    \"use_cosine\": " << (config.use_cosine ? "true" : "false") << ",\n";
  ofs << "    \"enable_streaming\": " << (config.enable_streaming ? "true" : "false") << ",\n";
  ofs << "    \"main_index_rows_initial\": " << main_rows_initial << ",\n";
  ofs << "    \"main_rows_final\": " << main_rows_current << ",\n";
  ofs << "    \"rows_after_main\": " << rows_after_main << ",\n";
  ofs << "    \"delta_train_window\": " << config.delta_train_window << ",\n";
  ofs << "    \"delta_train_rows\": " << delta_train_rows << ",\n";
  ofs << "    \"delta_ivf_nlist\": " << config.delta_ivf_nlist << ",\n";
  ofs << "    \"delta_ivf_nlist_resolved\": " << delta_ivf_nlist << ",\n";
  ofs << "    \"merge_score_alpha\": " << config.merge_score_alpha << ",\n";
  ofs << "    \"merge_score_beta\": " << config.merge_score_beta << ",\n";
  ofs << "    \"merge_score_threshold\": " << config.merge_score_threshold << ",\n";
  ofs << "    \"merge_trigger_mode\": \"" << config.merge_trigger_mode << "\",\n";
  ofs << "    \"enable_merge\": " << (config.enable_merge ? "true" : "false") << ",\n";
  ofs << "    \"merge_trigger_rows\": " << config.merge_trigger_rows << ",\n";
  ofs << "    \"merge_trigger_rows_resolved\": " << merge_trigger_rows << ",\n";
  ofs << "    \"merge_trigger_qe_ratio\": " << config.merge_trigger_qe_ratio << ",\n";
  ofs << "    \"merge_trigger_drift\": " << config.merge_trigger_drift << ",\n";
  ofs << "    \"merge_trigger_delta_main_ratio\": " << config.merge_trigger_delta_main_ratio
      << ",\n";
  ofs << "    \"merge_trigger_imbalance_ratio\": " << config.merge_trigger_imbalance_ratio
      << ",\n";
  ofs << "    \"merge_assignment_mode\": \"" << config.merge_assignment_mode << "\",\n";
  ofs << "    \"merge_assignment_top_r\": " << config.merge_assignment_top_r << ",\n";
  ofs << "    \"merge_assignment_gamma\": " << config.merge_assignment_gamma << ",\n";
  ofs << "    \"merge_assignment_hard_cap_ratio\": " << config.merge_assignment_hard_cap_ratio
      << ",\n";
  ofs << "    \"merge_assignment_lambda\": " << config.merge_assignment_lambda << ",\n";
  ofs << "    \"enable_global_rebuild\": "
      << (config.enable_global_rebuild ? "true" : "false") << ",\n";
  ofs << "    \"global_rebuild_max_count\": " << config.global_rebuild_max_count << ",\n";
  ofs << "    \"global_rebuild_main_imbalance_ratio\": "
      << config.global_rebuild_main_imbalance_ratio << ",\n";
  ofs << "    \"global_rebuild_force_main_rows\": " << config.global_rebuild_force_main_rows
      << ",\n";
  ofs << "    \"global_rebuild_cooldown_rows\": " << config.global_rebuild_cooldown_rows
      << ",\n";
  ofs << "    \"delta_kmeans_iterations\": " << kDeltaKMeansIterationsDefault << ",\n";
  ofs << "    \"stream_start_row\": " << stream_start_idx << ",\n";
  ofs << "    \"stream_rows\": " << total_stream_rows << ",\n";
  ofs << "    \"streaming_mode\": \""
      << (config.enable_streaming ? config.streaming_mode : "offline") << "\",\n";
  ofs << "    \"stream_batch_size\": " << config.stream_batch_size << ",\n";
  ofs << "    \"streaming_use_stream_batch_size\": "
      << (config.streaming_use_stream_batch_size ? "true" : "false") << ",\n";
  ofs << "    \"enable_dynamic_ground_truth\": "
      << (config.enable_dynamic_ground_truth ? "true" : "false") << ",\n";
  ofs << "    \"skip_query_ground_truth\": "
      << (config.skip_query_ground_truth ? "true" : "false") << ",\n";
  ofs << "    \"enable_miss_diag\": " << (config.enable_miss_diag ? "true" : "false") << ",\n";
  ofs << "    \"enable_rerank_source_diag\": "
      << (config.enable_rerank_source_diag ? "true" : "false") << ",\n";
  ofs << "    \"enable_latency_debug\": "
      << (config.enable_latency_debug ? "true" : "false") << ",\n";
  ofs << "    \"exact_rerank_enable\": " << (config.exact_rerank_enable ? "true" : "false")
      << ",\n";
  ofs << "    \"exact_rerank_candidates_per_route\": "
      << config.exact_rerank_candidates_per_route << ",\n";
  ofs << "    \"main_exact_rerank_candidates\": " << config.main_exact_rerank_candidates
      << ",\n";
  ofs << "    \"active_exact_rerank_candidates\": " << config.active_exact_rerank_candidates
      << ",\n";
  ofs << "    \"frozen_exact_rerank_candidates\": " << config.frozen_exact_rerank_candidates
      << ",\n";
  ofs << "    \"snapshot_interval\": " << config.snapshot_interval << ",\n";
  ofs << "    \"snapshot_span\": " << snapshot_span << ",\n";
  ofs << "    \"online_pq_enable\": " << (config.online_pq_enable ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_update_scheme\": \"" << config.online_pq_update_scheme << "\",\n";
  ofs << "    \"online_pq_sliding_window_size\": " << config.online_pq_sliding_window_size << ",\n";
  ofs << "    \"online_pq_sliding_window_use_batches\": "
      << (config.online_pq_sliding_window_use_batches ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_sliding_window_rows\": " << sliding_window_rows << ",\n";
  ofs << "    \"online_pq_qe_ratio_threshold\": " << config.online_pq_qe_ratio_threshold << ",\n";
  ofs << "    \"online_pq_ema_alpha\": " << config.online_pq_ema_alpha << ",\n";
  ofs << "    \"online_pq_eps\": " << config.online_pq_eps << ",\n";
  ofs << "    \"online_pq_warmup_enable\": "
      << (config.online_pq_warmup_enable ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_warmup_batches\": " << config.online_pq_warmup_batches << ",\n";
  ofs << "    \"online_pq_force_update_interval\": "
      << config.online_pq_force_update_interval << ",\n";
  ofs << "    \"online_pq_partial_top_alpha\": "
      << (config.online_pq_partial_top_alpha ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_alpha\": " << config.online_pq_alpha << ",\n";
  ofs << "    \"online_pq_partial_top_lambda\": "
      << (config.online_pq_partial_top_lambda ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_lambda\": " << config.online_pq_lambda << ",\n";
  ofs << "    \"online_pq_reencode_batch\": "
      << (config.online_pq_reencode_batch ? "true" : "false") << "\n";
  ofs << "  },\n";
  ofs << "  \"metrics\": {\n";
  ofs << "    \"recall_available\": "
      << (final_metrics.recall_available ? "true" : "false") << ",\n";
  ofs << "    \"avg_query_ms\": " << final_metrics.avg_query_ms << ",\n";
  ofs << "    \"end_to_end_overhead_ms\": " << final_metrics.end_to_end_overhead_ms << ",\n";
  ofs << "    \"query_qps\": " << final_metrics.query_qps << ",\n";
  ofs << "    \"avg_pq_lut_build_us\": " << final_metrics.avg_pq_lut_build_us << ",\n";
  ofs << "    \"avg_pq_adc_scan_us\": " << final_metrics.avg_pq_adc_scan_us << ",\n";
  ofs << "    \"recall@" << config.topk << "\": " << final_metrics.recall << ",\n";
  ofs << "    \"full_run_wall_ms\": " << full_run_wall_ms << ",\n";
  ofs << "    \"recall_new\": " << final_metrics.recall_new << ",\n";
  ofs << "    \"recall_old\": " << final_metrics.recall_old << ",\n";
  ofs << "    \"gt_new_ratio\": " << final_metrics.gt_new_ratio << ",\n";
  ofs << "    \"gt_new_total\": " << final_metrics.gt_new_total << ",\n";
  ofs << "    \"gt_old_total\": " << final_metrics.gt_old_total << ",\n";
  ofs << "    \"hit_new_total\": " << final_metrics.hit_new_total << ",\n";
  ofs << "    \"hit_old_total\": " << final_metrics.hit_old_total << ",\n";
  ofs << "    \"build_rebuild_ms\": " << rebuild_ms_total << ",\n";
  ofs << "    \"initial_used_prebuilt_index\": "
      << (used_prebuilt_index ? "true" : "false") << ",\n";
  ofs << "    \"initial_index_load_ms\": " << init_load_ms << ",\n";
  ofs << "    \"initial_build_aligned_ms\": " << init_rebuild_ms << ",\n";
  ofs << "    \"initial_whitening_ms\": " << init_whitening_ms << ",\n";
  ofs << "    \"initial_whitening_transform_ms\": " << init_whitening_transform_ms << ",\n";
  ofs << "    \"initial_main_build_ms\": " << init_main_build_ms << ",\n";
  ofs << "    \"initial_main_add_ms\": " << init_main_add_ms << ",\n";
  ofs << "    \"initial_main_pq_encode_us\": "
      << initial_main_ingest_profile.encode_us << ",\n";
  ofs << "    \"initial_delta_seed_pq_encode_us\": "
      << initial_delta_seed_ingest_profile.encode_us << ",\n";
  ofs << "    \"initial_build_profile_available\": "
      << (initial_build_profile.has_value() ? "true" : "false") << ",\n";
  ofs << "    \"initial_build_coarse_kmeans_us\": "
      << (initial_build_profile ? initial_build_profile->build_coarse_kmeans_us : 0.0) << ",\n";
  ofs << "    \"initial_build_coarse_assignment_us\": "
      << (initial_build_profile ? initial_build_profile->build_coarse_kmeans_assignment_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_coarse_update_us\": "
      << (initial_build_profile ? initial_build_profile->build_coarse_kmeans_update_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_pq_training_us\": "
      << (initial_build_profile ? initial_build_profile->build_pq_training_total_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_pq_codebook_soa_us\": "
      << (initial_build_profile ? initial_build_profile->build_pq_codebook_soa_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_pq_precomputed_table_us\": "
      << (initial_build_profile ? initial_build_profile->build_pq_precomputed_table_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_pq_assignment_us\": "
      << (initial_build_profile ? initial_build_profile->build_pq_kmeans_assignment_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_pq_update_us\": "
      << (initial_build_profile ? initial_build_profile->build_pq_kmeans_update_us : 0.0)
      << ",\n";
  ofs << "    \"initial_build_publication_us\": "
      << (initial_build_profile ? initial_build_profile->build_publication_us : 0.0) << ",\n";
  ofs << "    \"initial_build_wall_ms\": " << init_total_wall_ms << ",\n";
  ofs << "    \"global_rebuild_count\": " << global_rebuild_count << ",\n";
  ofs << "    \"update_total_ms\": " << final_metrics.update_total_ms << ",\n";
  ofs << "    \"update_apply_ms\": " << total_update_ms << ",\n";
  ofs << "    \"update_whitening_ms\": " << total_update_whitening_ms << ",\n";
  ofs << "    \"update_insert_ms\": " << total_update_insert_ms << ",\n";
  ofs << "    \"update_record_build_ms\": " << total_update_record_build_ms << ",\n";
  ofs << "    \"update_insert_encode_ms\": " << total_update_insert_encode_ms << ",\n";
  ofs << "    \"update_insert_commit_ms\": " << total_update_insert_commit_ms << ",\n";
  ofs << "    \"update_onlinepq_maintenance_ms\": " << total_update_onlinepq_maintenance_ms << ",\n";
  ofs << "    \"update_delete_ms\": " << total_update_delete_ms << ",\n";
  ofs << "    \"update_codebook_update_ms\": " << total_update_codebook_update_ms << ",\n";
  ofs << "    \"update_reencode_ms\": " << total_update_reencode_ms << ",\n";
  ofs << "    \"delta_ingest_assignment_us\": "
      << total_delta_ingest_assignment_us << ",\n";
  ofs << "    \"delta_ingest_assignment_records\": "
      << total_delta_ingest_assignment_records << ",\n";
  ofs << "    \"update_merge_compute_ms\": " << total_merge_compute_ms << ",\n";
  ofs << "    \"update_global_rebuild_ms\": " << total_global_rebuild_ms << ",\n";
  ofs << "    \"update_per_vector_ms\": " << final_metrics.update_per_vector_ms << ",\n";
  ofs << "    \"update_throughput_vecps\": " << final_metrics.update_throughput_vecps << ",\n";
  ofs << "    \"scanned_avg\": " << final_metrics.scanned_avg << ",\n";
  ofs << "    \"gt_probed_rate\": " << final_metrics.gt_probed_rate << ",\n";
  ofs << "    \"recall_on_probed_gt\": " << final_metrics.recall_on_probed_gt << ",\n";
  ofs << "    \"exact_recall_on_probed_candidates\": "
      << final_metrics.exact_recall_on_probed_candidates << ",\n";
  ofs << "    \"avg_pq_rank_loss\": " << final_metrics.avg_pq_rank_loss << ",\n";
  ofs << "    \"miss_not_probed\": " << final_metrics.miss_not_probed << ",\n";
  ofs << "    \"miss_probed_filtered_by_pq\": " << final_metrics.miss_probed_filtered_by_pq << ",\n";
  ofs << "    \"pq_rank_loss_count\": " << final_metrics.pq_rank_loss_count << ",\n";
  ofs << "    \"rerank_topk_main_total\": " << final_metrics.rerank_topk_main_total << ",\n";
  ofs << "    \"rerank_topk_delta_total\": " << final_metrics.rerank_topk_delta_total << ",\n";
  ofs << "    \"rerank_topk_main_ratio\": " << final_metrics.rerank_topk_main_ratio << ",\n";
  ofs << "    \"rerank_topk_delta_ratio\": " << final_metrics.rerank_topk_delta_ratio << ",\n";
  ofs << "    \"rerank_topk_main_avg\": " << final_metrics.rerank_topk_main_avg << ",\n";
  ofs << "    \"rerank_topk_delta_avg\": " << final_metrics.rerank_topk_delta_avg << "\n";
  ofs << "  },\n";
  ofs << "  \"summary\": {\n";
  ofs << "    \"recall_avg\": " << recall_summary.avg << ",\n";
  ofs << "    \"recall_p5\": " << recall_summary.p5 << ",\n";
  ofs << "    \"recall_min\": " << recall_summary.min << ",\n";
  ofs << "    \"recall_final\": " << final_metrics.recall << ",\n";
  ofs << "    \"recall_new\": " << summary_recall_new << ",\n";
  ofs << "    \"recall_old\": " << summary_recall_old << ",\n";
  ofs << "    \"gt_new_ratio\": " << summary_gt_new_ratio << ",\n";
  ofs << "    \"gt_new_total\": " << summary_gt_new_total << ",\n";
  ofs << "    \"gt_old_total\": " << summary_gt_old_total << ",\n";
  ofs << "    \"hit_new_total\": " << summary_hit_new_total << ",\n";
  ofs << "    \"hit_old_total\": " << summary_hit_old_total << ",\n";
  ofs << "    \"qps_avg\": " << qps_summary.avg << ",\n";
  ofs << "    \"qps_p5\": " << qps_summary.p5 << ",\n";
  ofs << "    \"latency_avg_ms\": " << latency_summary.avg << ",\n";
  ofs << "    \"latency_p95_ms\": " << latency_summary.p95 << ",\n";
  ofs << "    \"latency_p99_ms\": " << latency_summary.p99 << ",\n";
  ofs << "    \"e2e_latency_avg_ms\": " << e2e_latency_summary.avg << ",\n";
  ofs << "    \"total_maintenance_ms\": " << total_maintenance_ms << ",\n";
  ofs << "    \"update_throughput_avg_vecps\": "
      << update_throughput_summary.avg << ",\n";
  ofs << "    \"amortized_update_throughput_avg_vecps\": "
      << amortized_update_throughput_summary.avg << ",\n";
  ofs << "    \"merge_count\": " << merge_events.size() << ",\n";
  ofs << "    \"global_rebuild_count\": " << global_rebuild_count << ",\n";
  ofs << "    \"merge_nodes\": [";
  for (size_t ni = 0; ni < merge_events.size(); ++ni) {
    ofs << merge_events[ni].base_rows;
    if (ni + 1 < merge_events.size()) ofs << ", ";
  }
  ofs << "],\n";
  ofs << "    \"global_rebuild_nodes\": [";
  for (size_t ni = 0; ni < global_rebuild_events.size(); ++ni) {
    ofs << global_rebuild_events[ni].base_rows;
    if (ni + 1 < global_rebuild_events.size()) ofs << ", ";
  }
  ofs << "],\n";
  ofs << "    \"process_rss_bytes\": " << ReadProcStatusBytes("VmRSS:") << ",\n";
  ofs << "    \"process_peak_rss_bytes\": " << ReadProcStatusBytes("VmHWM:") << "\n";
  ofs << "  },\n";
  ofs << "  \"route_execution\": {\n";
  ofs << "    \"main_queries\": " << final_metrics.main_route_queries << ",\n";
  ofs << "    \"frozen_delta_queries\": " << final_metrics.frozen_delta_route_queries << ",\n";
  ofs << "    \"active_delta_queries\": " << final_metrics.active_delta_route_queries << "\n";
  ofs << "  },\n";
  ofs << "  \"pre_stream_metrics\": ";
  if (pre_stream_metrics.has_value()) {
    ofs << "{\n";
    ofs << "    \"base_rows\": " << stream_start_idx << ",\n";
    ofs << "    \"recall\": " << pre_stream_metrics->recall << ",\n";
    ofs << "    \"avg_query_ms\": " << pre_stream_metrics->avg_query_ms << ",\n";
    ofs << "    \"end_to_end_overhead_ms\": " << pre_stream_metrics->end_to_end_overhead_ms << ",\n";
    ofs << "    \"avg_search_ms\": " << pre_stream_metrics->avg_search_ms << ",\n";
    ofs << "    \"query_qps\": " << pre_stream_metrics->query_qps << ",\n";
    ofs << "    \"avg_scanned\": " << pre_stream_metrics->scanned_avg << ",\n";
    ofs << "    \"query_eval_ms\": " << pre_stream_metrics->query_eval_ms << ",\n";
    ofs << "    \"gt_probed_rate\": " << pre_stream_metrics->gt_probed_rate << ",\n";
    ofs << "    \"recall_on_probed_gt\": " << pre_stream_metrics->recall_on_probed_gt << ",\n";
    ofs << "    \"exact_recall_on_probed_candidates\": "
        << pre_stream_metrics->exact_recall_on_probed_candidates << ",\n";
    ofs << "    \"avg_pq_rank_loss\": " << pre_stream_metrics->avg_pq_rank_loss << ",\n";
    ofs << "    \"miss_not_probed\": " << pre_stream_metrics->miss_not_probed << ",\n";
    ofs << "    \"miss_probed_filtered_by_pq\": "
        << pre_stream_metrics->miss_probed_filtered_by_pq << ",\n";
    ofs << "    \"pq_rank_loss_count\": " << pre_stream_metrics->pq_rank_loss_count << ",\n";
    ofs << "    \"rerank_topk_main_total\": " << pre_stream_metrics->rerank_topk_main_total
        << ",\n";
    ofs << "    \"rerank_topk_delta_total\": " << pre_stream_metrics->rerank_topk_delta_total
        << ",\n";
    ofs << "    \"rerank_topk_main_ratio\": " << pre_stream_metrics->rerank_topk_main_ratio
        << ",\n";
    ofs << "    \"rerank_topk_delta_ratio\": " << pre_stream_metrics->rerank_topk_delta_ratio
        << ",\n";
    ofs << "    \"rerank_topk_main_avg\": " << pre_stream_metrics->rerank_topk_main_avg << ",\n";
    ofs << "    \"rerank_topk_delta_avg\": " << pre_stream_metrics->rerank_topk_delta_avg
        << ",\n";
    ofs << "    \"worst_queries\": [\n";
    for (size_t wi = 0; wi < pre_stream_metrics->worst_queries.size(); ++wi) {
      ofs << "      \"" << pre_stream_metrics->worst_queries[wi] << "\"";
      if (wi + 1 < pre_stream_metrics->worst_queries.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "    ]\n";
    ofs << "  },\n";
  } else {
    ofs << "null,\n";
  }
  ofs << "  \"online_pq\": {\n";
  ofs << "    \"enabled\": " << (online_pq_enabled ? "true" : "false") << ",\n";
  ofs << "    \"batches\": " << online_pq_rollup.batches << ",\n";
  ofs << "    \"warmup_batches\": " << online_pq_rollup.warmup_batches << ",\n";
  ofs << "    \"triggered\": " << online_pq_rollup.triggered << ",\n";
  ofs << "    \"updated\": " << online_pq_rollup.updated << ",\n";
  ofs << "    \"reencoded\": " << online_pq_rollup.reencoded << ",\n";
  ofs << "    \"avg_nqe\": " << online_avg_nqe << ",\n";
  ofs << "    \"last_nqe\": " << online_pq_rollup.last_nqe_batch << ",\n";
  ofs << "    \"last_warmup_batches_left\": " << online_pq_rollup.last_warmup_batches_left << ",\n";
  ofs << "    \"avg_qe_ratio\": " << online_avg_qe_ratio << ",\n";
  ofs << "    \"last_qe_ratio\": " << online_pq_rollup.last_qe_ratio << ",\n";
  ofs << "    \"avg_codebook_drift\": " << online_avg_drift << ",\n";
  ofs << "    \"last_codebook_drift\": " << online_pq_rollup.last_codebook_drift << ",\n";
  ofs << "    \"updated_subspaces\": " << online_pq_rollup.updated_subspaces << ",\n";
  ofs << "    \"updated_codewords\": " << online_pq_rollup.updated_codewords << "\n";
  ofs << "  },\n";
  ofs << "  \"merge_events\": [\n";
  for (size_t i = 0; i < merge_events.size(); ++i) {
    const auto& ev = merge_events[i];
    ofs << "    {\n";
    ofs << "      \"base_rows\": " << ev.base_rows << ",\n";
    ofs << "      \"frozen_rows\": " << ev.frozen_rows << ",\n";
    ofs << "      \"patched_partitions\": " << ev.patched_partitions << ",\n";
    ofs << "      \"append_parts\": " << ev.append_partitions << ",\n";
    ofs << "      \"recluster_parts\": " << ev.recluster_partitions << ",\n";
    ofs << "      \"moved_delta_ratio\": " << ev.moved_delta_ratio << ",\n";
    ofs << "      \"avg_assignment_dist_ratio\": " << ev.avg_assignment_dist_ratio << ",\n";
    ofs << "      \"max_assignment_dist_ratio\": " << ev.max_assignment_dist_ratio << ",\n";
    ofs << "      \"imbalance_before\": " << ev.imbalance_before << ",\n";
    ofs << "      \"imbalance_after\": " << ev.imbalance_after << ",\n";
    ofs << "      \"main_non_empty_lists_after\": " << ev.main_non_empty_lists_after << ",\n";
    ofs << "      \"main_max_list_after\": " << ev.main_max_list_after << ",\n";
    ofs << "      \"main_avg_non_empty_list_after\": " << ev.main_avg_non_empty_list_after
        << ",\n";
    ofs << "      \"main_imbalance_after_real\": " << ev.main_imbalance_after_real << ",\n";
    ofs << "      \"trigger_mode\": \"" << ev.trigger_mode << "\",\n";
    ofs << "      \"trigger_reason\": \"" << ev.trigger_reason << "\",\n";
    ofs << "      \"trigger_rows\": " << (ev.trigger_rows ? "true" : "false") << ",\n";
    ofs << "      \"trigger_structure\": " << (ev.trigger_structure ? "true" : "false") << ",\n";
    ofs << "      \"trigger_qe_ratio\": " << (ev.trigger_qe_ratio ? "true" : "false") << ",\n";
    ofs << "      \"trigger_drift\": " << (ev.trigger_drift ? "true" : "false") << ",\n";
    ofs << "      \"trigger_delta_main_ratio\": "
        << (ev.trigger_delta_main_ratio ? "true" : "false") << ",\n";
    ofs << "      \"trigger_imbalance\": " << (ev.trigger_imbalance ? "true" : "false")
        << ",\n";
    ofs << "      \"trigger_qe_ratio_value\": " << ev.trigger_qe_ratio_value << ",\n";
    ofs << "      \"trigger_drift_value\": " << ev.trigger_drift_value << ",\n";
    ofs << "      \"trigger_delta_main_ratio_value\": " << ev.trigger_delta_main_ratio_value
        << ",\n";
    ofs << "      \"trigger_imbalance_value\": " << ev.trigger_imbalance_value << ",\n";
    ofs << "      \"trigger_active_rows\": " << ev.trigger_active_rows << ",\n";
    ofs << "      \"trigger_active_nlist\": " << ev.trigger_active_nlist << ",\n";
    ofs << "      \"trigger_active_non_empty_lists\": " << ev.trigger_active_non_empty_lists
        << ",\n";
    ofs << "      \"trigger_active_max_list\": " << ev.trigger_active_max_list << ",\n";
    ofs << "      \"trigger_active_avg_non_empty_list\": "
        << ev.trigger_active_avg_non_empty_list << ",\n";
    ofs << "      \"merge_compute_ms\": " << ev.merge_compute_ms << ",\n";
    ofs << "      \"codebook_rebuild_ms\": " << ev.codebook_rebuild_ms << ",\n";
    ofs << "      \"assignment_distance_us\": "
        << ev.profiling.merge_assignment_distance_us << ",\n";
    ofs << "      \"assignment_top_r_us\": "
        << ev.profiling.merge_assignment_top_r_us << ",\n";
    ofs << "      \"assignment_balance_us\": "
        << ev.profiling.merge_assignment_balance_us << ",\n";
    ofs << "      \"fetch_main_records_us\": "
        << ev.profiling.fetch_main_records_us << ",\n";
    ofs << "      \"repartition_distance_us\": "
        << ev.profiling.repartition_distance_us << ",\n";
    ofs << "      \"repartition_sort_us\": " << ev.profiling.repartition_sort_us << ",\n";
    ofs << "      \"prepare_pq_encode_us\": "
        << ev.profiling.prepare_pq_encode_us << ",\n";
    ofs << "      \"commit_us\": " << ev.profiling.commit_us << ",\n";
    ofs << "      \"pq_code_assignment_us\": "
        << ev.profiling.pq_code_assignment_us << ",\n";
    ofs << "      \"pq_code_copy_or_reuse_us\": "
        << ev.profiling.pq_code_copy_or_reuse_us << ",\n";
    ofs << "      \"pq_list_flatten_us\": " << ev.profiling.pq_list_flatten_us << ",\n";
    ofs << "      \"merge_ms\": " << ev.merge_ms << "\n";
    ofs << "    }";
    if (i + 1 < merge_events.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "  ],\n";
  ofs << "  \"global_rebuild_events\": [\n";
  for (size_t i = 0; i < global_rebuild_events.size(); ++i) {
    const auto& ev = global_rebuild_events[i];
    ofs << "    {\n";
    ofs << "      \"base_rows\": " << ev.base_rows << ",\n";
    ofs << "      \"old_main_rows\": " << ev.old_main_rows << ",\n";
    ofs << "      \"new_main_rows\": " << ev.new_main_rows << ",\n";
    ofs << "      \"active_seed_rows\": " << ev.active_seed_rows << ",\n";
    ofs << "      \"rebuild_count\": " << ev.rebuild_count << ",\n";
    ofs << "      \"max_count\": " << ev.max_count << ",\n";
    ofs << "      \"main_nlist\": " << ev.main_nlist << ",\n";
    ofs << "      \"main_non_empty_lists\": " << ev.main_non_empty_lists << ",\n";
    ofs << "      \"main_max_list_size\": " << ev.main_max_list_size << ",\n";
    ofs << "      \"main_avg_non_empty_list_size\": " << ev.main_avg_non_empty_list_size
        << ",\n";
    ofs << "      \"trigger_main_rows\": " << (ev.trigger_main_rows ? "true" : "false")
        << ",\n";
    ofs << "      \"trigger_imbalance\": " << (ev.trigger_imbalance ? "true" : "false")
        << ",\n";
    ofs << "      \"trigger_main_rows_since_last_rebuild\": "
        << ev.trigger_main_rows_since_last_rebuild << ",\n";
    ofs << "      \"trigger_main_rows_threshold\": " << ev.trigger_main_rows_threshold << ",\n";
    ofs << "      \"trigger_imbalance_threshold\": " << ev.trigger_imbalance_threshold
        << ",\n";
    ofs << "      \"threshold\": " << ev.threshold << ",\n";
    ofs << "      \"main_imbalance_ratio\": " << ev.main_imbalance_ratio << ",\n";
    ofs << "      \"reason\": \"" << ev.reason << "\",\n";
    ofs << "      \"whitening_ms\": " << ev.whitening_ms << ",\n";
    ofs << "      \"whitening_transform_ms\": " << ev.whitening_transform_ms << ",\n";
    ofs << "      \"main_build_ms\": " << ev.main_build_ms << ",\n";
    ofs << "      \"main_add_ms\": " << ev.main_add_ms << ",\n";
    ofs << "      \"delta_seed_ms\": " << ev.delta_seed_ms << ",\n";
    ofs << "      \"total_ms\": " << ev.total_ms << ",\n";
    ofs << "      \"wall_total_ms\": " << ev.wall_total_ms << "\n";
    ofs << "    }";
    if (i + 1 < global_rebuild_events.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "  ],\n";
  ofs << "  \"snapshots\": [\n";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto& snap = snapshots[i];
    ofs << "    {\n";
    ofs << "      \"base_rows\": " << snap.base_rows << ",\n";
    ofs << "      \"main_rows\": " << snap.main_rows << ",\n";
    ofs << "      \"frozen_delta_docs\": " << snap.frozen_delta_docs << ",\n";
    ofs << "      \"delta_rows\": " << snap.delta_rows << ",\n";
    ofs << "      \"active_delta_docs\": " << snap.active_delta_docs << ",\n";
    ofs << "      \"snapshot_rows\": " << snap.snapshot_rows << ",\n";
    ofs << "      \"recall\": " << snap.recall << ",\n";
    ofs << "      \"recall_new\": " << snap.recall_new << ",\n";
    ofs << "      \"recall_old\": " << snap.recall_old << ",\n";
    ofs << "      \"gt_new_ratio\": " << snap.gt_new_ratio << ",\n";
    ofs << "      \"gt_new_total\": " << snap.gt_new_total << ",\n";
    ofs << "      \"gt_old_total\": " << snap.gt_old_total << ",\n";
    ofs << "      \"hit_new_total\": " << snap.hit_new_total << ",\n";
    ofs << "      \"hit_old_total\": " << snap.hit_old_total << ",\n";
    ofs << "      \"latency_ms\": " << snap.latency_ms << ",\n";
    ofs << "      \"end_to_end_overhead_ms\": " << snap.end_to_end_overhead_ms << ",\n";
    ofs << "      \"avg_search_ms\": " << snap.avg_search_ms << ",\n";
    ofs << "      \"avg_scanned\": " << snap.avg_scanned << ",\n";
    ofs << "      \"query_qps\": " << snap.query_qps << ",\n";
    ofs << "      \"update_ms\": " << snap.update_ms << ",\n";
    ofs << "      \"update_whitening_ms\": " << snap.update_whitening_ms << ",\n";
    ofs << "      \"update_insert_ms\": " << snap.update_insert_ms << ",\n";
    ofs << "      \"update_record_build_ms\": " << snap.update_record_build_ms << ",\n";
    ofs << "      \"update_insert_encode_ms\": " << snap.update_insert_encode_ms << ",\n";
    ofs << "      \"update_insert_commit_ms\": " << snap.update_insert_commit_ms << ",\n";
    ofs << "      \"update_onlinepq_maintenance_ms\": "
        << snap.update_onlinepq_maintenance_ms << ",\n";
    ofs << "      \"update_delete_ms\": " << snap.update_delete_ms << ",\n";
    ofs << "      \"update_codebook_update_ms\": "
        << snap.update_codebook_update_ms << ",\n";
    ofs << "      \"update_reencode_ms\": " << snap.update_reencode_ms << ",\n";
    ofs << "      \"merge_compute_ms\": " << snap.merge_compute_ms << ",\n";
    ofs << "      \"global_rebuild_ms\": " << snap.global_rebuild_ms << ",\n";
    ofs << "      \"query_eval_ms\": " << snap.query_eval_ms << ",\n";
    ofs << "      \"snapshot_total_ms\": " << snap.snapshot_total_ms << ",\n";
    ofs << "      \"update_throughput_vecps\": " << snap.update_throughput_vecps << ",\n";
    ofs << "      \"amortized_update_throughput_vecps\": "
        << snap.amortized_update_throughput_vecps << ",\n";
    ofs << "      \"nqe_batch\": " << snap.nqe_batch << ",\n";
    ofs << "      \"qe_ratio\": " << snap.qe_ratio << ",\n";
    ofs << "      \"codebook_drift\": " << snap.codebook_drift << ",\n";
    ofs << "      \"pq_updated\": " << (snap.pq_updated ? "true" : "false") << ",\n";
    ofs << "      \"in_warmup\": " << (snap.in_warmup ? "true" : "false") << ",\n";
    ofs << "      \"warmup_batches_left\": " << snap.warmup_batches_left << ",\n";
    ofs << "      \"gt_probed_rate\": " << snap.gt_probed_rate << ",\n";
    ofs << "      \"recall_on_probed_gt\": " << snap.recall_on_probed_gt << ",\n";
    ofs << "      \"exact_recall_on_probed_candidates\": "
        << snap.exact_recall_on_probed_candidates << ",\n";
    ofs << "      \"avg_pq_rank_loss\": " << snap.avg_pq_rank_loss << ",\n";
    ofs << "      \"miss_not_probed\": " << snap.miss_not_probed << ",\n";
    ofs << "      \"miss_probed_filtered_by_pq\": " << snap.miss_probed_filtered_by_pq << ",\n";
    ofs << "      \"pq_rank_loss_count\": " << snap.pq_rank_loss_count << ",\n";
    ofs << "      \"rerank_topk_main_total\": " << snap.rerank_topk_main_total << ",\n";
    ofs << "      \"rerank_topk_delta_total\": " << snap.rerank_topk_delta_total << ",\n";
    ofs << "      \"rerank_topk_main_ratio\": " << snap.rerank_topk_main_ratio << ",\n";
    ofs << "      \"rerank_topk_delta_ratio\": " << snap.rerank_topk_delta_ratio << ",\n";
    ofs << "      \"rerank_topk_main_avg\": " << snap.rerank_topk_main_avg << ",\n";
    ofs << "      \"rerank_topk_delta_avg\": " << snap.rerank_topk_delta_avg << ",\n";
    ofs << "      \"worst_queries\": [\n";
    for (size_t wi = 0; wi < snap.worst_queries.size(); ++wi) {
      ofs << "        \"" << snap.worst_queries[wi] << "\"";
      if (wi + 1 < snap.worst_queries.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "      ],\n";
    ofs << "      \"minibatches\": [\n";
    for (size_t j = 0; j < snap.minibatches.size(); ++j) {
      const auto& mb = snap.minibatches[j];
      ofs << "        {\n";
      ofs << "          \"batch_id\": " << mb.batch_id << ",\n";
      ofs << "          \"base_rows\": " << mb.base_rows << ",\n";
      ofs << "          \"stream_rows_total\": " << mb.stream_rows_total << ",\n";
      ofs << "          \"batch_rows\": " << mb.batch_rows << ",\n";
      ofs << "          \"snapshot_rows_total\": " << mb.snapshot_rows_total << ",\n";
      ofs << "          \"recall\": " << mb.recall << ",\n";
      ofs << "          \"recall_new\": " << mb.recall_new << ",\n";
      ofs << "          \"recall_old\": " << mb.recall_old << ",\n";
      ofs << "          \"gt_new_ratio\": " << mb.gt_new_ratio << ",\n";
      ofs << "          \"gt_new_total\": " << mb.gt_new_total << ",\n";
      ofs << "          \"gt_old_total\": " << mb.gt_old_total << ",\n";
      ofs << "          \"hit_new_total\": " << mb.hit_new_total << ",\n";
      ofs << "          \"hit_old_total\": " << mb.hit_old_total << ",\n";
      ofs << "          \"latency_ms\": " << mb.latency_ms << ",\n";
      ofs << "          \"end_to_end_overhead_ms\": " << mb.end_to_end_overhead_ms << ",\n";
      ofs << "          \"avg_search_ms\": " << mb.avg_search_ms << ",\n";
      ofs << "          \"avg_scanned\": " << mb.avg_scanned << ",\n";
      ofs << "          \"query_qps\": " << mb.query_qps << ",\n";
      ofs << "          \"update_ms\": " << mb.update_ms << ",\n";
      ofs << "          \"update_whitening_ms\": " << mb.update_whitening_ms << ",\n";
      ofs << "          \"update_insert_ms\": " << mb.update_insert_ms << ",\n";
      ofs << "          \"update_record_build_ms\": " << mb.update_record_build_ms << ",\n";
      ofs << "          \"update_insert_encode_ms\": " << mb.update_insert_encode_ms << ",\n";
      ofs << "          \"update_insert_commit_ms\": " << mb.update_insert_commit_ms << ",\n";
      ofs << "          \"update_onlinepq_maintenance_ms\": "
          << mb.update_onlinepq_maintenance_ms << ",\n";
      ofs << "          \"update_delete_ms\": " << mb.update_delete_ms << ",\n";
      ofs << "          \"update_codebook_update_ms\": "
          << mb.update_codebook_update_ms << ",\n";
      ofs << "          \"update_reencode_ms\": " << mb.update_reencode_ms << ",\n";
      ofs << "          \"update_throughput_vecps\": " << mb.update_throughput_vecps << ",\n";
      ofs << "          \"query_eval_ms\": " << mb.query_eval_ms << ",\n";
      ofs << "          \"nqe_batch\": " << mb.nqe_batch << ",\n";
      ofs << "          \"qe_ratio\": " << mb.qe_ratio << ",\n";
      ofs << "          \"codebook_drift\": " << mb.codebook_drift << ",\n";
      ofs << "          \"pq_updated\": " << (mb.pq_updated ? "true" : "false") << ",\n";
      ofs << "          \"in_warmup\": " << (mb.in_warmup ? "true" : "false") << ",\n";
      ofs << "          \"warmup_batches_left\": " << mb.warmup_batches_left << ",\n";
      ofs << "          \"gt_probed_rate\": " << mb.gt_probed_rate << ",\n";
      ofs << "          \"recall_on_probed_gt\": " << mb.recall_on_probed_gt << ",\n";
      ofs << "          \"exact_recall_on_probed_candidates\": "
          << mb.exact_recall_on_probed_candidates << ",\n";
      ofs << "          \"avg_pq_rank_loss\": " << mb.avg_pq_rank_loss << ",\n";
      ofs << "          \"miss_not_probed\": " << mb.miss_not_probed << ",\n";
      ofs << "          \"miss_probed_filtered_by_pq\": " << mb.miss_probed_filtered_by_pq << ",\n";
      ofs << "          \"pq_rank_loss_count\": " << mb.pq_rank_loss_count << "\n";
      ofs << "        }";
      if (j + 1 < snap.minibatches.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "      ]";
    if (snap.latency_debug.has_value()) {
      const auto& debug = snap.latency_debug.value();
      ofs << ",\n";
      ofs << "      \"latency_debug\": {\n";
      ofs << "        \"route_count\": " << debug.route_count << ",\n";
      ofs << "        \"active_window_ready\": " << (snap.active_window_ready ? "true" : "false")
          << ",\n";
      ofs << "        \"will_commit_merge\": " << (snap.will_commit_merge ? "true" : "false")
          << ",\n";
      ofs << "        \"merge_ms\": {\n";
      ofs << "          \"avg\": " << debug.merge_ms.avg << ",\n";
      ofs << "          \"p50\": " << debug.merge_ms.p50 << ",\n";
      ofs << "          \"p90\": " << debug.merge_ms.p90 << ",\n";
      ofs << "          \"p99\": " << debug.merge_ms.p99 << ",\n";
      ofs << "          \"max\": " << debug.merge_ms.max << "\n";
      ofs << "        },\n";
      ofs << "        \"max_route_search_ms\": {\n";
      ofs << "          \"avg\": " << debug.max_route_search_ms.avg << ",\n";
      ofs << "          \"p50\": " << debug.max_route_search_ms.p50 << ",\n";
      ofs << "          \"p90\": " << debug.max_route_search_ms.p90 << ",\n";
      ofs << "          \"p99\": " << debug.max_route_search_ms.p99 << ",\n";
      ofs << "          \"max\": " << debug.max_route_search_ms.max << "\n";
      ofs << "        },\n";
      ofs << "        \"routes\": [\n";
      for (size_t ri = 0; ri < debug.routes.size(); ++ri) {
        const auto& route = debug.routes[ri];
        ofs << "          {\n";
        ofs << "            \"name\": \"" << route.route_name << "\",\n";
        ofs << "            \"from_new\": " << static_cast<uint32_t>(route.from_new) << ",\n";
        ofs << "            \"search_ms\": {\n";
        ofs << "              \"avg\": " << route.search_ms.avg << ",\n";
        ofs << "              \"p50\": " << route.search_ms.p50 << ",\n";
        ofs << "              \"p90\": " << route.search_ms.p90 << ",\n";
        ofs << "              \"p99\": " << route.search_ms.p99 << ",\n";
        ofs << "              \"max\": " << route.search_ms.max << "\n";
        ofs << "            },\n";
        ofs << "            \"scanned_candidates\": {\n";
        ofs << "              \"avg\": " << route.scanned_candidates.avg << ",\n";
        ofs << "              \"p50\": " << route.scanned_candidates.p50 << ",\n";
        ofs << "              \"p90\": " << route.scanned_candidates.p90 << ",\n";
        ofs << "              \"p99\": " << route.scanned_candidates.p99 << ",\n";
        ofs << "              \"max\": " << route.scanned_candidates.max << "\n";
        ofs << "            }\n";
        ofs << "          }";
        if (ri + 1 < debug.routes.size()) {
          ofs << ",";
        }
        ofs << "\n";
      }
      ofs << "        ],\n";
      ofs << "        \"slow_queries\": [\n";
      for (size_t si = 0; si < debug.slow_queries.size(); ++si) {
        const auto& slow = debug.slow_queries[si];
        ofs << "          {\n";
        ofs << "            \"query_id\": " << slow.query_id << ",\n";
        ofs << "            \"total_search_ms\": " << slow.total_search_ms << ",\n";
        ofs << "            \"merge_ms\": " << slow.merge_ms << ",\n";
        ofs << "            \"max_route_search_ms\": " << slow.max_route_search_ms << ",\n";
        ofs << "            \"merged_scanned\": " << slow.merged_scanned << ",\n";
        ofs << "            \"route_search_ms\": [";
        for (size_t rsi = 0; rsi < slow.route_search_ms.size(); ++rsi) {
          ofs << slow.route_search_ms[rsi];
          if (rsi + 1 < slow.route_search_ms.size()) {
            ofs << ", ";
          }
        }
        ofs << "],\n";
        ofs << "            \"route_scanned_candidates\": [";
        for (size_t rsi = 0; rsi < slow.route_scanned_candidates.size(); ++rsi) {
          ofs << slow.route_scanned_candidates[rsi];
          if (rsi + 1 < slow.route_scanned_candidates.size()) {
            ofs << ", ";
          }
        }
        ofs << "]\n";
        ofs << "          }";
        if (si + 1 < debug.slow_queries.size()) {
          ofs << ",";
        }
        ofs << "\n";
      }
      ofs << "        ],\n";
      ofs << "        \"partition_debug\": [\n";
      for (size_t pi = 0; pi < snap.partition_debug.size(); ++pi) {
        const auto& part = snap.partition_debug[pi];
        ofs << "          {\n";
        ofs << "            \"route\": \"" << part.route_name << "\",\n";
        ofs << "            \"nlist\": " << part.nlist << ",\n";
        ofs << "            \"non_empty_lists\": " << part.non_empty_lists << ",\n";
        ofs << "            \"total_docs\": " << part.total_docs << ",\n";
        ofs << "            \"list_size\": {\n";
        ofs << "              \"avg\": " << part.list_size.avg << ",\n";
        ofs << "              \"p50\": " << part.list_size.p50 << ",\n";
        ofs << "              \"p90\": " << part.list_size.p90 << ",\n";
        ofs << "              \"p99\": " << part.list_size.p99 << ",\n";
        ofs << "              \"max\": " << part.list_size.max << "\n";
        ofs << "            }\n";
        ofs << "          }";
        if (pi + 1 < snap.partition_debug.size()) {
          ofs << ",";
        }
        ofs << "\n";
      }
      ofs << "        ]\n";
      ofs << "      }\n";
    } else {
      ofs << "\n";
    }
    ofs << "    }";
    if (i + 1 < snapshots.size()) {
      ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "  ]\n";
  ofs << "}\n";

  }
  return Status::OK();
}
}  // namespace ann::eval
