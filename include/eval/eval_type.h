#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/types.h"
#include "index/ivf.h"
#include "index/merge.h"

namespace ann::eval::run_eval {

struct DistributionStats {
  double avg{0.0};
  double p50{0.0};
  double p90{0.0};
  double p99{0.0};
  double max{0.0};
};

struct SeriesStats {
  double avg{0.0};
  double p5{0.0};
  double p50{0.0};
  double p95{0.0};
  double p99{0.0};
  double min{0.0};
  double max{0.0};
};

struct RouteDebugStats {
  std::string route_name;
  uint8_t from_new{0};
  DistributionStats search_ms;
  DistributionStats scanned_candidates;
};

struct SlowQueryDebug {
  uint32_t query_id{0};
  double total_search_ms{0.0};
  double merge_ms{0.0};
  double max_route_search_ms{0.0};
  double merged_scanned{0.0};
  std::vector<double> route_search_ms;
  std::vector<double> route_scanned_candidates;
};

struct LatencyDebugMetrics {
  uint32_t route_count{0};
  DistributionStats merge_ms;
  DistributionStats max_route_search_ms;
  std::vector<RouteDebugStats> routes;
  std::vector<SlowQueryDebug> slow_queries;
};

struct IndexPartitionDebug {
  std::string route_name;
  uint32_t nlist{0};
  uint32_t non_empty_lists{0};
  uint64_t total_docs{0};
  DistributionStats list_size;
};

struct RuntimeOptions {
  std::string config_path{"configs/sift/sift.json"};
  std::optional<std::string> dataset_spec;
  std::optional<std::string> query_spec;
  std::optional<std::string> prebuilt_index_dir;
  bool use_default_prebuilt_index{false};
  bool fresh_index{false};
};

struct DeltaShard {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint32_t rows{0};
  uint32_t shard_id{0};
};

struct EvalMetrics {
  bool recall_available{false};
  double recall{0.0};
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
  double whitening_p50{0.0};
  double whitening_p99{0.0};
  double avg_whiten_ms{0.0};
  double avg_search_ms{0.0};
  double avg_pq_lut_build_us{0.0};
  double avg_pq_adc_scan_us{0.0};
  double search_p50{0.0};
  double search_p99{0.0};
  double total_p50{0.0};
  double total_p99{0.0};
  double avg_query_ms{0.0};
  double end_to_end_overhead_ms{0.0};
  double query_qps{0.0};
  double rebuild_ms{0.0};
  double scanned_avg{0.0};
  double scanned_p50{0.0};
  double scanned_p99{0.0};
  double scanned_max{0.0};
  double update_total_ms{0.0};
  double update_per_vector_ms{0.0};
  double update_throughput_vecps{0.0};
  double query_eval_ms{0.0};
  uint32_t query_count{0};
  double gt_probed_rate{0.0};
  double recall_on_probed_gt{0.0};
  double exact_recall_on_probed_candidates{0.0};
  double avg_pq_rank_loss{0.0};
  uint32_t miss_not_probed{0};
  uint32_t miss_probed_filtered_by_pq{0};
  uint32_t pq_rank_loss_count{0};
  uint64_t rerank_topk_main_total{0};
  uint64_t rerank_topk_delta_total{0};
  double rerank_topk_main_ratio{0.0};
  double rerank_topk_delta_ratio{0.0};
  double rerank_topk_main_avg{0.0};
  double rerank_topk_delta_avg{0.0};
  uint32_t main_route_queries{0};
  uint32_t active_delta_route_queries{0};
  uint32_t frozen_delta_route_queries{0};
  std::vector<std::string> worst_queries;
  std::optional<LatencyDebugMetrics> latency_debug;
};

struct RecallAgeMetrics {
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
};

struct MinibatchRecord {
  uint32_t batch_id{0};
  uint32_t base_rows{0};
  uint32_t stream_rows_total{0};
  uint32_t batch_rows{0};
  uint32_t snapshot_rows_total{0};
  double recall{0.0};
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
  double latency_ms{0.0};
  double end_to_end_overhead_ms{0.0};
  double avg_search_ms{0.0};
  double avg_scanned{0.0};
  double query_qps{0.0};
  double update_ms{0.0};
  double update_whitening_ms{0.0};
  double update_insert_ms{0.0};
  double update_record_build_ms{0.0};
  double update_insert_encode_ms{0.0};
  double update_insert_commit_ms{0.0};
  double update_onlinepq_maintenance_ms{0.0};
  double update_delete_ms{0.0};
  double update_codebook_update_ms{0.0};
  double update_reencode_ms{0.0};
  double update_throughput_vecps{0.0};
  double query_eval_ms{0.0};
  double nqe_batch{0.0};
  double qe_ratio{1.0};
  double codebook_drift{0.0};
  bool pq_updated{false};
  bool in_warmup{false};
  uint32_t warmup_batches_left{0};
  double gt_probed_rate{0.0};
  double recall_on_probed_gt{0.0};
  double exact_recall_on_probed_candidates{0.0};
  double avg_pq_rank_loss{0.0};
  uint32_t miss_not_probed{0};
  uint32_t miss_probed_filtered_by_pq{0};
  uint32_t pq_rank_loss_count{0};
};

struct SnapshotRecord {
  uint32_t base_rows{0};
  uint32_t main_rows{0};
  uint32_t frozen_delta_docs{0};
  uint32_t delta_rows{0};
  uint32_t active_delta_docs{0};
  uint32_t snapshot_rows{0};
  double recall{0.0};
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
  double latency_ms{0.0};
  double end_to_end_overhead_ms{0.0};
  double avg_search_ms{0.0};
  double avg_scanned{0.0};
  double query_qps{0.0};
  double update_ms{0.0};
  double update_whitening_ms{0.0};
  double update_insert_ms{0.0};
  double update_record_build_ms{0.0};
  double update_insert_encode_ms{0.0};
  double update_insert_commit_ms{0.0};
  double update_onlinepq_maintenance_ms{0.0};
  double update_delete_ms{0.0};
  double update_codebook_update_ms{0.0};
  double update_reencode_ms{0.0};
  double query_eval_ms{0.0};
  double merge_compute_ms{0.0};
  double global_rebuild_ms{0.0};
  double snapshot_total_ms{0.0};
  double update_throughput_vecps{0.0};
  double amortized_update_throughput_vecps{0.0};
  double nqe_batch{0.0};
  double qe_ratio{1.0};
  double codebook_drift{0.0};
  bool pq_updated{false};
  bool in_warmup{false};
  uint32_t warmup_batches_left{0};
  double gt_probed_rate{0.0};
  double recall_on_probed_gt{0.0};
  double exact_recall_on_probed_candidates{0.0};
  double avg_pq_rank_loss{0.0};
  uint32_t miss_not_probed{0};
  uint32_t miss_probed_filtered_by_pq{0};
  uint32_t pq_rank_loss_count{0};
  uint64_t rerank_topk_main_total{0};
  uint64_t rerank_topk_delta_total{0};
  double rerank_topk_main_ratio{0.0};
  double rerank_topk_delta_ratio{0.0};
  double rerank_topk_main_avg{0.0};
  double rerank_topk_delta_avg{0.0};
  std::vector<std::string> worst_queries;
  std::vector<MinibatchRecord> minibatches;
  bool active_window_ready{false};
  bool will_commit_merge{false};
  std::optional<LatencyDebugMetrics> latency_debug;
  std::vector<IndexPartitionDebug> partition_debug;
};

struct MergeEventRecord {
  uint32_t base_rows{0};
  uint32_t frozen_rows{0};
  uint32_t patched_partitions{0};
  uint32_t append_partitions{0};
  uint32_t recluster_partitions{0};
  double moved_delta_ratio{0.0};
  double avg_assignment_dist_ratio{1.0};
  double max_assignment_dist_ratio{1.0};
  double imbalance_before{0.0};
  double imbalance_after{0.0};
  uint32_t main_non_empty_lists_after{0};
  uint32_t main_max_list_after{0};
  double main_avg_non_empty_list_after{0.0};
  double main_imbalance_after_real{0.0};
  std::string trigger_mode;
  std::string trigger_reason;
  bool trigger_rows{false};
  bool trigger_structure{false};
  bool trigger_qe_ratio{false};
  bool trigger_drift{false};
  bool trigger_delta_main_ratio{false};
  bool trigger_imbalance{false};
  double trigger_qe_ratio_value{0.0};
  double trigger_drift_value{0.0};
  double trigger_delta_main_ratio_value{0.0};
  double trigger_imbalance_value{0.0};
  uint32_t trigger_active_rows{0};
  uint32_t trigger_active_nlist{0};
  uint32_t trigger_active_non_empty_lists{0};
  uint32_t trigger_active_max_list{0};
  double trigger_active_avg_non_empty_list{0.0};
  double merge_compute_ms{0.0};
  double merge_ms{0.0};
  double codebook_rebuild_ms{0.0};
  MergeProfiling profiling;
};

struct MergeTriggerDecision {
  bool rows_trigger{false};
  bool qe_ratio_trigger{false};
  bool drift_trigger{false};
  bool delta_main_ratio_trigger{false};
  bool imbalance_trigger{false};
  bool structure_trigger{false};
  bool should_trigger{false};
  uint32_t active_rows{0};
  uint32_t active_nlist{0};
  uint32_t active_non_empty_lists{0};
  uint32_t active_max_list_size{0};
  double active_avg_non_empty_list_size{0.0};
  double qe_ratio_value{0.0};
  double drift_value{0.0};
  double delta_main_ratio_value{0.0};
  double imbalance_value{0.0};
  std::string mode;
  std::string reason;
};

struct GlobalRebuildDecision {
  bool should_trigger{false};
  bool main_rows_trigger{false};
  bool imbalance_trigger{false};
  uint32_t seen_rows{0};
  uint32_t main_rows_current{0};
  uint32_t main_rows_since_last_rebuild{0};
  uint32_t main_nlist{0};
  uint32_t main_non_empty_lists{0};
  uint32_t main_max_list_size{0};
  double main_avg_non_empty_list_size{0.0};
  double main_imbalance_ratio{0.0};
  std::string reason;
};

struct GlobalRebuildEventRecord {
  uint32_t base_rows{0};
  uint32_t old_main_rows{0};
  uint32_t new_main_rows{0};
  uint32_t active_seed_rows{0};
  uint32_t rebuild_count{0};
  uint32_t max_count{0};
  uint32_t main_nlist{0};
  uint32_t main_non_empty_lists{0};
  uint32_t main_max_list_size{0};
  double main_avg_non_empty_list_size{0.0};
  bool trigger_main_rows{false};
  bool trigger_imbalance{false};
  uint32_t trigger_main_rows_since_last_rebuild{0};
  uint32_t trigger_main_rows_threshold{0};
  double trigger_imbalance_threshold{0.0};
  double threshold{0.0};
  double main_imbalance_ratio{0.0};
  double whitening_ms{0.0};
  double whitening_transform_ms{0.0};
  double main_build_ms{0.0};
  double main_add_ms{0.0};
  double delta_seed_ms{0.0};
  double total_ms{0.0};
  double wall_total_ms{0.0};
  std::string reason;
};

struct SearchRoute {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint8_t from_new{0};
  std::string name;
  uint32_t rows{0};
};

struct ExactDocCandidate {
  float dist{0.0f};
  DocId doc_id{0};
};

}  // namespace ann::eval::run_eval

namespace ann::eval::run_eval_ms {

struct WorkloadOp {
  uint32_t op_id{0};
  std::string operation;
  uint32_t start{0};
  uint32_t end{0};
  uint32_t active_rows{0};
  int32_t round{-1};
  int32_t cluster{-1};
  uint32_t source_op_id{0};
};

struct Workload {
  std::string path;
  std::string dataset;
  uint32_t num_vectors{0};
  uint32_t dim{0};
  std::vector<WorkloadOp> operations;
};

struct DistributionStatsLite {
  double avg{0.0};
  double p5{0.0};
  double p50{0.0};
  double p95{0.0};
  double p99{0.0};
  double min{0.0};
  double max{0.0};
};

struct DeltaShard {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint32_t rows{0};
  uint32_t shard_id{0};
};

struct SearchRoute {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint8_t from_new{0};
  std::string name;
  uint32_t rows{0};
};

struct EvalMetricsLite {
  bool recall_available{true};
  double recall{0.0};
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
  double whitening_p50{0.0};
  double whitening_p99{0.0};
  double avg_whiten_ms{0.0};
  double avg_search_ms{0.0};
  double avg_pq_lut_build_us{0.0};
  double avg_pq_adc_scan_us{0.0};
  double search_p50{0.0};
  double search_p99{0.0};
  double total_p50{0.0};
  double total_p99{0.0};
  double avg_query_ms{0.0};
  double end_to_end_overhead_ms{0.0};
  double p50_query_ms{0.0};
  double p99_query_ms{0.0};
  double qps{0.0};
  double avg_scanned{0.0};
  double scanned_p50{0.0};
  double scanned_p99{0.0};
  double scanned_max{0.0};
  double scan_ratio{0.0};
  double scanned_per_topk{0.0};
  double query_eval_ms{0.0};
  uint32_t query_count{0};
  double gt_probed_rate{0.0};
  double recall_on_probed_gt{0.0};
  double exact_recall_on_probed_candidates{0.0};
  double avg_pq_rank_loss{0.0};
  uint32_t miss_not_probed{0};
  uint32_t miss_probed_filtered_by_pq{0};
  uint32_t pq_rank_loss_count{0};
  uint64_t rerank_topk_main_total{0};
  uint64_t rerank_topk_delta_total{0};
  double rerank_topk_main_ratio{0.0};
  double rerank_topk_delta_ratio{0.0};
  double rerank_topk_main_avg{0.0};
  double rerank_topk_delta_avg{0.0};
  uint32_t main_route_queries{0};
  uint32_t active_delta_route_queries{0};
  uint32_t frozen_delta_route_queries{0};
  std::vector<std::string> worst_queries;
};

struct SnapshotLite {
  uint32_t search_id{0};
  uint32_t op_id{0};
  uint32_t active_rows{0};
  uint32_t main_rows{0};
  uint32_t active_delta_rows{0};
  uint32_t frozen_delta_rows{0};
  int32_t round{-1};
  int32_t cluster{-1};
  double recall{0.0};
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
  double avg_search_ms{0.0};
  double end_to_end_overhead_ms{0.0};
  double avg_pq_lut_build_us{0.0};
  double avg_pq_adc_scan_us{0.0};
  double search_p50{0.0};
  double search_p99{0.0};
  double total_p50{0.0};
  double total_p99{0.0};
  double avg_query_ms{0.0};
  double p50_query_ms{0.0};
  double p99_query_ms{0.0};
  double qps{0.0};
  double avg_scanned{0.0};
  double scanned_p50{0.0};
  double scanned_p99{0.0};
  double scanned_max{0.0};
  double scan_ratio{0.0};
  double scanned_per_topk{0.0};
  double query_eval_ms{0.0};
  uint32_t query_count{0};
  uint32_t main_route_queries{0};
  uint32_t active_delta_route_queries{0};
  uint32_t frozen_delta_route_queries{0};
  double imbalance_ratio{0.0};
  double cumulative_update_ms{0.0};
  double cumulative_merge_ms{0.0};
  double cumulative_global_rebuild_ms{0.0};
  double snapshot_rows{0.0};
  double snapshot_maintenance_ms{0.0};
  double maintenance_ms_per_vector{0.0};
  double update_throughput_vecps{0.0};
  double amortized_update_throughput_vecps{0.0};
  double gt_probed_rate{0.0};
  double recall_on_probed_gt{0.0};
  double exact_recall_on_probed_candidates{0.0};
  double avg_pq_rank_loss{0.0};
  uint32_t miss_not_probed{0};
  uint32_t miss_probed_filtered_by_pq{0};
  uint32_t pq_rank_loss_count{0};
  uint64_t rerank_topk_main_total{0};
  uint64_t rerank_topk_delta_total{0};
  double rerank_topk_main_ratio{0.0};
  double rerank_topk_delta_ratio{0.0};
  double rerank_topk_main_avg{0.0};
  double rerank_topk_delta_avg{0.0};
  std::vector<std::string> worst_queries;
  bool skipped_warmup{false};
};

struct SummaryOutput {
  bool recall_available{false};
  DistributionStatsLite recall_stats;
  DistributionStatsLite qps_stats;
  DistributionStatsLite latency_stats;
  DistributionStatsLite scanned_stats;
  DistributionStatsLite update_throughput_stats;
  const SnapshotLite* final_snapshot{nullptr};
  double final_recall{0.0};
  double total_maintenance_ms{0.0};
  double build_rebuild_ms{0.0};
  double amortized_update_throughput{0.0};
  double maintenance_ms_per_vector{0.0};
  double final_imbalance_ratio{0.0};
  double final_scan_ratio{0.0};
  double final_scanned_per_topk{0.0};
  uint64_t summary_gt_new_total{0};
  uint64_t summary_gt_old_total{0};
  uint64_t summary_hit_new_total{0};
  uint64_t summary_hit_old_total{0};
  double summary_recall_new{0.0};
  double summary_recall_old{0.0};
  double summary_gt_new_ratio{0.0};
  double update_per_vector_ms{0.0};
  double update_throughput_vecps{0.0};
};

struct RecallAgeMetrics {
  double recall_new{0.0};
  double recall_old{0.0};
  double gt_new_ratio{0.0};
  uint64_t gt_new_total{0};
  uint64_t gt_old_total{0};
  uint64_t hit_new_total{0};
  uint64_t hit_old_total{0};
};

struct ExactDocCandidate {
  float dist{0.0f};
  DocId doc_id{0};
};

}  // namespace ann::eval::run_eval_ms
