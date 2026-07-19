#pragma once

#include <cstdint>
#include <string>

#include "common/result.h"

namespace ann {

struct Config {
  uint32_t ivf_nlist{1024};
  uint32_t topk{10};
  uint32_t nprobe{8};
  bool use_whitening{true};
  bool use_cosine{false};
  bool enable_dual_route{true};
  // When true, query only main index route and skip delta routes.
  bool main_query_only{false};
  uint32_t dim{128};
  uint32_t seed{42};
  bool pq_enable{false};
  uint32_t pq_m{0};
  uint32_t pq_nbits{8};
  bool pq_residual{true};
  uint32_t max_queries{0};
  uint32_t snapshot_interval{0};
  bool enable_miss_diag{true};
  bool enable_rerank_source_diag{false};
  // When true, emit extra latency diagnostics for route-level breakdown and slow queries.
  bool enable_latency_debug{false};
  bool exact_rerank_enable{false};
  uint32_t exact_rerank_candidates_per_route{0};
  // Route-specific exact-rerank candidate caps. 0 falls back to
  // exact_rerank_candidates_per_route.
  uint32_t main_exact_rerank_candidates{0};
  uint32_t active_exact_rerank_candidates{0};
  uint32_t frozen_exact_rerank_candidates{0};
  bool enable_streaming{false};
  uint32_t main_index_rows{0};
  uint32_t delta_train_window{1};
  // 0 means fallback to ivf_nlist.
  uint32_t delta_ivf_nlist{0};
  // Partition merge score:
  // score = alpha * growth_ratio + beta * avg_residual_dist.
  // If score >= threshold, the partition is reclustered during merge.
  double merge_score_alpha{1.0};
  double merge_score_beta{1e-7};
  double merge_score_threshold{10.0};
  // Merge trigger config is decoupled from delta training window.
  // mode:
  // - "rows": only row-based trigger
  // - "qe_ratio": only OnlinePQ qe-ratio trigger
  // - "drift": only codebook drift trigger
  // - "imbalance": only active-delta list imbalance trigger
  // - "delta_main_ratio": only active-delta/main ratio trigger
  // - "state": any state trigger (qe_ratio/drift/imbalance/delta_main_ratio)
  // - "hybrid": rows or any state trigger
  std::string merge_trigger_mode{"rows"};
  // Row upper bound of current active-delta (including warm-up vectors).
  // 0 means fallback to delta_train_rows for backward compatibility.
  uint32_t merge_trigger_rows{0};
  // Optional merge triggers; 0 means disabled.
  double merge_trigger_qe_ratio{0.0};
  double merge_trigger_drift{0.0};
  // Trigger when active_delta_rows / main_rows >= threshold. 0 disables it.
  double merge_trigger_delta_main_ratio{0.0};
  // Trigger when max_list_size / avg_non_empty_list_size >= threshold. 0 disables it.
  double merge_trigger_imbalance_ratio{0.0};
  // Delta->main assignment mode during merge:
  // - "nearest": always nearest main centroid
  // - "balanced_append": constrained local balancing + neighborhood repartition patching
  std::string merge_assignment_mode{"balanced_append"};
  // top-r nearest centroids considered by constrained assignment.
  uint32_t merge_assignment_top_r{4};
  // candidate must satisfy dist <= gamma * nearest_dist.
  double merge_assignment_gamma{1.05};
  // hard cap ratio on projected list size: hard_cap = avg_after * ratio.
  double merge_assignment_hard_cap_ratio{1.5};
  // cost weight for balance penalty in constrained assignment.
  double merge_assignment_lambda{0.2};
  // Global rebuild trigger config (keep small and explicit).
  bool enable_global_rebuild{false};
  // Hard cap of global rebuild count in one run; 0 disables global rebuild.
  uint32_t global_rebuild_max_count{0};
  // Trigger global rebuild when main max_list_size / avg_non_empty_list_size >= threshold.
  // 0 disables this trigger.
  double global_rebuild_main_imbalance_ratio{0.0};
  // Force global rebuild when main rows increased by this amount since last global rebuild.
  // 0 disables this fallback trigger.
  uint32_t global_rebuild_force_main_rows{0};
  // Minimum row gap between two global rebuilds. 0 disables cooldown.
  uint32_t global_rebuild_cooldown_rows{0};
  std::string streaming_mode{"streaming"};
  uint32_t stream_batch_size{100};
  bool streaming_use_stream_batch_size{false};
  bool online_pq_enable{true};
  std::string online_pq_update_scheme{"minibatch"};
  uint32_t online_pq_sliding_window_size{0};
  bool online_pq_sliding_window_use_batches{true};
  double online_pq_qe_ratio_threshold{1.05};
  double online_pq_ema_alpha{0.1};
  double online_pq_eps{1e-6};
  bool online_pq_warmup_enable{false};
  uint32_t online_pq_warmup_batches{0};
  uint32_t online_pq_force_update_interval{0};
  bool online_pq_partial_top_alpha{false};
  double online_pq_alpha{1.0};
  bool online_pq_partial_top_lambda{false};
  double online_pq_lambda{1.0};
  bool online_pq_reencode_batch{true};

  std::string ToString() const;
};

Result<Config> LoadConfigFromJson(const std::string& path);

}  // namespace ann