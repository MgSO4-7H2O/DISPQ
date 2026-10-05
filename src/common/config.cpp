#include "common/config.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ann {

namespace {

bool ExtractUint(const std::string& text, const std::string& key, uint32_t* out) {
  std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (std::regex_search(text, match, re)) {
    *out = static_cast<uint32_t>(std::stoul(match[1]));
    return true;
  }
  return false;
}

bool ExtractUintArray(const std::string& text,
                      const std::string& key,
                      std::vector<uint32_t>* out) {
  std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(text, match, re)) return false;
  out->clear();
  const std::string body = match[1];
  std::regex number("[0-9]+");
  for (std::sregex_iterator it(body.begin(), body.end(), number), end; it != end; ++it) {
    const unsigned long long value = std::stoull((*it)[0]);
    if (value > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("nprobe_sweep value exceeds uint32_t");
    }
    out->push_back(static_cast<uint32_t>(value));
  }
  return true;
}

bool ExtractBool(const std::string& text, const std::string& key, bool* out) {
  std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (std::regex_search(text, match, re)) {
    *out = (match[1] == "true");
    return true;
  }
  return false;
}

bool ExtractString(const std::string& text, const std::string& key, std::string* out) {
  std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  if (std::regex_search(text, match, re)) {
    *out = match[1];
    return true;
  }
  return false;
}

bool ExtractDouble(const std::string& text, const std::string& key, double* out) {
  std::regex re("\"" + key +
                "\"\\s*:\\s*(-?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
  std::smatch match;
  if (std::regex_search(text, match, re)) {
    *out = std::stod(match[1]);
    return true;
  }
  return false;
}

}  // namespace

std::string Config::ToString() const {
  std::ostringstream oss;
  oss << "Config{"
      << "ivf_nlist=" << ivf_nlist << ", "
      << "topk=" << topk << ", "
      << "nprobe=" << nprobe << ", "
      << "nprobe_sweep_size=" << nprobe_sweep.size() << ", "
      << "final_state_cache_mode=" << final_state_cache_mode << ", "
      << "final_state_cache_store_vectors=" << std::boolalpha
      << final_state_cache_store_vectors << ", "
      << "use_whitening=" << std::boolalpha << use_whitening << ", "
      << "use_cosine=" << std::boolalpha << use_cosine << ", "
      << "dim=" << dim << ", "
      << "seed=" << seed << ", "
      << "pq_enable=" << std::boolalpha << pq_enable << ", "
      << "pq_m=" << pq_m << ", "
      << "pq_nbits=" << pq_nbits << ", "
      << "pq_residual=" << std::boolalpha << pq_residual << ", "
      << "pq_codebook_dimension_major=" << std::boolalpha << pq_codebook_dimension_major << ", "
      << "pq_codes_subquantizer_major=" << std::boolalpha << pq_codes_subquantizer_major << ", "
      << "max_queries=" << max_queries << ", "
      << "snapshot_interval=" << snapshot_interval << ", "
      << "enable_dynamic_ground_truth=" << std::boolalpha << enable_dynamic_ground_truth
      << ", "
      << "skip_query_ground_truth=" << std::boolalpha << skip_query_ground_truth << ", "
      << "enable_miss_diag=" << std::boolalpha << enable_miss_diag << ", "
      << "enable_rerank_source_diag=" << std::boolalpha << enable_rerank_source_diag << ", "
      << "enable_latency_debug=" << std::boolalpha << enable_latency_debug << ", "
      << "exact_rerank_enable=" << std::boolalpha << exact_rerank_enable << ", "
      << "exact_rerank_candidates_per_route=" << exact_rerank_candidates_per_route << ", "
      << "main_exact_rerank_candidates=" << main_exact_rerank_candidates << ", "
      << "active_exact_rerank_candidates=" << active_exact_rerank_candidates << ", "
      << "frozen_exact_rerank_candidates=" << frozen_exact_rerank_candidates << ", "
      << "enable_streaming=" << std::boolalpha << enable_streaming << ", "
      << "main_index_rows=" << main_index_rows << ", "
      << "delta_train_window=" << delta_train_window << ", "
      << "delta_ivf_nlist=" << delta_ivf_nlist << ", "
      << "merge_score_alpha=" << merge_score_alpha << ", "
      << "merge_score_beta=" << merge_score_beta << ", "
      << "merge_score_threshold=" << merge_score_threshold << ", "
      << "merge_trigger_mode=" << merge_trigger_mode << ", "
      << "merge_trigger_rows=" << merge_trigger_rows << ", "
      << "merge_trigger_qe_ratio=" << merge_trigger_qe_ratio << ", "
      << "merge_trigger_drift=" << merge_trigger_drift << ", "
      << "merge_trigger_delta_main_ratio=" << merge_trigger_delta_main_ratio << ", "
      << "merge_trigger_imbalance_ratio=" << merge_trigger_imbalance_ratio << ", "
      << "enable_merge=" << std::boolalpha << enable_merge << ", "
      << "merge_assignment_mode=" << merge_assignment_mode << ", "
      << "merge_assignment_top_r=" << merge_assignment_top_r << ", "
      << "merge_assignment_gamma=" << merge_assignment_gamma << ", "
      << "merge_assignment_hard_cap_ratio=" << merge_assignment_hard_cap_ratio << ", "
      << "merge_assignment_lambda=" << merge_assignment_lambda << ", "
      << "enable_global_rebuild=" << std::boolalpha << enable_global_rebuild << ", "
      << "global_rebuild_max_count=" << global_rebuild_max_count << ", "
      << "global_rebuild_main_imbalance_ratio=" << global_rebuild_main_imbalance_ratio << ", "
      << "global_rebuild_force_main_rows=" << global_rebuild_force_main_rows << ", "
      << "global_rebuild_cooldown_rows=" << global_rebuild_cooldown_rows << ", "
      << "streaming_mode=" << streaming_mode << ", "
      << "stream_batch_size=" << stream_batch_size << ", "
      << "streaming_use_stream_batch_size=" << std::boolalpha << streaming_use_stream_batch_size
      << ", "
      << "online_pq_enable=" << std::boolalpha << online_pq_enable << ", "
      << "online_pq_update_scheme=" << online_pq_update_scheme << ", "
      << "online_pq_sliding_window_size=" << online_pq_sliding_window_size << ", "
      << "online_pq_sliding_window_use_batches=" << std::boolalpha
      << online_pq_sliding_window_use_batches << ", "
      << "online_pq_qe_ratio_threshold=" << online_pq_qe_ratio_threshold << ", "
      << "online_pq_ema_alpha=" << online_pq_ema_alpha << ", "
      << "online_pq_eps=" << online_pq_eps << ", "
      << "online_pq_warmup_enable=" << std::boolalpha << online_pq_warmup_enable << ", "
      << "online_pq_warmup_batches=" << online_pq_warmup_batches << ", "
      << "online_pq_force_update_interval=" << online_pq_force_update_interval << ", "
      << "online_pq_partial_top_alpha=" << std::boolalpha << online_pq_partial_top_alpha << ", "
      << "online_pq_alpha=" << online_pq_alpha << ", "
      << "online_pq_partial_top_lambda=" << std::boolalpha << online_pq_partial_top_lambda << ", "
      << "online_pq_lambda=" << online_pq_lambda << ", "
      << "online_pq_reencode_batch=" << std::boolalpha << online_pq_reencode_batch << ", "
      << "concurrent_workload_enable=" << std::boolalpha << concurrent_workload_enable << ", "
      << "concurrent_query_workers=" << concurrent_query_workers << ", "
      << "concurrent_query_threads_per_request=" << concurrent_query_threads_per_request << ", "
      << "concurrent_insert_threads=" << concurrent_insert_threads << ", "
      << "concurrent_maintenance_threads=" << concurrent_maintenance_threads << ", "
      << "concurrent_target_insert_vecps=" << concurrent_target_insert_vecps << ", "
      << "concurrent_gt_mode=" << concurrent_gt_mode << ", "
      << "concurrent_gt_path=" << concurrent_gt_path << "}";
  return oss.str();
}

Result<Config> LoadConfigFromJson(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    return Status::IOError("Failed to open config: " + path);
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  const std::string text = buffer.str();

  Config cfg;
  ExtractUint(text, "ivf_nlist", &cfg.ivf_nlist);
  ExtractUint(text, "topk", &cfg.topk);
  ExtractUint(text, "nprobe", &cfg.nprobe);
  ExtractUintArray(text, "nprobe_sweep", &cfg.nprobe_sweep);
  ExtractString(text, "final_state_cache_mode", &cfg.final_state_cache_mode);
  ExtractString(text, "final_state_cache_path", &cfg.final_state_cache_path);
  ExtractBool(text, "final_state_cache_store_vectors", &cfg.final_state_cache_store_vectors);
  ExtractUint(text, "dim", &cfg.dim);
  ExtractUint(text, "seed", &cfg.seed);
  ExtractBool(text, "use_whitening", &cfg.use_whitening);
  ExtractBool(text, "use_cosine", &cfg.use_cosine);
  ExtractBool(text, "pq_enable", &cfg.pq_enable);
  ExtractUint(text, "pq_m", &cfg.pq_m);
  ExtractUint(text, "pq_nbits", &cfg.pq_nbits);
  ExtractBool(text, "pq_residual", &cfg.pq_residual);
  ExtractBool(text, "pq_codebook_dimension_major", &cfg.pq_codebook_dimension_major);
  ExtractBool(text, "pq_codes_subquantizer_major", &cfg.pq_codes_subquantizer_major);
  ExtractUint(text, "max_queries", &cfg.max_queries);
  ExtractUint(text, "snapshot_interval", &cfg.snapshot_interval);
  ExtractBool(text, "enable_dynamic_ground_truth", &cfg.enable_dynamic_ground_truth);
  ExtractBool(text, "skip_query_ground_truth", &cfg.skip_query_ground_truth);
  ExtractBool(text, "enable_miss_diag", &cfg.enable_miss_diag);
  ExtractBool(text, "enable_rerank_source_diag", &cfg.enable_rerank_source_diag);
  ExtractBool(text, "enable_latency_debug", &cfg.enable_latency_debug);
  ExtractBool(text, "exact_rerank_enable", &cfg.exact_rerank_enable);
  ExtractUint(text,
              "exact_rerank_candidates_per_route",
              &cfg.exact_rerank_candidates_per_route);
  ExtractUint(text, "main_exact_rerank_candidates", &cfg.main_exact_rerank_candidates);
  ExtractUint(text, "active_exact_rerank_candidates", &cfg.active_exact_rerank_candidates);
  ExtractUint(text, "frozen_exact_rerank_candidates", &cfg.frozen_exact_rerank_candidates);
  ExtractBool(text, "enable_streaming", &cfg.enable_streaming);
  ExtractUint(text, "main_index_rows", &cfg.main_index_rows);
  ExtractUint(text, "delta_train_window", &cfg.delta_train_window);
  ExtractUint(text, "delta_ivf_nlist", &cfg.delta_ivf_nlist);
  ExtractDouble(text, "merge_score_alpha", &cfg.merge_score_alpha);
  ExtractDouble(text, "merge_score_beta", &cfg.merge_score_beta);
  ExtractDouble(text, "merge_score_threshold", &cfg.merge_score_threshold);
  ExtractString(text, "merge_trigger_mode", &cfg.merge_trigger_mode);
  ExtractUint(text, "merge_trigger_rows", &cfg.merge_trigger_rows);
  ExtractDouble(text, "merge_trigger_qe_ratio", &cfg.merge_trigger_qe_ratio);
  ExtractDouble(text, "merge_trigger_drift", &cfg.merge_trigger_drift);
  ExtractDouble(text, "merge_trigger_delta_main_ratio", &cfg.merge_trigger_delta_main_ratio);
  ExtractDouble(text, "merge_trigger_imbalance_ratio", &cfg.merge_trigger_imbalance_ratio);
  ExtractBool(text, "enable_merge", &cfg.enable_merge);
  ExtractString(text, "merge_assignment_mode", &cfg.merge_assignment_mode);
  ExtractUint(text, "merge_assignment_top_r", &cfg.merge_assignment_top_r);
  ExtractDouble(text, "merge_assignment_gamma", &cfg.merge_assignment_gamma);
  ExtractDouble(text, "merge_assignment_hard_cap_ratio", &cfg.merge_assignment_hard_cap_ratio);
  ExtractDouble(text, "merge_assignment_lambda", &cfg.merge_assignment_lambda);
  ExtractBool(text, "enable_global_rebuild", &cfg.enable_global_rebuild);
  ExtractUint(text, "global_rebuild_max_count", &cfg.global_rebuild_max_count);
  ExtractDouble(text,
                "global_rebuild_main_imbalance_ratio",
                &cfg.global_rebuild_main_imbalance_ratio);
  ExtractUint(text, "global_rebuild_force_main_rows", &cfg.global_rebuild_force_main_rows);
  ExtractUint(text, "global_rebuild_cooldown_rows", &cfg.global_rebuild_cooldown_rows);
  ExtractString(text, "streaming_mode", &cfg.streaming_mode);
  ExtractUint(text, "stream_batch_size", &cfg.stream_batch_size);
  ExtractBool(text, "streaming_use_stream_batch_size", &cfg.streaming_use_stream_batch_size);
  ExtractBool(text, "online_pq_enable", &cfg.online_pq_enable);
  ExtractString(text, "online_pq_update_scheme", &cfg.online_pq_update_scheme);
  ExtractUint(text, "online_pq_sliding_window_size", &cfg.online_pq_sliding_window_size);
  ExtractBool(text,
              "online_pq_sliding_window_use_batches",
              &cfg.online_pq_sliding_window_use_batches);
  ExtractDouble(text, "online_pq_qe_ratio_threshold", &cfg.online_pq_qe_ratio_threshold);
  ExtractDouble(text, "online_pq_ema_alpha", &cfg.online_pq_ema_alpha);
  ExtractDouble(text, "online_pq_eps", &cfg.online_pq_eps);
  ExtractBool(text, "online_pq_warmup_enable", &cfg.online_pq_warmup_enable);
  ExtractUint(text, "online_pq_warmup_batches", &cfg.online_pq_warmup_batches);
  ExtractUint(text, "online_pq_force_update_interval", &cfg.online_pq_force_update_interval);
  ExtractBool(text, "online_pq_partial_top_alpha", &cfg.online_pq_partial_top_alpha);
  ExtractDouble(text, "online_pq_alpha", &cfg.online_pq_alpha);
  ExtractBool(text, "online_pq_partial_top_lambda", &cfg.online_pq_partial_top_lambda);
  ExtractDouble(text, "online_pq_lambda", &cfg.online_pq_lambda);
  ExtractBool(text, "online_pq_reencode_batch", &cfg.online_pq_reencode_batch);
  ExtractBool(text, "concurrent_workload_enable", &cfg.concurrent_workload_enable);
  ExtractUint(text, "concurrent_query_workers", &cfg.concurrent_query_workers);
  ExtractUint(text,
              "concurrent_query_threads_per_request",
              &cfg.concurrent_query_threads_per_request);
  ExtractUint(text, "concurrent_insert_threads", &cfg.concurrent_insert_threads);
  ExtractUint(text, "concurrent_maintenance_threads", &cfg.concurrent_maintenance_threads);
  ExtractDouble(text, "concurrent_target_insert_vecps", &cfg.concurrent_target_insert_vecps);
  ExtractString(text, "concurrent_gt_mode", &cfg.concurrent_gt_mode);
  ExtractString(text, "concurrent_gt_path", &cfg.concurrent_gt_path);

  if (cfg.streaming_mode != "streaming" && cfg.streaming_mode != "batch") {
    return Status::InvalidArgument(
        "streaming_mode must be either \"streaming\" or \"batch\"");
  }
  if (cfg.stream_batch_size == 0) {
    return Status::InvalidArgument("stream_batch_size must be > 0");
  }
  if (cfg.concurrent_gt_mode != "off" && cfg.concurrent_gt_mode != "prepare" &&
      cfg.concurrent_gt_mode != "load") {
    return Status::InvalidArgument("concurrent_gt_mode must be off, prepare, or load");
  }
  if (cfg.concurrent_target_insert_vecps < 0.0) {
    return Status::InvalidArgument("concurrent_target_insert_vecps must be >= 0");
  }
  if (cfg.concurrent_workload_enable &&
      (cfg.concurrent_query_workers == 0 || cfg.concurrent_query_threads_per_request == 0 ||
       cfg.concurrent_insert_threads == 0 || cfg.concurrent_maintenance_threads == 0)) {
    return Status::InvalidArgument("concurrent workload thread counts must be > 0");
  }
  if (cfg.exact_rerank_enable && cfg.exact_rerank_candidates_per_route == 0) {
    return Status::InvalidArgument(
        "exact_rerank_candidates_per_route must be > 0 when exact_rerank_enable=true");
  }
  if (cfg.exact_rerank_enable && cfg.exact_rerank_candidates_per_route < cfg.topk) {
    return Status::InvalidArgument(
        "exact_rerank_candidates_per_route must be >= topk when exact_rerank_enable=true");
  }
  if (cfg.enable_rerank_source_diag && !cfg.exact_rerank_enable) {
    return Status::InvalidArgument(
        "enable_rerank_source_diag requires exact_rerank_enable=true");
  }
  if (cfg.enable_streaming && cfg.delta_train_window == 0) {
    return Status::InvalidArgument("delta_train_window must be > 0 when enable_streaming=true");
  }
  if (cfg.merge_score_alpha < 0.0) {
    return Status::InvalidArgument("merge_score_alpha must be >= 0");
  }
  if (cfg.merge_score_beta < 0.0) {
    return Status::InvalidArgument("merge_score_beta must be >= 0");
  }
  if (cfg.merge_score_threshold < 0.0) {
    return Status::InvalidArgument("merge_score_threshold must be >= 0");
  }
  if (cfg.merge_trigger_mode != "rows" && cfg.merge_trigger_mode != "qe_ratio" &&
      cfg.merge_trigger_mode != "drift" && cfg.merge_trigger_mode != "imbalance" &&
      cfg.merge_trigger_mode != "delta_main_ratio" && cfg.merge_trigger_mode != "state" &&
      cfg.merge_trigger_mode != "hybrid") {
    return Status::InvalidArgument(
        "merge_trigger_mode must be one of \"rows\", \"qe_ratio\", \"drift\", "
        "\"imbalance\", \"delta_main_ratio\", \"state\", \"hybrid\"");
  }
  if (cfg.merge_trigger_qe_ratio < 0.0) {
    return Status::InvalidArgument("merge_trigger_qe_ratio must be >= 0");
  }
  if (cfg.merge_trigger_drift < 0.0) {
    return Status::InvalidArgument("merge_trigger_drift must be >= 0");
  }
  if (cfg.merge_trigger_delta_main_ratio < 0.0) {
    return Status::InvalidArgument("merge_trigger_delta_main_ratio must be >= 0");
  }
  if (cfg.merge_trigger_imbalance_ratio < 0.0) {
    return Status::InvalidArgument("merge_trigger_imbalance_ratio must be >= 0");
  }
  if (cfg.merge_assignment_mode == "local_constrained") {
    // Backward-compat alias: local_constrained now uses balanced_append behavior.
    cfg.merge_assignment_mode = "balanced_append";
  }
  if (cfg.merge_assignment_mode != "nearest" &&
      cfg.merge_assignment_mode != "balanced_append") {
    return Status::InvalidArgument(
        "merge_assignment_mode must be one of \"nearest\", \"balanced_append\"");
  }
  if (cfg.merge_assignment_top_r == 0) {
    return Status::InvalidArgument("merge_assignment_top_r must be > 0");
  }
  if (cfg.merge_assignment_gamma <= 0.0) {
    return Status::InvalidArgument("merge_assignment_gamma must be > 0");
  }
  if (cfg.merge_assignment_hard_cap_ratio <= 0.0) {
    return Status::InvalidArgument("merge_assignment_hard_cap_ratio must be > 0");
  }
  if (cfg.merge_assignment_lambda < 0.0) {
    return Status::InvalidArgument("merge_assignment_lambda must be >= 0");
  }
  if (cfg.global_rebuild_main_imbalance_ratio < 0.0) {
    return Status::InvalidArgument("global_rebuild_main_imbalance_ratio must be >= 0");
  }
  if (cfg.merge_trigger_mode == "qe_ratio" && cfg.merge_trigger_qe_ratio <= 0.0) {
    return Status::InvalidArgument(
        "merge_trigger_qe_ratio must be > 0 when merge_trigger_mode=\"qe_ratio\"");
  }
  if (cfg.merge_trigger_mode == "drift" && cfg.merge_trigger_drift <= 0.0) {
    return Status::InvalidArgument(
        "merge_trigger_drift must be > 0 when merge_trigger_mode=\"drift\"");
  }
  if (cfg.merge_trigger_mode == "imbalance" && cfg.merge_trigger_imbalance_ratio <= 0.0) {
    return Status::InvalidArgument(
        "merge_trigger_imbalance_ratio must be > 0 when merge_trigger_mode=\"imbalance\"");
  }
  if (cfg.merge_trigger_mode == "delta_main_ratio" && cfg.merge_trigger_delta_main_ratio <= 0.0) {
    return Status::InvalidArgument(
        "merge_trigger_delta_main_ratio must be > 0 when "
        "merge_trigger_mode=\"delta_main_ratio\"");
  }
  if (cfg.merge_trigger_mode == "state" && cfg.merge_trigger_qe_ratio <= 0.0 &&
      cfg.merge_trigger_drift <= 0.0 && cfg.merge_trigger_delta_main_ratio <= 0.0 &&
      cfg.merge_trigger_imbalance_ratio <= 0.0) {
    return Status::InvalidArgument(
        "state merge trigger needs at least one enabled trigger: "
        "qe_ratio/drift/delta_main_ratio/imbalance");
  }
  if (cfg.merge_trigger_mode == "hybrid" && cfg.merge_trigger_rows == 0 &&
      cfg.merge_trigger_qe_ratio <= 0.0 && cfg.merge_trigger_drift <= 0.0 &&
      cfg.merge_trigger_delta_main_ratio <= 0.0 && cfg.merge_trigger_imbalance_ratio <= 0.0) {
    return Status::InvalidArgument(
        "hybrid merge trigger needs at least one enabled trigger: "
        "rows/qe_ratio/drift/delta_main_ratio/imbalance");
  }
  if (cfg.enable_global_rebuild && cfg.global_rebuild_max_count == 0) {
    return Status::InvalidArgument(
        "global_rebuild_max_count must be > 0 when enable_global_rebuild=true");
  }
  if (cfg.enable_global_rebuild && cfg.global_rebuild_main_imbalance_ratio <= 0.0) {
    if (cfg.global_rebuild_force_main_rows == 0) {
      return Status::InvalidArgument(
          "enable_global_rebuild=true requires at least one trigger: "
          "global_rebuild_main_imbalance_ratio>0 or global_rebuild_force_main_rows>0");
    }
  }
  if (cfg.online_pq_update_scheme != "minibatch" &&
      cfg.online_pq_update_scheme != "sliding_window") {
    return Status::InvalidArgument(
        "online_pq_update_scheme must be either \"minibatch\" or \"sliding_window\"");
  }
  if (cfg.online_pq_update_scheme == "sliding_window" && cfg.online_pq_sliding_window_size == 0) {
    return Status::InvalidArgument(
        "online_pq_sliding_window_size must be > 0 when online_pq_update_scheme=\"sliding_window\"");
  }
  if (cfg.online_pq_update_scheme == "sliding_window" &&
      cfg.online_pq_force_update_interval == 0) {
    return Status::InvalidArgument(
        "online_pq_force_update_interval must be > 0 in sliding_window mode");
  }
  if (cfg.online_pq_ema_alpha <= 0.0 || cfg.online_pq_ema_alpha > 1.0) {
    return Status::InvalidArgument("online_pq_ema_alpha must be in (0,1]");
  }
  if (cfg.online_pq_eps <= 0.0) {
    return Status::InvalidArgument("online_pq_eps must be > 0");
  }
  if (cfg.online_pq_warmup_enable && cfg.online_pq_warmup_batches == 0) {
    return Status::InvalidArgument(
        "online_pq_warmup_batches must be > 0 when online_pq_warmup_enable=true");
  }
  if (cfg.online_pq_alpha <= 0.0 || cfg.online_pq_alpha > 1.0) {
    return Status::InvalidArgument("online_pq_alpha must be in (0,1]");
  }
  if (cfg.online_pq_lambda <= 0.0 || cfg.online_pq_lambda > 1.0) {
    return Status::InvalidArgument("online_pq_lambda must be in (0,1]");
  }

  if (std::any_of(cfg.nprobe_sweep.begin(), cfg.nprobe_sweep.end(),
                  [](uint32_t value) { return value == 0; })) {
    return Status::InvalidArgument("nprobe_sweep values must be positive");
  }
  if (cfg.final_state_cache_mode != "off" && cfg.final_state_cache_mode != "save" &&
      cfg.final_state_cache_mode != "load" && cfg.final_state_cache_mode != "prepare") {
    return Status::InvalidArgument("final_state_cache_mode must be off, save, load, or prepare");
  }
  if (cfg.final_state_cache_mode != "off" && cfg.final_state_cache_path.empty()) {
    return Status::InvalidArgument("final_state_cache_path is required when cache mode is enabled");
  }
  if (cfg.final_state_cache_mode == "load" && cfg.nprobe_sweep.empty()) {
    return Status::InvalidArgument("final-state cache load requires non-empty nprobe_sweep");
  }
  if (cfg.final_state_cache_mode == "prepare" && !cfg.nprobe_sweep.empty()) {
    return Status::InvalidArgument("final-state cache prepare cannot be combined with nprobe_sweep");
  }
  return cfg;
}

}  // namespace ann
