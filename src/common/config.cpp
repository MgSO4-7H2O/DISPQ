#include "common/config.h"

#include <fstream>
#include <regex>
#include <sstream>

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
  std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
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
      << "use_whitening=" << std::boolalpha << use_whitening << ", "
      << "enable_dual_route=" << std::boolalpha << enable_dual_route << ", "
      << "dim=" << dim << ", "
      << "seed=" << seed << ", "
      << "pq_enable=" << std::boolalpha << pq_enable << ", "
      << "pq_m=" << pq_m << ", "
      << "pq_nbits=" << pq_nbits << ", "
      << "pq_residual=" << std::boolalpha << pq_residual << ", "
      << "max_queries=" << max_queries << ", "
      << "snapshot_interval=" << snapshot_interval << ", "
      << "enable_streaming=" << std::boolalpha << enable_streaming << ", "
      << "main_index_rows=" << main_index_rows << ", "
      << "streaming_mode=" << streaming_mode << ", "
      << "stream_batch_size=" << stream_batch_size << ", "
      << "drift_window_size=" << drift_window_size << ", "
      << "drift_confirm_k=" << drift_confirm_k << ", "
      << "drift_confirm_m=" << drift_confirm_m << ", "
      << "drift_confirm_ratio=" << drift_confirm_ratio << ", "
      << "drift_min_delta_lifetime_windows=" << drift_min_delta_lifetime_windows << ", "
      << "drift_soft_npd_ratio=" << drift_soft_npd_ratio << ", "
      << "drift_soft_gain=" << drift_soft_gain << ", "
      << "drift_hard_nre_ratio=" << drift_hard_nre_ratio << ", "
      << "drift_hard_cm_z=" << drift_hard_cm_z << ", "
      << "drift_hard_lds=" << drift_hard_lds << ", "
      << "drift_max_closed_deltas=" << drift_max_closed_deltas << ", "
      << "drift_max_closed_ratio=" << drift_max_closed_ratio << ", "
      << "drift_active_delta_max_docs=" << drift_active_delta_max_docs << "}";
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
  ExtractUint(text, "dim", &cfg.dim);
  ExtractUint(text, "seed", &cfg.seed);
  ExtractBool(text, "use_whitening", &cfg.use_whitening);
  ExtractBool(text, "enable_dual_route", &cfg.enable_dual_route);
  ExtractBool(text, "pq_enable", &cfg.pq_enable);
  ExtractUint(text, "pq_m", &cfg.pq_m);
  ExtractUint(text, "pq_nbits", &cfg.pq_nbits);
  ExtractBool(text, "pq_residual", &cfg.pq_residual);
  ExtractUint(text, "max_queries", &cfg.max_queries);
  ExtractUint(text, "snapshot_interval", &cfg.snapshot_interval);
  ExtractBool(text, "enable_streaming", &cfg.enable_streaming);
  ExtractUint(text, "main_index_rows", &cfg.main_index_rows);
  ExtractString(text, "streaming_mode", &cfg.streaming_mode);
  ExtractUint(text, "stream_batch_size", &cfg.stream_batch_size);
  ExtractUint(text, "drift_window_size", &cfg.drift_window_size);
  ExtractUint(text, "drift_confirm_k", &cfg.drift_confirm_k);
  ExtractUint(text, "drift_confirm_m", &cfg.drift_confirm_m);
  ExtractDouble(text, "drift_confirm_ratio", &cfg.drift_confirm_ratio);
  ExtractUint(text, "drift_min_delta_lifetime_windows", &cfg.drift_min_delta_lifetime_windows);
  ExtractDouble(text, "drift_soft_npd_ratio", &cfg.drift_soft_npd_ratio);
  ExtractDouble(text, "drift_soft_gain", &cfg.drift_soft_gain);
  ExtractDouble(text, "drift_hard_nre_ratio", &cfg.drift_hard_nre_ratio);
  ExtractDouble(text, "drift_hard_cm_z", &cfg.drift_hard_cm_z);
  ExtractDouble(text, "drift_hard_lds", &cfg.drift_hard_lds);
  ExtractUint(text, "drift_max_closed_deltas", &cfg.drift_max_closed_deltas);
  ExtractDouble(text, "drift_max_closed_ratio", &cfg.drift_max_closed_ratio);
  ExtractUint(text, "drift_active_delta_max_docs", &cfg.drift_active_delta_max_docs);

  if (cfg.streaming_mode != "streaming" && cfg.streaming_mode != "batch") {
    return Status::InvalidArgument(
        "streaming_mode must be either \"streaming\" or \"batch\"");
  }
  if (cfg.stream_batch_size == 0) {
    return Status::InvalidArgument("stream_batch_size must be > 0");
  }
  if (cfg.drift_window_size == 0) {
    return Status::InvalidArgument("drift_window_size must be > 0");
  }
  if (cfg.drift_confirm_k == 0 || cfg.drift_confirm_m == 0) {
    return Status::InvalidArgument("drift_confirm_k and drift_confirm_m must be > 0");
  }
  if (cfg.drift_confirm_k > cfg.drift_confirm_m) {
    return Status::InvalidArgument("drift_confirm_k must be <= drift_confirm_m");
  }
  if (cfg.drift_confirm_ratio <= 0.0 || cfg.drift_confirm_ratio > 1.0) {
    return Status::InvalidArgument("drift_confirm_ratio must be in (0,1]");
  }
  if (cfg.drift_soft_npd_ratio <= 0.0 || cfg.drift_hard_nre_ratio <= 0.0) {
    return Status::InvalidArgument("drift threshold ratios must be > 0");
  }
  if (cfg.drift_soft_gain < 0.0) {
    return Status::InvalidArgument("drift_soft_gain must be >= 0");
  }
  if (cfg.drift_max_closed_ratio < 0.0 || cfg.drift_max_closed_ratio > 1.0) {
    return Status::InvalidArgument("drift_max_closed_ratio must be in [0,1]");
  }

  return cfg;
}

}  // namespace ann
