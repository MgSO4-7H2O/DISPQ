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
      << "snapshot_interval=" << snapshot_interval << "}";
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

  return cfg;
}

}  // namespace ann
