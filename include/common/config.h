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
  bool enable_dual_route{true};
  uint32_t dim{128};
  uint32_t seed{42};
  bool pq_enable{false};
  uint32_t pq_m{0};
  uint32_t pq_nbits{8};
  bool pq_residual{true};
  uint32_t max_queries{0};
  uint32_t snapshot_interval{0};
  bool enable_streaming{false};
  uint32_t main_index_rows{0};
  std::string streaming_mode{"streaming"};
  uint32_t stream_batch_size{100};
  uint32_t drift_window_size{1000};
  uint32_t drift_confirm_k{3};
  uint32_t drift_confirm_m{5};
  double drift_confirm_ratio{0.6};
  uint32_t drift_min_delta_lifetime_windows{2};
  double drift_soft_npd_ratio{1.2};
  double drift_soft_gain{0.05};
  double drift_hard_nre_ratio{1.5};
  double drift_hard_cm_z{3.0};
  double drift_hard_lds{0.2};
  uint32_t drift_max_closed_deltas{3};
  double drift_max_closed_ratio{0.15};
  uint32_t drift_active_delta_max_docs{20000};

  std::string ToString() const;
};

Result<Config> LoadConfigFromJson(const std::string& path);

}  // namespace ann
