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
  bool enable_miss_diag{true};
  bool exact_rerank_enable{false};
  uint32_t exact_rerank_candidates_per_route{0};
  bool enable_streaming{false};
  uint32_t main_index_rows{0};
  uint32_t delta_train_window{1};
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
