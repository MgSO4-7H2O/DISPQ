#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/config.h"
#include "common/dataset.h"
#include "common/prebuilt_index.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "index/merge.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

constexpr uint32_t kDeltaKMeansIterationsDefault = 10;
constexpr uint32_t kWorstQueryDiagCount = 10;
constexpr uint32_t kSlowQueryDebugCount = 5;
constexpr uint32_t kAddBlockRows = 65536;
constexpr uint32_t kWhitenedMinGrowthRows = 65536;
constexpr size_t kGroundTruthBlockTargetBytes = static_cast<size_t>(64) << 20;

void EnsureWhitenedCapacity(MatrixRM* matrix,
                            uint32_t required_rows,
                            uint32_t max_rows,
                            uint32_t dim) {
  if (matrix == nullptr || required_rows <= static_cast<uint32_t>(matrix->rows())) {
    return;
  }
  const uint32_t current_rows = static_cast<uint32_t>(matrix->rows());
  const uint64_t growth = std::max<uint64_t>(current_rows / 2, kWhitenedMinGrowthRows);
  const uint64_t grown_rows = static_cast<uint64_t>(current_rows) + growth;
  const uint32_t new_rows = static_cast<uint32_t>(std::min<uint64_t>(
      max_rows, std::max<uint64_t>(required_rows, grown_rows)));
  matrix->conservativeResize(static_cast<Eigen::Index>(new_rows),
                             static_cast<Eigen::Index>(dim));
}

struct DistributionStats {
  double avg{0.0};
  double p50{0.0};
  double p90{0.0};
  double p99{0.0};
  double max{0.0};
};

double Percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double idx = q * static_cast<double>(values.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  if (hi >= values.size()) {
    hi = values.size() - 1;
  }
  const double frac = idx - static_cast<double>(lo);
  return values[lo] + (values[hi] - values[lo]) * frac;
}

DistributionStats SummarizeDistribution(const std::vector<double>& values) {
  DistributionStats stats;
  if (values.empty()) {
    return stats;
  }
  stats.avg =
      std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  stats.p50 = Percentile(values, 0.50);
  stats.p90 = Percentile(values, 0.90);
  stats.p99 = Percentile(values, 0.99);
  stats.max = *std::max_element(values.begin(), values.end());
  return stats;
}

struct SeriesStats {
  double avg{0.0};
  double p5{0.0};
  double p50{0.0};
  double p95{0.0};
  double p99{0.0};
  double min{0.0};
  double max{0.0};
};

SeriesStats SummarizeSeries(const std::vector<double>& values) {
  SeriesStats stats;
  if (values.empty()) {
    return stats;
  }
  stats.avg =
      std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  stats.p5 = Percentile(values, 0.05);
  stats.p50 = Percentile(values, 0.50);
  stats.p95 = Percentile(values, 0.95);
  stats.p99 = Percentile(values, 0.99);
  stats.min = *std::min_element(values.begin(), values.end());
  stats.max = *std::max_element(values.begin(), values.end());
  return stats;
}

uint64_t ReadProcStatusBytes(const std::string& key) {
  std::ifstream ifs("/proc/self/status");
  if (!ifs) {
    return 0;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.rfind(key, 0) != 0) {
      continue;
    }
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

Result<IndexPartitionDebug> BuildPartitionDebug(const std::string& route_name,
                                                const std::shared_ptr<IVFIndex>& ivf,
                                                const VersionSet& versions) {
  if (!ivf) {
    return Status::InvalidArgument("BuildPartitionDebug: null ivf for route " + route_name);
  }
  auto sizes_res = ivf->GetPartitionSizes(versions);
  if (!sizes_res.ok()) {
    return sizes_res.status();
  }
  const std::vector<uint32_t>& partition_sizes = sizes_res.value();
  IndexPartitionDebug out;
  out.route_name = route_name;
  out.nlist = static_cast<uint32_t>(partition_sizes.size());
  std::vector<double> values;
  values.reserve(partition_sizes.size());
  for (uint32_t size : partition_sizes) {
    values.push_back(static_cast<double>(size));
    out.total_docs += static_cast<uint64_t>(size);
    if (size > 0) {
      out.non_empty_lists++;
    }
  }
  out.list_size = SummarizeDistribution(values);
  return out;
}

MatrixRM GenerateRandom(uint32_t rows, uint32_t cols, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  MatrixRM m(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      m(r, c) = dist(gen);
    }
  }
  return m;
}

void NormalizeRowsL2(MatrixRM* X) {
  if (X == nullptr) {
    return;
  }
  for (Eigen::Index i = 0; i < X->rows(); ++i) {
    const float norm = X->row(i).norm();
    if (norm > 0.0f) {
      X->row(i) /= norm;
    }
  }
}

void NormalizeVectorL2(Eigen::VectorXf* x) {
  if (x == nullptr) {
    return;
  }
  const float norm = x->norm();
  if (norm > 0.0f) {
    *x /= norm;
  }
}

uint32_t ResolveMainRows(const Config& config, uint32_t total_rows) {
  if (total_rows == 0) {
    return 0;
  }
  if (!config.enable_streaming) {
    return total_rows;
  }
  uint32_t main_rows = config.main_index_rows;
  if (main_rows == 0) {
    main_rows = std::max<uint32_t>(1, total_rows / 2);
  }
  if (total_rows > 1 && main_rows >= total_rows) {
    main_rows = total_rows - 1;
  }
  return std::max<uint32_t>(1, main_rows);
}

uint32_t ResolveDeltaTrainRows(const Config& config,
                               uint32_t available_stream_rows,
                               uint32_t insert_step,
                               bool use_stream_batch_size) {
  if (!config.enable_streaming || available_stream_rows == 0) {
    return 0;
  }
  // In batch insertion mode, delta_train_window is interpreted as number of batches.
  const uint64_t row_factor = use_stream_batch_size ? static_cast<uint64_t>(insert_step) : 1ull;
  const uint64_t requested_rows_u64 = static_cast<uint64_t>(config.delta_train_window) * row_factor;
  const uint32_t requested_rows = requested_rows_u64 >= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                                      ? std::numeric_limits<uint32_t>::max()
                                      : static_cast<uint32_t>(requested_rows_u64);
  return std::min<uint32_t>(available_stream_rows, requested_rows);
}

struct RuntimeOptions {
  std::string config_path{"configs/sift/sift.json"};
  std::optional<std::string> dataset_spec;
  std::optional<std::string> query_spec;
  std::optional<std::string> prebuilt_index_dir;
  bool use_default_prebuilt_index{false};
  bool fresh_index{false};
};

std::string SanitizePathPart(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) || ch == '-' || ch == '_') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "default" : out;
}

std::string DefaultPrebuiltIndexDir(const std::string& config_path,
                                    const std::string& dataset_label) {
  const std::filesystem::path config_fs_path(config_path);
  const std::string config_stem =
      config_fs_path.stem().string().empty() ? "config" : config_fs_path.stem().string();
  return (std::filesystem::path("index") / SanitizePathPart(dataset_label) /
          SanitizePathPart(config_stem))
      .string();
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [config.json] [dataset_dir_or_base.fvecs] [query_dir_or_query.fvecs] "
               "[--use-prebuilt-index | --prebuilt-index DIR | --fresh-index]\n";
}

Result<RuntimeOptions> ParseRuntimeOptions(int argc, char** argv) {
  RuntimeOptions opts;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return Status::InvalidArgument("help requested");
    }
    if (arg == "--use-prebuilt-index") {
      opts.use_default_prebuilt_index = true;
      continue;
    }
    if (arg == "--prebuilt-index" || arg == "--index") {
      if (i + 1 >= argc) {
        return Status::InvalidArgument(arg + " requires a directory path");
      }
      opts.prebuilt_index_dir = argv[++i];
      continue;
    }
    if (arg == "--fresh-index") {
      opts.fresh_index = true;
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      return Status::InvalidArgument("Unknown option: " + arg);
    }
    positional.push_back(arg);
  }

  if (positional.size() > 3) {
    return Status::InvalidArgument("Too many positional arguments");
  }
  if (!positional.empty()) {
    opts.config_path = positional[0];
  }
  if (positional.size() > 1) {
    opts.dataset_spec = positional[1];
  }
  if (positional.size() > 2) {
    opts.query_spec = positional[2];
  }
  if (opts.fresh_index &&
      (opts.use_default_prebuilt_index || opts.prebuilt_index_dir.has_value())) {
    return Status::InvalidArgument("--fresh-index cannot be combined with prebuilt index options");
  }
  if (opts.use_default_prebuilt_index && opts.prebuilt_index_dir.has_value()) {
    return Status::InvalidArgument(
        "--use-prebuilt-index cannot be combined with --prebuilt-index");
  }
  return opts;
}

double ComputeNonEmptyListImbalance(const std::vector<uint32_t>& sizes,
                                    uint32_t* non_empty_out = nullptr,
                                    uint32_t* max_list_out = nullptr,
                                    double* avg_non_empty_out = nullptr) {
  uint32_t non_empty = 0;
  uint32_t max_list = 0;
  uint64_t non_empty_total = 0;
  for (uint32_t sz : sizes) {
    if (sz > 0) {
      non_empty++;
      non_empty_total += static_cast<uint64_t>(sz);
    }
    max_list = std::max<uint32_t>(max_list, sz);
  }
  const double avg_non_empty =
      non_empty > 0 ? static_cast<double>(non_empty_total) / static_cast<double>(non_empty) : 0.0;
  if (non_empty_out != nullptr) {
    *non_empty_out = non_empty;
  }
  if (max_list_out != nullptr) {
    *max_list_out = max_list;
  }
  if (avg_non_empty_out != nullptr) {
    *avg_non_empty_out = avg_non_empty;
  }
  if (non_empty == 0) {
    return 0.0;
  }
  return static_cast<double>(max_list) / std::max(1.0, avg_non_empty);
}

struct DeltaShard {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint32_t rows{0};
  uint32_t shard_id{0};
};

struct EvalMetrics {
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

RecallAgeMetrics ComputeRecallByRecentInsert(const std::vector<std::vector<DocId>>& gt,
                                             const std::vector<std::vector<DocId>>& pred,
                                             uint32_t new_begin,
                                             uint32_t new_end,
                                             uint32_t seen_rows) {
  RecallAgeMetrics out;
  if (new_begin > new_end || new_end > seen_rows) {
    new_begin = seen_rows;
    new_end = seen_rows;
  }
  const size_t nq = std::min(gt.size(), pred.size());
  for (size_t qi = 0; qi < nq; ++qi) {
    std::unordered_set<DocId> pred_set;
    pred_set.reserve(pred[qi].size() * 2 + 1);
    for (DocId doc : pred[qi]) {
      pred_set.insert(doc);
    }
    for (DocId doc : gt[qi]) {
      if (doc >= seen_rows) {
        continue;
      }
      const bool is_new = doc >= new_begin && doc < new_end;
      const bool hit = pred_set.find(doc) != pred_set.end();
      if (is_new) {
        out.gt_new_total++;
        if (hit) out.hit_new_total++;
      } else {
        out.gt_old_total++;
        if (hit) out.hit_old_total++;
      }
    }
  }
  const uint64_t gt_total = out.gt_new_total + out.gt_old_total;
  out.recall_new = out.gt_new_total > 0
                       ? static_cast<double>(out.hit_new_total) /
                             static_cast<double>(out.gt_new_total)
                       : 0.0;
  out.recall_old = out.gt_old_total > 0
                       ? static_cast<double>(out.hit_old_total) /
                             static_cast<double>(out.gt_old_total)
                       : 0.0;
  out.gt_new_ratio = gt_total > 0
                         ? static_cast<double>(out.gt_new_total) /
                               static_cast<double>(gt_total)
                         : 0.0;
  return out;
}

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

uint32_t ResolveRouteExactRerankCandidates(const Config& config, const SearchRoute& route) {
  uint32_t candidates = config.exact_rerank_candidates_per_route;
  if (route.name == "main" && config.main_exact_rerank_candidates > 0) {
    candidates = config.main_exact_rerank_candidates;
  } else if (route.name == "active_delta" && config.active_exact_rerank_candidates > 0) {
    candidates = config.active_exact_rerank_candidates;
  } else if (route.name == "frozen_delta" && config.frozen_exact_rerank_candidates > 0) {
    candidates = config.frozen_exact_rerank_candidates;
  }
  return std::max<uint32_t>(1u, candidates);
}

struct OnlinePQRollup {
  uint32_t batches{0};
  uint32_t warmup_batches{0};
  uint32_t triggered{0};
  uint32_t updated{0};
  uint32_t reencoded{0};
  uint32_t updated_subspaces{0};
  uint32_t updated_codewords{0};
  double sum_nqe_batch{0.0};
  double sum_qe_ratio{0.0};
  double sum_codebook_drift{0.0};
  double last_nqe_batch{0.0};
  double last_qe_ratio{1.0};
  double last_codebook_drift{0.0};
  uint32_t last_warmup_batches_left{0};
};

Status AddRangeToIndex(const std::shared_ptr<IVFIndex>& ivf,
                       const MatrixRM& x_whitened,
                       uint32_t begin,
                       uint32_t end,
                       uint32_t dim,
                       const VersionSet& versions) {
  if (!ivf) {
    return Status::InvalidArgument("AddRangeToIndex: null ivf");
  }
  if (begin > end || end > static_cast<uint32_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddRangeToIndex: invalid range");
  }
  if (begin == end) {
    return Status::OK();
  }

  for (uint32_t block_begin = begin; block_begin < end;) {
    const uint32_t block_end =
        block_begin + std::min<uint32_t>(kAddBlockRows, end - block_begin);
    AlignedVector<VectorRecord> records;
    records.reserve(static_cast<size_t>(block_end - block_begin));
    for (uint32_t i = block_begin; i < block_end; ++i) {
      VectorRecord rec;
      rec.doc_id = i;
      rec.dim = dim;
      rec.versions = versions;
      rec.ivf_id = 0;
      rec.x = x_whitened.row(static_cast<int64_t>(i)).transpose();
      records.push_back(std::move(rec));
    }
    const Status add_status = ivf->AddBatch(records, block_end == end);
    if (!add_status.ok()) {
      return add_status;
    }
    block_begin = block_end;
  }
  return Status::OK();
}

Result<OnlinePQUpdateStats> AddRangeToIndexWithOnlinePQ(
    const std::shared_ptr<IVFIndex>& ivf,
    const MatrixRM& x_whitened,
    uint32_t begin,
    uint32_t end,
    uint32_t dim,
    const VersionSet& versions,
    const OnlinePQUpdateOptions& options) {
  if (!ivf) {
    return Status::InvalidArgument("AddRangeToIndexWithOnlinePQ: null ivf");
  }
  if (begin > end || end > static_cast<uint32_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddRangeToIndexWithOnlinePQ: invalid range");
  }
  OnlinePQUpdateStats empty_stats;
  if (begin == end) {
    return empty_stats;
  }

  AlignedVector<VectorRecord> records;
  records.resize(static_cast<size_t>(end - begin));
  Timer record_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t offset = 0; offset < static_cast<int64_t>(records.size()); ++offset) {
    const uint32_t i = begin + static_cast<uint32_t>(offset);
    VectorRecord rec;
    rec.doc_id = i;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<int64_t>(i)).transpose();
    records[static_cast<size_t>(offset)] = std::move(rec);
  }
  const double record_build_ms = record_timer.ElapsedMillis();
  auto add_res = ivf->AddWithOnlinePQ(records, options);
  if (!add_res.ok()) {
    return add_res.status();
  }
  OnlinePQUpdateStats stats = add_res.value();
  stats.record_build_ms += record_build_ms;
  stats.insert_ms += record_build_ms;
  return stats;
}

Result<OnlinePQUpdateStats> AddRangeToIndexWithOnlinePQSlidingWindow(
    const std::shared_ptr<IVFIndex>& ivf,
    const MatrixRM& x_whitened,
    uint32_t begin,
    uint32_t end,
    uint32_t dim,
    const VersionSet& versions,
    const std::vector<DocId>& delete_doc_ids,
    const OnlinePQUpdateOptions& options) {
  if (!ivf) {
    return Status::InvalidArgument("AddRangeToIndexWithOnlinePQSlidingWindow: null ivf");
  }
  if (begin > end || end > static_cast<uint32_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddRangeToIndexWithOnlinePQSlidingWindow: invalid range");
  }

  AlignedVector<VectorRecord> records;
  records.resize(static_cast<size_t>(end - begin));
  Timer record_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t offset = 0; offset < static_cast<int64_t>(records.size()); ++offset) {
    const uint32_t i = begin + static_cast<uint32_t>(offset);
    VectorRecord rec;
    rec.doc_id = i;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<int64_t>(i)).transpose();
    records[static_cast<size_t>(offset)] = std::move(rec);
  }
  const double record_build_ms = record_timer.ElapsedMillis();
  AlignedVector<VectorRecord> delete_records;
  delete_records.reserve(delete_doc_ids.size());
  for (DocId doc_id : delete_doc_ids) {
    if (static_cast<Eigen::Index>(doc_id) >= x_whitened.rows()) {
      return Status::InvalidArgument(
          "AddRangeToIndexWithOnlinePQSlidingWindow: delete doc_id out of range");
    }
    VectorRecord rec;
    rec.doc_id = doc_id;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<Eigen::Index>(doc_id)).transpose();
    delete_records.push_back(std::move(rec));
  }
  auto add_res = ivf->AddWithOnlinePQSlidingWindowRecords(records, delete_records, options);
  if (!add_res.ok()) {
    return add_res.status();
  }
  OnlinePQUpdateStats stats = add_res.value();
  stats.record_build_ms += record_build_ms;
  stats.insert_ms += record_build_ms;
  return stats;
}

Result<DeltaShard> BuildDeltaShard(Eigen::Ref<const MatrixRM> train_data,
                                   const IVFParams& ivf_params,
                                   VersionId whiten_version,
                                   uint32_t shard_id) {
  if (train_data.rows() == 0 || train_data.cols() == 0) {
    return Status::InvalidArgument("BuildDeltaShard: empty training data");
  }
  auto ivf = CreateIVFIndex();
  std::vector<DocId> ids(static_cast<size_t>(train_data.rows()));
  std::iota(ids.begin(), ids.end(), 0);
  auto version_res = ivf->Build(train_data, ids, ivf_params, 0);
  if (!version_res.ok()) {
    return version_res.status();
  }
  DeltaShard shard;
  shard.ivf = ivf;
  shard.versions = VersionSet{whiten_version, version_res.value()};
  shard.rows = 0;
  shard.shard_id = shard_id;
  return shard;
}

struct ExactDocCandidate {
  float dist{0.0f};
  DocId doc_id{0};
};

bool ExactDocBetter(const ExactDocCandidate& a, const ExactDocCandidate& b) {
  if (a.dist != b.dist) {
    return a.dist < b.dist;
  }
  return a.doc_id < b.doc_id;
}

size_t ResolveGroundTruthBlockRows(Eigen::Index nq, Eigen::Index dim, size_t total_docs) {
  if (total_docs == 0) {
    return 0;
  }
  const size_t nq_size = std::max<size_t>(1, static_cast<size_t>(nq));
  const size_t dim_size = std::max<size_t>(1, static_cast<size_t>(dim));
  const size_t bytes_per_row = (nq_size + dim_size + 1) * sizeof(float);
  const size_t rows_by_budget =
      std::max<size_t>(1, kGroundTruthBlockTargetBytes / std::max<size_t>(1, bytes_per_row));
  return std::min(total_docs, rows_by_budget);
}

Result<std::vector<std::vector<DocId>>> ExactSearchDocIdsBlockwise(
    Eigen::Ref<const MatrixRM> queries,
    Eigen::Ref<const MatrixRM> database,
    const std::vector<DocId>& doc_ids,
    uint32_t topk) {
  if (queries.cols() == 0 || database.cols() == 0) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwise: empty matrices");
  }
  if (queries.cols() != database.cols()) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwise: dimension mismatch");
  }
  if (queries.rows() == 0) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwise: no queries");
  }
  if (doc_ids.empty()) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwise: no searchable docs");
  }
  if (topk == 0) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwise: topk must be positive");
  }

  const size_t limit = std::min<size_t>(topk, doc_ids.size());
  std::vector<std::vector<ExactDocCandidate>> heaps(static_cast<size_t>(queries.rows()));
  for (auto& heap : heaps) {
    heap.reserve(limit);
  }
  const Eigen::VectorXf query_norms = queries.rowwise().squaredNorm();
  const size_t block_rows =
      ResolveGroundTruthBlockRows(queries.rows(), queries.cols(), doc_ids.size());

  for (size_t block_begin = 0; block_begin < doc_ids.size(); block_begin += block_rows) {
    const size_t block_count = std::min(block_rows, doc_ids.size() - block_begin);
    MatrixRM block(static_cast<Eigen::Index>(block_count), database.cols());
    Eigen::VectorXf block_norms(static_cast<Eigen::Index>(block_count));
    for (size_t bi = 0; bi < block_count; ++bi) {
      const DocId doc_id = doc_ids[block_begin + bi];
      if (static_cast<Eigen::Index>(doc_id) >= database.rows()) {
        return Status::InvalidArgument("ExactSearchDocIdsBlockwise: doc_id out of database range");
      }
      block.row(static_cast<Eigen::Index>(bi)) =
          database.row(static_cast<Eigen::Index>(doc_id));
      block_norms(static_cast<Eigen::Index>(bi)) =
          block.row(static_cast<Eigen::Index>(bi)).squaredNorm();
    }

    const MatrixRM dots = queries * block.transpose();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t qi = 0; qi < static_cast<int64_t>(queries.rows()); ++qi) {
      auto& heap = heaps[static_cast<size_t>(qi)];
      const float qnorm = query_norms(static_cast<Eigen::Index>(qi));
      for (size_t bi = 0; bi < block_count; ++bi) {
        float dist = qnorm + block_norms(static_cast<Eigen::Index>(bi)) -
                     2.0f * dots(static_cast<Eigen::Index>(qi),
                                  static_cast<Eigen::Index>(bi));
        if (dist < 0.0f) {
          dist = 0.0f;
        }
        ExactDocCandidate cand{dist, doc_ids[block_begin + bi]};
        if (heap.size() < limit) {
          heap.push_back(cand);
          std::push_heap(heap.begin(), heap.end(), ExactDocBetter);
        } else if (ExactDocBetter(cand, heap.front())) {
          std::pop_heap(heap.begin(), heap.end(), ExactDocBetter);
          heap.back() = cand;
          std::push_heap(heap.begin(), heap.end(), ExactDocBetter);
        }
      }
    }
  }

  std::vector<std::vector<DocId>> all_ids(static_cast<size_t>(queries.rows()));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t qi = 0; qi < static_cast<int64_t>(queries.rows()); ++qi) {
    auto& heap = heaps[static_cast<size_t>(qi)];
    std::sort(heap.begin(), heap.end(), ExactDocBetter);
    std::vector<DocId> ids;
    ids.reserve(heap.size());
    for (const auto& cand : heap) {
      ids.push_back(cand.doc_id);
    }
    all_ids[static_cast<size_t>(qi)] = std::move(ids);
  }
  return all_ids;
}

Result<SearchResult> MergeTopK(const std::vector<SearchResult>& partial_results,
                               uint32_t topk,
                               bool exact_rerank_enable,
                               const std::vector<uint32_t>& rerank_candidates_per_route,
                               Eigen::Ref<const Eigen::VectorXf> query_whitened,
                               const MatrixRM& base_whitened) {
  SearchResult out;
  if (partial_results.empty() || topk == 0) {
    return out;
  }
  std::unordered_map<DocId, Candidate> best_by_doc;
  uint64_t scanned = 0;
  for (size_t ri = 0; ri < partial_results.size(); ++ri) {
    const auto& part = partial_results[ri];
    scanned += part.scanned_candidates;
    size_t route_take = part.topk.size();
    if (exact_rerank_enable) {
      const uint32_t route_cap = ri < rerank_candidates_per_route.size()
                                     ? rerank_candidates_per_route[ri]
                                     : static_cast<uint32_t>(route_take);
      route_take = std::min(route_take, static_cast<size_t>(route_cap));
    }
    for (size_t i = 0; i < route_take; ++i) {
      const auto& cand = part.topk[i];
      auto it = best_by_doc.find(cand.doc_id);
      if (it == best_by_doc.end() || cand.approx_dist < it->second.approx_dist) {
        best_by_doc[cand.doc_id] = cand;
      }
    }
  }

  std::vector<Candidate> merged;
  merged.reserve(best_by_doc.size());
  for (const auto& kv : best_by_doc) {
    merged.push_back(kv.second);
  }

  auto approx_less = [](const Candidate& a, const Candidate& b) {
    if (a.approx_dist != b.approx_dist) {
      return a.approx_dist < b.approx_dist;
    }
    return a.doc_id < b.doc_id;
  };
  auto rerank_less = [](const Candidate& a, const Candidate& b) {
    if (a.rerank_dist != b.rerank_dist) {
      return a.rerank_dist < b.rerank_dist;
    }
    if (a.approx_dist != b.approx_dist) {
      return a.approx_dist < b.approx_dist;
    }
    return a.doc_id < b.doc_id;
  };

  if (exact_rerank_enable) {
    for (const auto& cand : merged) {
      if (cand.doc_id >= static_cast<DocId>(base_whitened.rows())) {
        return Status::InvalidArgument("MergeTopK: doc id out of range for exact rerank");
      }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && merged.size() > 1)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(merged.size()); ++i) {
      auto& cand = merged[static_cast<size_t>(i)];
      const Eigen::VectorXf diff =
          query_whitened - base_whitened.row(static_cast<Eigen::Index>(cand.doc_id)).transpose();
      cand.rerank_dist = diff.squaredNorm();
    }
    if (merged.size() > topk) {
      std::nth_element(merged.begin(),
                       merged.begin() + static_cast<int64_t>(topk),
                       merged.end(),
                       rerank_less);
      merged.resize(topk);
    }
    std::sort(merged.begin(), merged.end(), rerank_less);
  } else {
    if (merged.size() > topk) {
      std::nth_element(merged.begin(),
                       merged.begin() + static_cast<int64_t>(topk),
                       merged.end(),
                       approx_less);
      merged.resize(topk);
    }
    std::sort(merged.begin(), merged.end(), approx_less);
  }

  out.topk = std::move(merged);
  out.scanned_candidates = scanned;
  return out;
}

Result<EvalMetrics> EvaluateState(const Config& config,
                                  const MatrixRM& base_whitened,
                                  uint32_t seen_rows,
                                  uint32_t new_begin,
                                  uint32_t new_end,
                                  uint32_t main_rows,
                                  const MatrixRM& queries_raw,
                                  const MatrixRM& queries_whitened,
                                  const std::shared_ptr<WhiteningModel>& whitening,
                                  VersionId whitening_version,
                                  const std::shared_ptr<IVFIndex>& main_ivf,
                                  const VersionSet& main_versions,
                                  const std::optional<DeltaShard>& frozen_delta,
                                  const std::optional<DeltaShard>& active_delta,
                                  const SearchParams& params) {
  if (seen_rows == 0 || seen_rows > static_cast<uint32_t>(base_whitened.rows())) {
    return Status::InvalidArgument("EvaluateState: invalid seen_rows");
  }
  if (main_rows == 0 || main_rows > seen_rows) {
    return Status::InvalidArgument("EvaluateState: invalid main_rows");
  }
  if (!main_ivf) {
    return Status::InvalidArgument("EvaluateState: main index is null");
  }

  auto main_doc_ids_res = main_ivf->SnapshotDocIds(main_versions);
  if (!main_doc_ids_res.ok()) {
    return main_doc_ids_res.status();
  }

  std::vector<DocId> searchable_doc_ids;
  searchable_doc_ids.reserve(main_doc_ids_res.value().size() +
                             (frozen_delta.has_value() ? frozen_delta->rows : 0u) +
                             (active_delta.has_value() ? active_delta->rows : 0u));
  std::unordered_set<DocId> searchable_seen;
  searchable_seen.reserve(searchable_doc_ids.capacity() * 2 + 1);
  auto append_unique = [&](const std::vector<DocId>& doc_ids) -> Status {
    for (DocId doc_id : doc_ids) {
      if (doc_id >= seen_rows) {
        return Status::InvalidArgument("EvaluateState: route doc_id exceeds seen_rows");
      }
      if (searchable_seen.insert(doc_id).second) {
        searchable_doc_ids.push_back(doc_id);
      }
    }
    return Status::OK();
  };

  Status append_main = append_unique(main_doc_ids_res.value());
  if (!append_main.ok()) {
    return append_main;
  }
  if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
    auto frozen_doc_ids_res = frozen_delta->ivf->SnapshotDocIds(frozen_delta->versions);
    if (!frozen_doc_ids_res.ok()) {
      return frozen_doc_ids_res.status();
    }
    Status append_frozen = append_unique(frozen_doc_ids_res.value());
    if (!append_frozen.ok()) {
      return append_frozen;
    }
  }
  if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
    auto active_doc_ids_res = active_delta->ivf->SnapshotDocIds(active_delta->versions);
    if (!active_doc_ids_res.ok()) {
      return active_doc_ids_res.status();
    }
    Status append_active = append_unique(active_doc_ids_res.value());
    if (!append_active.ok()) {
      return append_active;
    }
  }
  if (searchable_doc_ids.empty()) {
    return Status::InvalidArgument("EvaluateState: no searchable docs");
  }

  auto gt_res = ExactSearchDocIdsBlockwise(
      queries_whitened, base_whitened, searchable_doc_ids, config.topk);
  if (!gt_res.ok()) {
    return gt_res.status();
  }
  std::vector<std::vector<DocId>> ground_truth = std::move(gt_res.value());

  std::vector<SearchRoute> routes;
  routes.push_back(SearchRoute{main_ivf, main_versions, 0, "main", main_rows});
  if (!config.main_query_only) {
    if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
      routes.push_back(
          SearchRoute{frozen_delta->ivf, frozen_delta->versions, 1, "frozen_delta", frozen_delta->rows});
    }
    if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
      routes.push_back(
          SearchRoute{active_delta->ivf, active_delta->versions, 1, "active_delta", active_delta->rows});
    }
  }

  const uint32_t nq = static_cast<uint32_t>(queries_raw.rows());
  const uint32_t route_count = static_cast<uint32_t>(routes.size());
  const bool enable_exact_rerank = config.exact_rerank_enable;
  std::vector<uint32_t> route_nprobes(route_count, std::max<uint32_t>(1u, params.nprobe));
  std::vector<uint32_t> route_topks(route_count, std::max<uint32_t>(1u, params.topk));
  std::vector<uint32_t> route_rerank_candidates(route_count,
                                                 std::max<uint32_t>(1u, params.topk));
  for (uint32_t ri = 0; ri < route_count; ++ri) {
    if (enable_exact_rerank) {
      route_rerank_candidates[static_cast<size_t>(ri)] =
          ResolveRouteExactRerankCandidates(config, routes[static_cast<size_t>(ri)]);
      route_topks[static_cast<size_t>(ri)] =
          std::max(params.topk, route_rerank_candidates[static_cast<size_t>(ri)]);
    }
  }
  std::vector<std::vector<DocId>> predictions(nq);
  std::vector<double> whitening_ms(nq, 0.0);
  std::vector<double> search_ms(nq, 0.0);
  std::vector<double> total_ms(nq, 0.0);
  std::vector<double> scanned_counts(nq, 0.0);

  std::atomic<bool> failed{false};
  std::mutex error_mu;
  Status error_status;

  MatrixRM queries_whitened_runtime(queries_raw.rows(), queries_raw.cols());
  std::vector<std::vector<SearchResult>> route_results(
      nq, std::vector<SearchResult>(route_count));
  std::vector<double> route_search_ms(static_cast<size_t>(nq) * route_count, 0.0);
  std::vector<double> route_scanned_candidates(static_cast<size_t>(nq) * route_count, 0.0);
  std::vector<double> merge_topk_ms(nq, 0.0);
  std::vector<double> max_route_search_ms(nq, 0.0);
  std::vector<double> route_wall_ms(route_count, 0.0);
  double whitening_wall_ms = 0.0;
  double merge_wall_ms = 0.0;

  Timer whitening_wall_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < static_cast<int64_t>(nq); ++i) {
    if (failed.load()) {
      continue;
    }
    Eigen::VectorXf qvec = queries_raw.row(i).transpose();
    Timer wtimer;
    Eigen::VectorXf qbuf(qvec.size());
    auto wstatus = whitening->Transform(qvec, whitening_version, qbuf);
    const double whiten_elapsed = wtimer.ElapsedMillis();
    if (!wstatus.ok()) {
      std::lock_guard<std::mutex> lock(error_mu);
      if (!failed.exchange(true)) {
        error_status = wstatus.status();
      }
      continue;
    }
    if (config.use_cosine) {
      NormalizeVectorL2(&qbuf);
    }
    queries_whitened_runtime.row(i) = qbuf.transpose();
    whitening_ms[static_cast<size_t>(i)] = whiten_elapsed;
  }
  whitening_wall_ms = whitening_wall_timer.ElapsedMillis();

  if (failed.load()) {
    return error_status;
  }

  for (uint32_t ri = 0; ri < route_count; ++ri) {
    Timer route_timer;
    auto sres_batch = routes[static_cast<size_t>(ri)].ivf->SearchBatch(
        queries_whitened_runtime,
        route_topks[static_cast<size_t>(ri)],
        route_nprobes[static_cast<size_t>(ri)],
        routes[static_cast<size_t>(ri)].versions,
        routes[static_cast<size_t>(ri)].from_new,
        config.enable_miss_diag);
    route_wall_ms[static_cast<size_t>(ri)] = route_timer.ElapsedMillis();
    if (!sres_batch.ok()) {
      return sres_batch.status();
    }
    std::vector<SearchResult> route_batch = std::move(sres_batch.value());
    if (route_batch.size() != static_cast<size_t>(nq)) {
      return Status::InvalidArgument("EvaluateState: route batch result size mismatch");
    }
    const double per_query_route_ms =
        nq > 0 ? (route_wall_ms[static_cast<size_t>(ri)] / static_cast<double>(nq)) : 0.0;
    for (uint32_t qi = 0; qi < nq; ++qi) {
      const size_t idx = static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri);
      route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)] =
          std::move(route_batch[static_cast<size_t>(qi)]);
      route_scanned_candidates[idx] = static_cast<double>(
          route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)].scanned_candidates);
      route_search_ms[idx] = per_query_route_ms;
    }
  }

  uint32_t main_route_queries = nq;
  uint32_t active_delta_route_queries = 0;
  uint32_t frozen_delta_route_queries = 0;
  for (uint32_t ri = 1; ri < route_count; ++ri) {
    if (routes[static_cast<size_t>(ri)].name == "active_delta") {
      active_delta_route_queries = nq;
    } else if (routes[static_cast<size_t>(ri)].name == "frozen_delta") {
      frozen_delta_route_queries = nq;
    }
  }

  Timer merge_wall_timer;
  std::atomic<uint64_t> rerank_topk_main_total_atomic{0};
  std::atomic<uint64_t> rerank_topk_delta_total_atomic{0};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
    if (failed.load()) {
      continue;
    }
    Eigen::VectorXf q = queries_whitened_runtime.row(static_cast<Eigen::Index>(qi)).transpose();
    Timer merge_timer;
    auto merged_res = MergeTopK(route_results[static_cast<size_t>(qi)],
                                params.topk,
                                enable_exact_rerank,
                                route_rerank_candidates,
                                q,
                                base_whitened);
    if (!merged_res.ok()) {
      std::lock_guard<std::mutex> lock(error_mu);
      if (!failed.exchange(true)) {
        error_status = merged_res.status();
      }
      continue;
    }
    SearchResult merged = merged_res.value();
    const double merge_elapsed = merge_timer.ElapsedMillis();
    double route_search_max = 0.0;
    for (uint32_t ri = 0; ri < route_count; ++ri) {
      route_search_max = std::max(
          route_search_max,
          route_search_ms[static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri)]);
    }
    const double q_search_ms = route_search_max + merge_elapsed;
    max_route_search_ms[static_cast<size_t>(qi)] = route_search_max;
    merge_topk_ms[static_cast<size_t>(qi)] = merge_elapsed;
    std::vector<DocId> row;
    row.reserve(merged.topk.size());
    for (const auto& cand : merged.topk) {
      row.push_back(cand.doc_id);
      if (config.enable_rerank_source_diag) {
        if (cand.from_new == 1) {
          rerank_topk_delta_total_atomic.fetch_add(1, std::memory_order_relaxed);
        } else {
          rerank_topk_main_total_atomic.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    predictions[static_cast<size_t>(qi)] = std::move(row);
    search_ms[static_cast<size_t>(qi)] = q_search_ms;
    total_ms[static_cast<size_t>(qi)] = whitening_ms[static_cast<size_t>(qi)] + q_search_ms;
    scanned_counts[static_cast<size_t>(qi)] = static_cast<double>(merged.scanned_candidates);
  }
  merge_wall_ms = merge_wall_timer.ElapsedMillis();
  if (failed.load()) {
    return error_status;
  }
  const uint64_t rerank_topk_main_total =
      rerank_topk_main_total_atomic.load(std::memory_order_relaxed);
  const uint64_t rerank_topk_delta_total =
      rerank_topk_delta_total_atomic.load(std::memory_order_relaxed);

  uint64_t gt_total = 0;
  uint64_t gt_probed = 0;
  uint64_t gt_hit_and_probed = 0;
  uint64_t miss_not_probed = 0;
  uint64_t miss_pq_filtered = 0;
  double exact_recall_sum = 0.0;
  uint32_t exact_recall_count = 0;
  double pq_rank_loss_sum = 0.0;
  uint64_t pq_rank_loss_count = 0;
  std::vector<std::string> worst_queries;
  if (config.enable_miss_diag) {
    struct QueryMissDiag {
      double recall{0.0};
      uint32_t misses{0};
      std::string summary;
    };
    std::vector<QueryMissDiag> query_diags;
    query_diags.reserve(nq);

    for (uint32_t qi = 0; qi < nq; ++qi) {
      const auto& gt_row = ground_truth[static_cast<size_t>(qi)];
      if (gt_row.empty()) {
        continue;
      }
      const auto& pred_row = predictions[static_cast<size_t>(qi)];
      std::unordered_set<DocId> pred_set(pred_row.begin(), pred_row.end());
      size_t scanned_reserve = 0;
      for (uint32_t ri = 0; ri < route_count; ++ri) {
        scanned_reserve +=
            route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)].scanned_doc_ids.size();
      }
      std::unordered_map<DocId, float> approx_by_doc;
      approx_by_doc.reserve(scanned_reserve * 2 + 1);
      for (uint32_t ri = 0; ri < route_count; ++ri) {
        const SearchResult& rr = route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)];
        const size_t n = std::min(rr.scanned_doc_ids.size(), rr.scanned_approx_dists.size());
        for (size_t si = 0; si < n; ++si) {
          const DocId doc = rr.scanned_doc_ids[si];
          const float approx = rr.scanned_approx_dists[si];
          auto it = approx_by_doc.find(doc);
          if (it == approx_by_doc.end() || approx < it->second) {
            approx_by_doc[doc] = approx;
          }
        }
      }

      std::vector<std::pair<float, DocId>> approx_ranked;
      std::vector<std::pair<float, DocId>> exact_ranked;
      approx_ranked.reserve(approx_by_doc.size());
      exact_ranked.reserve(approx_by_doc.size());
      const Eigen::VectorXf qv =
          queries_whitened_runtime.row(static_cast<Eigen::Index>(qi)).transpose();
      for (const auto& kv : approx_by_doc) {
        approx_ranked.push_back({kv.second, kv.first});
        const Eigen::VectorXf diff =
            qv - base_whitened.row(static_cast<Eigen::Index>(kv.first)).transpose();
        exact_ranked.push_back({diff.squaredNorm(), kv.first});
      }
      std::sort(approx_ranked.begin(), approx_ranked.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      std::sort(exact_ranked.begin(), exact_ranked.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

      std::unordered_map<DocId, uint32_t> approx_rank;
      std::unordered_map<DocId, uint32_t> exact_rank;
      approx_rank.reserve(approx_ranked.size() * 2 + 1);
      exact_rank.reserve(exact_ranked.size() * 2 + 1);
      for (size_t i = 0; i < approx_ranked.size(); ++i) {
        approx_rank[approx_ranked[i].second] = static_cast<uint32_t>(i + 1);
      }
      for (size_t i = 0; i < exact_ranked.size(); ++i) {
        exact_rank[exact_ranked[i].second] = static_cast<uint32_t>(i + 1);
      }

      std::unordered_set<DocId> exact_topk_set;
      const size_t exact_topk = std::min<size_t>(params.topk, exact_ranked.size());
      exact_topk_set.reserve(exact_topk * 2 + 1);
      for (size_t i = 0; i < exact_topk; ++i) {
        exact_topk_set.insert(exact_ranked[i].second);
      }
      uint32_t exact_hits_row = 0;
      for (DocId doc : gt_row) {
        if (exact_topk_set.find(doc) != exact_topk_set.end()) {
          exact_hits_row++;
        }
      }
      const double exact_recall_row =
          static_cast<double>(exact_hits_row) / static_cast<double>(gt_row.size());
      exact_recall_sum += exact_recall_row;
      exact_recall_count++;

      uint32_t gt_hits_row = 0;
      uint32_t gt_probed_row = 0;
      uint32_t miss_not_probed_row = 0;
      uint32_t miss_pq_filtered_row = 0;
      uint32_t pq_rank_loss_count_row = 0;
      uint32_t pq_rank_loss_max_row = 0;
      double pq_rank_loss_sum_row = 0.0;

      for (DocId doc : gt_row) {
        gt_total++;
        const bool hit = pred_set.find(doc) != pred_set.end();
        if (hit) {
          gt_hits_row++;
        }
        const auto ait = approx_rank.find(doc);
        const bool probed = ait != approx_rank.end();
        if (probed) {
          gt_probed++;
          gt_probed_row++;
          if (hit) {
            gt_hit_and_probed++;
          }
        }
        if (!hit) {
          if (!probed) {
            miss_not_probed++;
            miss_not_probed_row++;
          } else {
            miss_pq_filtered++;
            miss_pq_filtered_row++;
            const auto eit = exact_rank.find(doc);
            if (eit != exact_rank.end()) {
              const uint32_t loss =
                  (ait->second > eit->second) ? (ait->second - eit->second) : 0u;
              pq_rank_loss_sum += static_cast<double>(loss);
              pq_rank_loss_count++;
              pq_rank_loss_sum_row += static_cast<double>(loss);
              pq_rank_loss_count_row++;
              pq_rank_loss_max_row = std::max(pq_rank_loss_max_row, loss);
            }
          }
        }
      }

      const double recall_row =
          static_cast<double>(gt_hits_row) / static_cast<double>(gt_row.size());
      if (miss_not_probed_row + miss_pq_filtered_row > 0) {
        const double gt_probed_rate_row =
            static_cast<double>(gt_probed_row) / static_cast<double>(gt_row.size());
        const double pq_rank_loss_avg_row =
            pq_rank_loss_count_row > 0
                ? (pq_rank_loss_sum_row / static_cast<double>(pq_rank_loss_count_row))
                : 0.0;
        std::string summary =
            "qid=" + std::to_string(qi) + ", recall=" + std::to_string(recall_row) +
            ", gt_probed_rate=" + std::to_string(gt_probed_rate_row) +
            ", miss_not_probed=" + std::to_string(miss_not_probed_row) +
            ", miss_pq_filtered=" + std::to_string(miss_pq_filtered_row) +
            ", exact_recall_on_probed=" + std::to_string(exact_recall_row) +
            ", avg_pq_rank_loss=" + std::to_string(pq_rank_loss_avg_row) +
            ", max_pq_rank_loss=" + std::to_string(pq_rank_loss_max_row);
        query_diags.push_back(QueryMissDiag{
            recall_row,
            miss_not_probed_row + miss_pq_filtered_row,
            std::move(summary),
        });
      }
    }

    std::sort(query_diags.begin(), query_diags.end(),
              [](const QueryMissDiag& a, const QueryMissDiag& b) {
                if (a.recall != b.recall) {
                  return a.recall < b.recall;
                }
                return a.misses > b.misses;
              });
    const size_t keep = std::min<size_t>(kWorstQueryDiagCount, query_diags.size());
    worst_queries.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
      worst_queries.push_back(query_diags[i].summary);
    }
  }

  const double slowest_route_wall_ms = route_wall_ms.empty()
                                           ? 0.0
                                           : *std::max_element(route_wall_ms.begin(),
                                                               route_wall_ms.end());
  const double wall_elapsed_no_merge_ms = whitening_wall_ms + slowest_route_wall_ms;
  const double wall_elapsed_with_merge_ms = wall_elapsed_no_merge_ms + merge_wall_ms;

  auto recall_res = RecallAtK(ground_truth, predictions, config.topk);
  if (!recall_res.ok()) {
    return recall_res.status();
  }
  const RecallAgeMetrics age_metrics =
      ComputeRecallByRecentInsert(ground_truth, predictions, new_begin, new_end, seen_rows);
  auto whiten_summary = SummarizeLatencies(whitening_ms);
  if (!whiten_summary.ok()) {
    return whiten_summary.status();
  }
  auto search_summary = SummarizeLatencies(search_ms);
  if (!search_summary.ok()) {
    return search_summary.status();
  }
  auto total_summary = SummarizeLatencies(total_ms);
  if (!total_summary.ok()) {
    return total_summary.status();
  }

  double avg_whiten = 0.0;
  double avg_search = 0.0;
  double avg_total = 0.0;
  if (nq > 0) {
    avg_whiten =
        std::accumulate(whitening_ms.begin(), whitening_ms.end(), 0.0) / static_cast<double>(nq);
    avg_search =
        std::accumulate(search_ms.begin(), search_ms.end(), 0.0) / static_cast<double>(nq);
    avg_total = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / static_cast<double>(nq);
  }

  double scanned_avg = 0.0;
  double scanned_p50 = 0.0;
  double scanned_p99 = 0.0;
  double scanned_max = 0.0;
  if (nq > 0) {
    scanned_avg =
        std::accumulate(scanned_counts.begin(), scanned_counts.end(), 0.0) / static_cast<double>(nq);
    scanned_p50 = Percentile(scanned_counts, 0.50);
    scanned_p99 = Percentile(scanned_counts, 0.99);
    scanned_max = *std::max_element(scanned_counts.begin(), scanned_counts.end());
  }

  const double query_qps = wall_elapsed_with_merge_ms > 0.0
                               ? (1000.0 * static_cast<double>(nq) / wall_elapsed_with_merge_ms)
                               : 0.0;
  const double avg_query_wall_ms =
      nq > 0 ? (wall_elapsed_no_merge_ms / static_cast<double>(nq)) : 0.0;

  EvalMetrics metrics;
  metrics.recall = recall_res.value();
  metrics.recall_new = age_metrics.recall_new;
  metrics.recall_old = age_metrics.recall_old;
  metrics.gt_new_ratio = age_metrics.gt_new_ratio;
  metrics.gt_new_total = age_metrics.gt_new_total;
  metrics.gt_old_total = age_metrics.gt_old_total;
  metrics.hit_new_total = age_metrics.hit_new_total;
  metrics.hit_old_total = age_metrics.hit_old_total;
  metrics.whitening_p50 = whiten_summary.value().p50_ms;
  metrics.whitening_p99 = whiten_summary.value().p99_ms;
  metrics.avg_whiten_ms = avg_whiten;
  metrics.avg_search_ms = avg_search;
  metrics.search_p50 = search_summary.value().p50_ms;
  metrics.search_p99 = search_summary.value().p99_ms;
  metrics.total_p50 = total_summary.value().p50_ms;
  metrics.total_p99 = total_summary.value().p99_ms;
  metrics.avg_query_ms = avg_query_wall_ms;
  metrics.end_to_end_overhead_ms = avg_total;
  metrics.query_qps = query_qps;
  metrics.scanned_avg = scanned_avg;
  metrics.scanned_p50 = scanned_p50;
  metrics.scanned_p99 = scanned_p99;
  metrics.scanned_max = scanned_max;
  metrics.query_eval_ms = wall_elapsed_with_merge_ms;
  metrics.query_count = nq;
  metrics.main_route_queries = main_route_queries;
  metrics.active_delta_route_queries = active_delta_route_queries;
  metrics.frozen_delta_route_queries = frozen_delta_route_queries;
  if (config.enable_rerank_source_diag) {
    metrics.rerank_topk_main_total = rerank_topk_main_total;
    metrics.rerank_topk_delta_total = rerank_topk_delta_total;
    const uint64_t rerank_topk_total = rerank_topk_main_total + rerank_topk_delta_total;
    if (rerank_topk_total > 0) {
      metrics.rerank_topk_main_ratio =
          static_cast<double>(rerank_topk_main_total) / static_cast<double>(rerank_topk_total);
      metrics.rerank_topk_delta_ratio =
          static_cast<double>(rerank_topk_delta_total) / static_cast<double>(rerank_topk_total);
    }
    if (nq > 0) {
      metrics.rerank_topk_main_avg =
          static_cast<double>(rerank_topk_main_total) / static_cast<double>(nq);
      metrics.rerank_topk_delta_avg =
          static_cast<double>(rerank_topk_delta_total) / static_cast<double>(nq);
    }
  }
  if (config.enable_miss_diag) {
    metrics.gt_probed_rate =
        gt_total > 0 ? static_cast<double>(gt_probed) / static_cast<double>(gt_total) : 0.0;
    metrics.recall_on_probed_gt =
        gt_probed > 0 ? static_cast<double>(gt_hit_and_probed) / static_cast<double>(gt_probed)
                      : 0.0;
    metrics.exact_recall_on_probed_candidates =
        exact_recall_count > 0 ? exact_recall_sum / static_cast<double>(exact_recall_count) : 0.0;
    metrics.avg_pq_rank_loss =
        pq_rank_loss_count > 0 ? pq_rank_loss_sum / static_cast<double>(pq_rank_loss_count) : 0.0;
    metrics.miss_not_probed = static_cast<uint32_t>(
        std::min<uint64_t>(miss_not_probed, std::numeric_limits<uint32_t>::max()));
    metrics.miss_probed_filtered_by_pq = static_cast<uint32_t>(
        std::min<uint64_t>(miss_pq_filtered, std::numeric_limits<uint32_t>::max()));
    metrics.pq_rank_loss_count = static_cast<uint32_t>(
        std::min<uint64_t>(pq_rank_loss_count, std::numeric_limits<uint32_t>::max()));
    metrics.worst_queries = std::move(worst_queries);
  }
  if (config.enable_latency_debug) {
    LatencyDebugMetrics debug;
    debug.route_count = route_count;
    debug.merge_ms = SummarizeDistribution(merge_topk_ms);
    debug.max_route_search_ms = SummarizeDistribution(max_route_search_ms);
    debug.routes.reserve(route_count);
    for (uint32_t ri = 0; ri < route_count; ++ri) {
      std::vector<double> route_ms_values;
      std::vector<double> route_scanned_values;
      route_ms_values.reserve(nq);
      route_scanned_values.reserve(nq);
      for (uint32_t qi = 0; qi < nq; ++qi) {
        const size_t idx = static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri);
        route_ms_values.push_back(route_search_ms[idx]);
        route_scanned_values.push_back(route_scanned_candidates[idx]);
      }
      RouteDebugStats route_debug;
      route_debug.route_name = routes[static_cast<size_t>(ri)].name;
      route_debug.from_new = routes[static_cast<size_t>(ri)].from_new;
      route_debug.search_ms = SummarizeDistribution(route_ms_values);
      route_debug.scanned_candidates = SummarizeDistribution(route_scanned_values);
      debug.routes.push_back(std::move(route_debug));
    }

    std::vector<uint32_t> slow_query_order(nq);
    std::iota(slow_query_order.begin(), slow_query_order.end(), 0u);
    std::sort(slow_query_order.begin(),
              slow_query_order.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                return search_ms[static_cast<size_t>(lhs)] >
                       search_ms[static_cast<size_t>(rhs)];
              });
    const size_t keep = std::min<size_t>(kSlowQueryDebugCount, slow_query_order.size());
    debug.slow_queries.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
      const uint32_t qi = slow_query_order[i];
      SlowQueryDebug slow;
      slow.query_id = qi;
      slow.total_search_ms = search_ms[static_cast<size_t>(qi)];
      slow.merge_ms = merge_topk_ms[static_cast<size_t>(qi)];
      slow.max_route_search_ms = max_route_search_ms[static_cast<size_t>(qi)];
      slow.merged_scanned = scanned_counts[static_cast<size_t>(qi)];
      slow.route_search_ms.reserve(route_count);
      slow.route_scanned_candidates.reserve(route_count);
      for (uint32_t ri = 0; ri < route_count; ++ri) {
        const size_t idx = static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri);
        slow.route_search_ms.push_back(route_search_ms[idx]);
        slow.route_scanned_candidates.push_back(route_scanned_candidates[idx]);
      }
      debug.slow_queries.push_back(std::move(slow));
    }
    metrics.latency_debug = std::move(debug);
  }
  return metrics;
}

}  // namespace

int main(int argc, char** argv) {
  auto runtime_opts_res = ParseRuntimeOptions(argc, argv);
  if (!runtime_opts_res.ok()) {
    const std::string message = runtime_opts_res.status().message();
    if (message == "help requested") {
      return 0;
    }
    std::cerr << runtime_opts_res.status().ToString() << std::endl;
    PrintUsage(argv[0]);
    return 1;
  }
  RuntimeOptions runtime_opts = runtime_opts_res.value();
  const std::string config_path = runtime_opts.config_path;
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  std::optional<std::string> dataset_spec = runtime_opts.dataset_spec;
  std::optional<std::string> query_spec = runtime_opts.query_spec;
  std::string dataset_label = "synthetic";

  std::optional<std::string> base_dataset_path;
  if (dataset_spec) {
    auto resolved = ResolveFvecsPath(*dataset_spec, "_base.fvecs");
    if (!resolved.ok()) {
      std::cerr << resolved.status().ToString() << std::endl;
      return 1;
    }
    base_dataset_path = resolved.value();
    std::cout << "[INFO] Using base dataset: " << *base_dataset_path << std::endl;
    std::filesystem::path ds_path(*base_dataset_path);
    auto parent_name = ds_path.parent_path().filename().string();
    if (!parent_name.empty()) {
      dataset_label = parent_name;
    } else {
      dataset_label = ds_path.stem().string();
    }
  }
  if (runtime_opts.use_default_prebuilt_index) {
    runtime_opts.prebuilt_index_dir = DefaultPrebuiltIndexDir(config_path, dataset_label);
  }
  if (runtime_opts.prebuilt_index_dir) {
    std::cout << "[INFO] Prebuilt main index enabled: "
              << *runtime_opts.prebuilt_index_dir << std::endl;
  } else if (runtime_opts.fresh_index) {
    std::cout << "[INFO] Fresh main index build forced." << std::endl;
  }

  std::optional<std::string> query_dataset_path;
  if (query_spec) {
    auto resolved_query = ResolveFvecsPath(*query_spec, "_query.fvecs");
    if (!resolved_query.ok()) {
      std::cerr << resolved_query.status().ToString() << std::endl;
      return 1;
    }
    query_dataset_path = resolved_query.value();
    std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
  } else if (dataset_spec) {
    namespace fs = std::filesystem;
    fs::path spec_path(*dataset_spec);
    std::error_code ec;
    if (fs::is_directory(spec_path, ec)) {
      auto resolved_query = ResolveFvecsPath(*dataset_spec, "_query.fvecs");
      if (resolved_query.ok()) {
        query_dataset_path = resolved_query.value();
        std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
      } else {
        std::cout << "[WARN] " << resolved_query.status().ToString()
                  << ". Falling back to random queries." << std::endl;
      }
    } else if (base_dataset_path) {
      auto parent = fs::path(*base_dataset_path).parent_path();
      if (!parent.empty()) {
        auto resolved_query = ResolveFvecsPath(parent.string(), "_query.fvecs");
        if (resolved_query.ok()) {
          query_dataset_path = resolved_query.value();
          std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
        }
      }
    }
  }

  uint32_t nx = 0;
  MatrixRM X;
  if (base_dataset_path) {
    auto load_res = LoadFvecs(*base_dataset_path);
    if (!load_res.ok()) {
      std::cerr << load_res.status().ToString() << std::endl;
      return 1;
    }
    X = load_res.value();
    nx = static_cast<uint32_t>(X.rows());
    if (nx == 0) {
      std::cerr << "Base dataset contains no vectors." << std::endl;
      return 1;
    }
    if (config.dim != static_cast<uint32_t>(X.cols())) {
      std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << X.cols() << std::endl;
      config.dim = static_cast<uint32_t>(X.cols());
    }
  } else {
    nx = 64;
    X = GenerateRandom(nx, config.dim, config.seed);
  }

  uint32_t nq = 0;
  MatrixRM Q;
  if (query_dataset_path) {
    auto load_res = LoadFvecs(*query_dataset_path);
    if (!load_res.ok()) {
      std::cerr << load_res.status().ToString() << std::endl;
      return 1;
    }
    Q = load_res.value();
    nq = static_cast<uint32_t>(Q.rows());
    if (nq == 0) {
      std::cerr << "Query dataset contains no vectors." << std::endl;
      return 1;
    }
    if (static_cast<uint32_t>(Q.cols()) != config.dim) {
      if (!base_dataset_path) {
        std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << Q.cols() << std::endl;
        config.dim = static_cast<uint32_t>(Q.cols());
      } else {
        std::cerr << "Query dimension " << Q.cols() << " mismatches base " << config.dim << std::endl;
        return 1;
      }
    }
  } else {
    nq = (config.max_queries > 0) ? config.max_queries : 8;
    Q = GenerateRandom(nq, config.dim, config.seed + 1);
  }

  const uint32_t main_rows_initial = ResolveMainRows(config, nx);
  const bool use_stream_batch_size =
      config.streaming_mode == "batch" ||
      (config.streaming_mode == "streaming" && config.streaming_use_stream_batch_size);
  const uint32_t insert_step = use_stream_batch_size ? std::max(1u, config.stream_batch_size) : 1u;
  const uint32_t rows_after_main = nx > main_rows_initial ? nx - main_rows_initial : 0;
  const uint32_t delta_train_rows =
      ResolveDeltaTrainRows(config, rows_after_main, insert_step, use_stream_batch_size);
  const uint32_t delta_ivf_nlist =
      (config.delta_ivf_nlist > 0) ? config.delta_ivf_nlist : config.ivf_nlist;
  const uint32_t merge_trigger_rows =
      (config.merge_trigger_rows > 0) ? config.merge_trigger_rows : delta_train_rows;
  const uint32_t stream_start_idx = main_rows_initial + delta_train_rows;
  const uint32_t total_stream_rows = nx > stream_start_idx ? nx - stream_start_idx : 0;
  std::cout << "[INFO] main_rows=" << main_rows_initial
            << ", delta_train_rows=" << delta_train_rows
            << ", delta_ivf_nlist=" << delta_ivf_nlist
            << ", merge_trigger_mode=" << config.merge_trigger_mode
            << ", merge_trigger_rows=" << merge_trigger_rows
            << ", merge_trigger_qe_ratio=" << config.merge_trigger_qe_ratio
            << ", merge_trigger_drift=" << config.merge_trigger_drift
            << ", merge_trigger_delta_main_ratio=" << config.merge_trigger_delta_main_ratio
            << ", merge_trigger_imbalance_ratio=" << config.merge_trigger_imbalance_ratio
            << ", enable_global_rebuild=" << std::boolalpha << config.enable_global_rebuild
            << ", global_rebuild_max_count=" << config.global_rebuild_max_count
            << ", global_rebuild_main_imbalance_ratio="
            << config.global_rebuild_main_imbalance_ratio
            << ", global_rebuild_force_main_rows=" << config.global_rebuild_force_main_rows
            << ", global_rebuild_cooldown_rows=" << config.global_rebuild_cooldown_rows
            << ", merge_assignment_mode=" << config.merge_assignment_mode
            << ", merge_assignment_top_r=" << config.merge_assignment_top_r
            << ", merge_assignment_gamma=" << config.merge_assignment_gamma
            << ", merge_assignment_hard_cap_ratio=" << config.merge_assignment_hard_cap_ratio
            << ", merge_assignment_lambda=" << config.merge_assignment_lambda
            << ", merge_score_alpha=" << config.merge_score_alpha
            << ", merge_score_beta=" << config.merge_score_beta
            << ", merge_score_threshold=" << config.merge_score_threshold
            << ", main_exact_rerank_candidates=" << config.main_exact_rerank_candidates
            << ", active_exact_rerank_candidates=" << config.active_exact_rerank_candidates
            << ", frozen_exact_rerank_candidates=" << config.frozen_exact_rerank_candidates
            << ", stream_rows=" << total_stream_rows
            << ", stream_batch_size=" << insert_step
            << ", streaming_mode=" << config.streaming_mode << std::endl;

  SearchParams params;
  params.topk = config.topk;
  params.nprobe = config.nprobe;
  params.use_whitening = false;
  params.enable_dual_route = config.enable_dual_route;

  IVFParams ivf_params;
  ivf_params.nlist = std::max(1u, config.ivf_nlist);
  ivf_params.dim = config.dim;
  ivf_params.pq.enable = config.pq_enable;
  ivf_params.pq.M = config.pq_m;
  ivf_params.pq.nbits = config.pq_nbits;
  ivf_params.pq.residual = config.pq_residual;
  ivf_params.defer_pq_stats_to_add = true;

  auto whitening = CreateWhiteningModel();
  auto main_ivf = CreateIVFIndex();
  VersionId whiten_version = 0;
  VersionSet main_versions{};
  MatrixRM X_whitened;
  MatrixRM Q_whitened;
  uint32_t main_rows_current = main_rows_initial;
  uint32_t next_insert_idx = stream_start_idx;

  Timer init_total_timer;
  const uint32_t init_visible_rows = std::min<uint32_t>(nx, stream_start_idx);
  double init_whitening_ms = 0.0;
  double init_whitening_transform_ms = 0.0;
  double init_main_build_ms = 0.0;
  double init_main_add_ms = 0.0;
  double init_load_ms = 0.0;
  bool used_prebuilt_index = false;

  if (runtime_opts.prebuilt_index_dir) {
    Timer load_timer;
    auto artifact_res = LoadPrebuiltMainIndex(*runtime_opts.prebuilt_index_dir);
    if (!artifact_res.ok()) {
      std::cerr << artifact_res.status().ToString() << std::endl;
      return 1;
    }
    PrebuiltMainIndexArtifact artifact = std::move(artifact_res.value());
    if (artifact.metadata.dim != config.dim) {
      std::cerr << "Prebuilt index dim " << artifact.metadata.dim
                << " mismatches runtime dim " << config.dim << std::endl;
      return 1;
    }
    if (artifact.metadata.main_rows != main_rows_initial) {
      std::cerr << "Prebuilt index main_rows " << artifact.metadata.main_rows
                << " mismatches runtime main_rows " << main_rows_initial << std::endl;
      return 1;
    }
    if (artifact.metadata.total_rows != 0 && artifact.metadata.total_rows != nx) {
      std::cerr << "Prebuilt index total_rows " << artifact.metadata.total_rows
                << " mismatches runtime total_rows " << nx << std::endl;
      return 1;
    }
    Status whitening_status = whitening->Deserialize(artifact.whitening_bytes);
    if (!whitening_status.ok()) {
      std::cerr << whitening_status.ToString() << std::endl;
      return 1;
    }
    Status index_status = main_ivf->Deserialize(artifact.index_bytes);
    if (!index_status.ok()) {
      std::cerr << index_status.ToString() << std::endl;
      return 1;
    }
    whiten_version = artifact.metadata.whiten_version;
    main_versions = VersionSet{whiten_version, artifact.metadata.index_version};
    auto main_sizes_res = main_ivf->GetPartitionSizes(main_versions);
    if (!main_sizes_res.ok()) {
      std::cerr << main_sizes_res.status().ToString() << std::endl;
      return 1;
    }
    init_load_ms = load_timer.ElapsedMillis();
    used_prebuilt_index = true;
    if (base_dataset_path &&
        artifact.metadata.dataset_path != "synthetic" &&
        artifact.metadata.dataset_path != *base_dataset_path) {
      std::cout << "[WARN] Prebuilt index was built from dataset_path="
                << artifact.metadata.dataset_path
                << ", runtime dataset_path=" << *base_dataset_path << std::endl;
    }
  } else {
    Timer init_fit_timer;
    auto whiten_version_res = whitening->Fit(X.topRows(main_rows_initial));
    if (!whiten_version_res.ok()) {
      std::cerr << whiten_version_res.status().ToString() << std::endl;
      return 1;
    }
    whiten_version = whiten_version_res.value();
    init_whitening_ms = init_fit_timer.ElapsedMillis();
  }

  Timer init_transform_timer;
  auto xb_res = whitening->TransformBatch(X.topRows(init_visible_rows), whiten_version);
  if (!xb_res.ok()) {
    std::cerr << xb_res.status().ToString() << std::endl;
    return 1;
  }
  X_whitened = std::move(xb_res.value());
  if (config.use_cosine) {
    NormalizeRowsL2(&X_whitened);
  }
  init_whitening_transform_ms = init_transform_timer.ElapsedMillis();

  auto qb_res = whitening->TransformBatch(Q, whiten_version);
  if (!qb_res.ok()) {
    std::cerr << qb_res.status().ToString() << std::endl;
    return 1;
  }
  Q_whitened = std::move(qb_res.value());
  if (config.use_cosine) {
    NormalizeRowsL2(&Q_whitened);
    std::cout << "[INFO] Cosine mode enabled: normalized whitened base/query rows"
              << std::endl;
  }

  if (!used_prebuilt_index) {
    std::vector<DocId> main_ids(static_cast<size_t>(main_rows_initial));
    std::iota(main_ids.begin(), main_ids.end(), 0);
    Timer init_build_timer;
    auto main_version_res = main_ivf->Build(X_whitened.topRows(main_rows_initial),
                                            main_ids,
                                            ivf_params,
                                            0);
    if (!main_version_res.ok()) {
      std::cerr << main_version_res.status().ToString() << std::endl;
      return 1;
    }
    init_main_build_ms = init_build_timer.ElapsedMillis();
    main_versions = VersionSet{whiten_version, main_version_res.value()};
    Timer init_add_timer;
    Status add_main =
        AddRangeToIndex(main_ivf, X_whitened, 0, main_rows_initial, config.dim, main_versions);
    if (!add_main.ok()) {
      std::cerr << add_main.ToString() << std::endl;
      return 1;
    }
    init_main_add_ms = init_add_timer.ElapsedMillis();
  }
  const double init_rebuild_ms = init_whitening_ms + init_whitening_transform_ms +
                                 init_main_build_ms + init_main_add_ms;
  const double init_total_wall_ms = init_total_timer.ElapsedMillis();
  std::cout << (used_prebuilt_index ? "[INIT LOAD] " : "[INIT BUILD] ")
            << "prebuilt=" << std::boolalpha << used_prebuilt_index
            << ", index_load_ms=" << init_load_ms
            << ", whitening_ms=" << init_whitening_ms
            << ", whitening_transform_ms=" << init_whitening_transform_ms
            << ", main_build_ms=" << init_main_build_ms
            << ", main_add_ms=" << init_main_add_ms
            << ", aligned_ms=" << init_rebuild_ms
            << ", wall_ms=" << init_total_wall_ms << std::endl;
  std::optional<DeltaShard> active_delta;
  std::optional<DeltaShard> frozen_delta;
  std::deque<DocId> sliding_window_doc_ids;
  uint32_t next_delta_shard_id = 2;
  bool pending_active_train = false;
  uint32_t pending_active_train_begin = 0;
  MergeOptions merge_options;
  merge_options.alpha = config.merge_score_alpha;
  merge_options.beta = config.merge_score_beta;
  merge_options.recluster_threshold = config.merge_score_threshold;
  merge_options.assignment_mode = config.merge_assignment_mode;
  merge_options.assignment_top_r = config.merge_assignment_top_r;
  merge_options.assignment_gamma = config.merge_assignment_gamma;
  merge_options.assignment_hard_cap_ratio = config.merge_assignment_hard_cap_ratio;
  merge_options.assignment_lambda = config.merge_assignment_lambda;
  if (config.enable_streaming && rows_after_main > 0) {
    // Train delta with the reserved window, then preload the same window as existing delta docs.
    auto delta_train = X_whitened.middleRows(main_rows_initial, delta_train_rows);
    IVFParams delta_params = ivf_params;
    delta_params.nlist = std::max(1u, delta_ivf_nlist);
    delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;
    auto active_res = BuildDeltaShard(delta_train, delta_params, whiten_version, 1);
    if (!active_res.ok()) {
      std::cerr << active_res.status().ToString() << std::endl;
      return 1;
    }
    active_delta = active_res.value();
    const Status add_delta_seed = AddRangeToIndex(active_delta->ivf,
                                                  X_whitened,
                                                  main_rows_initial,
                                                  stream_start_idx,
                                                  config.dim,
                                                  active_delta->versions);
    if (!add_delta_seed.ok()) {
      std::cerr << add_delta_seed.ToString() << std::endl;
      return 1;
    }
    active_delta->rows = delta_train_rows;
  }

  auto ActivatePendingDeltaFromSubsequentWindow = [&](uint32_t end_row) -> Result<double> {
    if (!pending_active_train || active_delta.has_value()) {
      return 0.0;
    }
    if (delta_train_rows == 0) {
      return Status::InvalidArgument(
          "ActivatePendingDeltaFromSubsequentWindow: delta_train_rows is 0");
    }
    if (end_row <= pending_active_train_begin) {
      return 0.0;
    }
    const uint32_t available = end_row - pending_active_train_begin;
    if (available < delta_train_rows && end_row < nx) {
      return 0.0;
    }
    const uint32_t train_rows = std::min<uint32_t>(delta_train_rows, available);
    if (train_rows == 0) {
      return 0.0;
    }
    const uint32_t train_begin = pending_active_train_begin;
    auto delta_train = X_whitened.middleRows(train_begin, train_rows);
    IVFParams delta_params = ivf_params;
    delta_params.nlist = std::max(1u, delta_ivf_nlist);
    delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;
    auto active_res =
        BuildDeltaShard(delta_train, delta_params, whiten_version, next_delta_shard_id++);
    if (!active_res.ok()) {
      return active_res.status();
    }
    DeltaShard shard = active_res.value();
    Timer delta_seed_insert_timer;
    const Status add_delta_seed = AddRangeToIndex(shard.ivf,
                                                  X_whitened,
                                                  train_begin,
                                                  train_begin + train_rows,
                                                  config.dim,
                                                  shard.versions);
    if (!add_delta_seed.ok()) {
      return add_delta_seed;
    }
    const double delta_seed_insert_ms = delta_seed_insert_timer.ElapsedMillis();
    shard.rows = train_rows;
    active_delta = std::move(shard);
    pending_active_train = false;
    pending_active_train_begin = 0;
    sliding_window_doc_ids.clear();
    for (uint32_t i = train_begin; i < train_begin + train_rows; ++i) {
      sliding_window_doc_ids.push_back(i);
    }
    std::cout << "[MERGE] activate_delta shard=" << active_delta->shard_id
              << ", warmup_rows=" << train_rows << std::endl;
    return delta_seed_insert_ms;
  };

  std::optional<MergeTriggerDecision> frozen_trigger_decision;

  auto FreezeActiveDeltaWindow = [&](uint32_t end_row,
                                     const MergeTriggerDecision* trigger_info) -> Status {
    if (delta_train_rows == 0) {
      return Status::InvalidArgument("FreezeActiveDeltaWindow: delta_train_rows is 0");
    }
    if (!active_delta.has_value()) {
      return Status::InvalidArgument("FreezeActiveDeltaWindow: active_delta is missing");
    }
    if (frozen_delta.has_value()) {
      return Status::AlreadyExists("FreezeActiveDeltaWindow: frozen_delta already exists");
    }
    if (active_delta->rows == 0) {
      return Status::InvalidArgument("FreezeActiveDeltaWindow: active_delta is empty");
    }
    frozen_delta = active_delta;
    active_delta.reset();
    pending_active_train = true;
    pending_active_train_begin = end_row;
    sliding_window_doc_ids.clear();
    if (trigger_info != nullptr) {
      frozen_trigger_decision = *trigger_info;
    } else {
      frozen_trigger_decision.reset();
    }
    std::cout << "[MERGE] freeze_delta rows=" << frozen_delta->rows
              << ", reason=" << (trigger_info != nullptr ? trigger_info->reason : "unknown")
              << ", rows_trigger=" << (trigger_info != nullptr && trigger_info->rows_trigger ? "true" : "false")
              << ", structure_trigger="
              << (trigger_info != nullptr && trigger_info->structure_trigger ? "true" : "false")
              << ", qe_ratio=" << (trigger_info != nullptr ? trigger_info->qe_ratio_value : 0.0)
              << ", drift=" << (trigger_info != nullptr ? trigger_info->drift_value : 0.0)
              << ", delta_main_ratio="
              << (trigger_info != nullptr ? trigger_info->delta_main_ratio_value : 0.0)
              << ", imbalance_ratio=" << (trigger_info != nullptr ? trigger_info->imbalance_value : 0.0)
              << ", start background-style merge window training from row="
              << pending_active_train_begin << std::endl;
    return Status::OK();
  };
  double rebuild_ms_total = init_rebuild_ms;
  double total_update_ms = 0.0;
  double total_update_whitening_ms = 0.0;
  double total_update_insert_ms = 0.0;
  double total_update_record_build_ms = 0.0;
  double total_update_insert_encode_ms = 0.0;
  double total_update_insert_commit_ms = 0.0;
  double total_update_onlinepq_maintenance_ms = 0.0;
  double total_update_delete_ms = 0.0;
  double total_update_codebook_update_ms = 0.0;
  double total_update_reencode_ms = 0.0;
  double pending_update_ms = 0.0;
  double pending_update_whitening_ms = 0.0;
  double pending_update_insert_ms = 0.0;
  double pending_update_record_build_ms = 0.0;
  double pending_update_insert_encode_ms = 0.0;
  double pending_update_insert_commit_ms = 0.0;
  double pending_update_onlinepq_maintenance_ms = 0.0;
  double pending_update_delete_ms = 0.0;
  double pending_update_codebook_update_ms = 0.0;
  double pending_update_reencode_ms = 0.0;
  double total_merge_compute_ms = 0.0;
  double pending_merge_compute_ms = 0.0;
  double total_global_rebuild_ms = 0.0;
  double pending_global_rebuild_ms = 0.0;

  OnlinePQUpdateOptions online_pq_options;
  online_pq_options.enable = config.online_pq_enable && config.pq_enable && config.pq_residual;
  online_pq_options.qe_ratio_threshold = config.online_pq_qe_ratio_threshold;
  online_pq_options.ema_alpha = config.online_pq_ema_alpha;
  online_pq_options.nqe_eps = config.online_pq_eps;
  online_pq_options.warmup_enable = config.online_pq_warmup_enable;
  online_pq_options.warmup_batches = config.online_pq_warmup_batches;
  online_pq_options.force_update_interval = config.online_pq_force_update_interval;
  online_pq_options.partial_top_alpha = config.online_pq_partial_top_alpha;
  online_pq_options.partial_alpha = config.online_pq_alpha;
  online_pq_options.partial_top_lambda = config.online_pq_partial_top_lambda;
  online_pq_options.partial_lambda = config.online_pq_lambda;
  online_pq_options.reencode_batch_after_update = config.online_pq_reencode_batch;

  const bool use_sliding_window = (config.online_pq_update_scheme == "sliding_window");
  uint32_t sliding_window_rows = 0;
  if (use_sliding_window) {
    const uint64_t row_factor =
        config.online_pq_sliding_window_use_batches ? static_cast<uint64_t>(insert_step) : 1ull;
    const uint64_t window_rows_u64 =
        static_cast<uint64_t>(config.online_pq_sliding_window_size) * row_factor;
    sliding_window_rows =
        window_rows_u64 >= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            ? std::numeric_limits<uint32_t>::max()
            : static_cast<uint32_t>(window_rows_u64);
    if (sliding_window_rows == 0) {
      std::cerr << "sliding_window_rows resolved to 0" << std::endl;
      return 1;
    }
  }

  if (active_delta.has_value() && delta_train_rows > 0) {
    for (uint32_t i = main_rows_initial; i < stream_start_idx; ++i) {
      sliding_window_doc_ids.push_back(i);
    }
  }

  OnlinePQRollup online_pq_rollup;
  OnlinePQUpdateStats last_online_pq_stats;
  std::vector<GlobalRebuildEventRecord> global_rebuild_events;
  uint32_t global_rebuild_count = 0;
  uint32_t last_global_rebuild_rows = 0;
  uint32_t last_global_rebuild_main_rows = main_rows_current;
  auto EvaluateMergeTriggerDecision =
      [&](const DeltaShard& shard, const OnlinePQUpdateStats& pq_stats)
      -> Result<MergeTriggerDecision> {
    MergeTriggerDecision decision;
    decision.mode = config.merge_trigger_mode;
    decision.active_rows = shard.rows;
    decision.qe_ratio_value = pq_stats.qe_ratio;
    decision.drift_value = pq_stats.codebook_drift_l2;
    if (main_rows_current > 0) {
      decision.delta_main_ratio_value =
          static_cast<double>(shard.rows) / static_cast<double>(main_rows_current);
    }

    if (delta_train_rows == 0 || shard.rows == 0) {
      decision.reason = "none";
      return decision;
    }

    decision.rows_trigger = merge_trigger_rows > 0 && shard.rows >= merge_trigger_rows;
    decision.qe_ratio_trigger =
        config.merge_trigger_qe_ratio > 0.0 &&
        pq_stats.qe_ratio >= config.merge_trigger_qe_ratio;
    decision.drift_trigger =
        config.merge_trigger_drift > 0.0 &&
        pq_stats.codebook_drift_l2 >= config.merge_trigger_drift;
    decision.delta_main_ratio_trigger =
        config.merge_trigger_delta_main_ratio > 0.0 &&
        decision.delta_main_ratio_value >= config.merge_trigger_delta_main_ratio;

    if (config.merge_trigger_imbalance_ratio > 0.0) {
      if (!shard.ivf) {
        return Status::InvalidArgument("EvaluateMergeTriggerDecision: active delta ivf is null");
      }
      auto sizes_res = shard.ivf->GetPartitionSizes(shard.versions);
      if (!sizes_res.ok()) {
        return sizes_res.status();
      }
      const std::vector<uint32_t>& sizes = sizes_res.value();
      decision.active_nlist = static_cast<uint32_t>(sizes.size());
      uint64_t non_empty_total = 0;
      for (uint32_t v : sizes) {
        if (v > 0) {
          decision.active_non_empty_lists++;
          non_empty_total += static_cast<uint64_t>(v);
        }
        decision.active_max_list_size = std::max<uint32_t>(decision.active_max_list_size, v);
      }
      if (decision.active_non_empty_lists > 0) {
        decision.active_avg_non_empty_list_size =
            static_cast<double>(non_empty_total) /
            static_cast<double>(decision.active_non_empty_lists);
      }
      const double denom = std::max(1.0, decision.active_avg_non_empty_list_size);
      decision.imbalance_value = static_cast<double>(decision.active_max_list_size) / denom;
      decision.imbalance_trigger =
          decision.imbalance_value >= config.merge_trigger_imbalance_ratio;
    }

    decision.structure_trigger = decision.qe_ratio_trigger || decision.drift_trigger ||
                                 decision.delta_main_ratio_trigger || decision.imbalance_trigger;

    if (config.merge_trigger_mode == "rows") {
      decision.should_trigger = decision.rows_trigger;
    } else if (config.merge_trigger_mode == "qe_ratio") {
      decision.should_trigger = decision.qe_ratio_trigger;
    } else if (config.merge_trigger_mode == "drift") {
      decision.should_trigger = decision.drift_trigger;
    } else if (config.merge_trigger_mode == "imbalance") {
      decision.should_trigger = decision.imbalance_trigger;
    } else if (config.merge_trigger_mode == "delta_main_ratio") {
      decision.should_trigger = decision.delta_main_ratio_trigger;
    } else if (config.merge_trigger_mode == "state") {
      decision.should_trigger = decision.structure_trigger;
    } else {
      decision.should_trigger = decision.rows_trigger || decision.structure_trigger;
    }

    if (!decision.should_trigger) {
      decision.reason = "none";
      return decision;
    }

    std::string reason;
    auto append_reason = [&](const std::string& token) {
      if (!reason.empty()) {
        reason += "+";
      }
      reason += token;
    };
    if (decision.rows_trigger) {
      append_reason("delta_full");
    }
    if (decision.imbalance_trigger) {
      append_reason("structure_imbalance");
    }
    if (decision.delta_main_ratio_trigger) {
      append_reason("delta_main_ratio");
    }
    if (decision.qe_ratio_trigger) {
      append_reason("qe_ratio");
    }
    if (decision.drift_trigger) {
      append_reason("drift");
    }
    if (reason.empty()) {
      reason = "triggered";
    }
    decision.reason = reason;
    return decision;
  };

  auto EvaluateGlobalRebuildDecision = [&](uint32_t seen_rows)
      -> Result<GlobalRebuildDecision> {
    GlobalRebuildDecision decision;
    decision.seen_rows = seen_rows;
    decision.main_rows_current = main_rows_current;
    if (main_rows_current >= last_global_rebuild_main_rows) {
      decision.main_rows_since_last_rebuild = main_rows_current - last_global_rebuild_main_rows;
    }
    decision.reason = "none";
    if (!config.enable_global_rebuild || config.global_rebuild_max_count == 0) {
      return decision;
    }
    if (global_rebuild_count >= config.global_rebuild_max_count) {
      decision.reason = "max_count_reached";
      return decision;
    }
    if (!main_ivf) {
      return Status::InvalidArgument("EvaluateGlobalRebuildDecision: main ivf is null");
    }
    decision.main_rows_trigger =
        config.global_rebuild_force_main_rows > 0 &&
        decision.main_rows_since_last_rebuild >= config.global_rebuild_force_main_rows;

    auto sizes_res = main_ivf->GetPartitionSizes(main_versions);
    if (!sizes_res.ok()) {
      return sizes_res.status();
    }
    const std::vector<uint32_t>& sizes = sizes_res.value();
    decision.main_nlist = static_cast<uint32_t>(sizes.size());
    uint64_t non_empty_total = 0;
    for (uint32_t sz : sizes) {
      if (sz > 0) {
        decision.main_non_empty_lists++;
        non_empty_total += static_cast<uint64_t>(sz);
      }
      decision.main_max_list_size = std::max<uint32_t>(decision.main_max_list_size, sz);
    }
    if (decision.main_non_empty_lists > 0) {
      decision.main_avg_non_empty_list_size =
          static_cast<double>(non_empty_total) /
          static_cast<double>(decision.main_non_empty_lists);
    }
    const double denom = std::max(1.0, decision.main_avg_non_empty_list_size);
    decision.main_imbalance_ratio =
        static_cast<double>(decision.main_max_list_size) / denom;
    if (config.global_rebuild_main_imbalance_ratio > 0.0) {
      decision.imbalance_trigger =
          decision.main_imbalance_ratio >= config.global_rebuild_main_imbalance_ratio;
    }

    decision.should_trigger = decision.main_rows_trigger || decision.imbalance_trigger;
    if (!decision.should_trigger) {
      decision.reason = "none";
      return decision;
    }
    if (config.global_rebuild_cooldown_rows > 0 && seen_rows > last_global_rebuild_rows &&
        (seen_rows - last_global_rebuild_rows) < config.global_rebuild_cooldown_rows) {
      decision.should_trigger = false;
      decision.reason = "cooldown";
      return decision;
    }

    std::string reason;
    auto append_reason = [&](const std::string& token) {
      if (!reason.empty()) {
        reason += "+";
      }
      reason += token;
    };
    if (decision.main_rows_trigger) {
      append_reason("main_rows");
    }
    if (decision.imbalance_trigger) {
      append_reason("main_structure_imbalance");
    }
    decision.reason = reason.empty() ? "triggered" : reason;
    return decision;
  };

  auto RunGlobalRebuild = [&](const GlobalRebuildDecision& trigger) -> Status {
    if (!trigger.should_trigger) {
      return Status::OK();
    }
    if (trigger.seen_rows == 0 || trigger.seen_rows > nx) {
      return Status::InvalidArgument("RunGlobalRebuild: invalid seen_rows");
    }
    if (global_rebuild_count >= config.global_rebuild_max_count) {
      return Status::OK();
    }

    Timer total_timer;
    const uint32_t seen_rows = trigger.seen_rows;
    const uint32_t old_main_rows = main_rows_current;
    uint32_t rebuild_main_rows = seen_rows;
    if (delta_train_rows > 0 && seen_rows > delta_train_rows) {
      rebuild_main_rows = seen_rows - delta_train_rows;
    }
    rebuild_main_rows = std::max<uint32_t>(1, rebuild_main_rows);
    const uint32_t active_seed_begin = rebuild_main_rows;
    const uint32_t active_seed_rows = seen_rows > active_seed_begin ? seen_rows - active_seed_begin : 0;

    Timer fit_timer;
    auto new_whiten_res = whitening->Fit(X.topRows(seen_rows));
    if (!new_whiten_res.ok()) {
      return new_whiten_res.status();
    }
    const VersionId new_whiten_version = new_whiten_res.value();
    const double whitening_ms = fit_timer.ElapsedMillis();

    Timer transform_timer;
    auto xb_res = whitening->TransformBatch(X.topRows(seen_rows), new_whiten_version);
    if (!xb_res.ok()) {
      return xb_res.status();
    }
    MatrixRM new_x_whitened = std::move(xb_res.value());
    if (config.use_cosine) {
      NormalizeRowsL2(&new_x_whitened);
    }
    const double whitening_transform_ms = transform_timer.ElapsedMillis();

    auto qb_res = whitening->TransformBatch(Q, new_whiten_version);
    if (!qb_res.ok()) {
      return qb_res.status();
    }
    MatrixRM new_q_whitened = std::move(qb_res.value());
    if (config.use_cosine) {
      NormalizeRowsL2(&new_q_whitened);
    }

    auto new_main_ivf = CreateIVFIndex();
    std::vector<DocId> main_ids(static_cast<size_t>(rebuild_main_rows));
    std::iota(main_ids.begin(), main_ids.end(), 0);
    Timer build_timer;
    auto new_main_version_res = new_main_ivf->Build(new_x_whitened.topRows(rebuild_main_rows),
                                                    main_ids,
                                                    ivf_params,
                                                    0);
    if (!new_main_version_res.ok()) {
      return new_main_version_res.status();
    }
    const double main_build_ms = build_timer.ElapsedMillis();
    const VersionSet new_main_versions{new_whiten_version, new_main_version_res.value()};

    Timer add_main_timer;
    Status add_main =
        AddRangeToIndex(new_main_ivf, new_x_whitened, 0, rebuild_main_rows, config.dim, new_main_versions);
    if (!add_main.ok()) {
      return add_main;
    }
    const double main_add_ms = add_main_timer.ElapsedMillis();

    std::optional<DeltaShard> rebuilt_active_delta;
    std::deque<DocId> rebuilt_sliding_window_doc_ids;
    bool rebuilt_pending_active_train = false;
    uint32_t rebuilt_pending_active_train_begin = 0;
    double delta_seed_ms = 0.0;
    if (config.enable_streaming && rows_after_main > 0) {
      if (active_seed_rows > 0) {
        Timer delta_timer;
        auto delta_train = new_x_whitened.middleRows(active_seed_begin, active_seed_rows);
        IVFParams delta_params = ivf_params;
        delta_params.nlist = std::max(1u, delta_ivf_nlist);
        delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;
        auto active_res =
            BuildDeltaShard(delta_train, delta_params, new_whiten_version, next_delta_shard_id++);
        if (!active_res.ok()) {
          return active_res.status();
        }
        DeltaShard shard = active_res.value();
        const Status add_delta_seed = AddRangeToIndex(shard.ivf,
                                                      new_x_whitened,
                                                      active_seed_begin,
                                                      seen_rows,
                                                      config.dim,
                                                      shard.versions);
        if (!add_delta_seed.ok()) {
          return add_delta_seed;
        }
        shard.rows = active_seed_rows;
        rebuilt_active_delta = std::move(shard);
        for (uint32_t i = active_seed_begin; i < seen_rows; ++i) {
          rebuilt_sliding_window_doc_ids.push_back(i);
        }
        delta_seed_ms = delta_timer.ElapsedMillis();
      } else if (seen_rows < nx && delta_train_rows > 0) {
        rebuilt_pending_active_train = true;
        rebuilt_pending_active_train_begin = seen_rows;
      }
    }

    X_whitened = std::move(new_x_whitened);
    Q_whitened = std::move(new_q_whitened);
    whiten_version = new_whiten_version;
    main_ivf = std::move(new_main_ivf);
    main_versions = new_main_versions;
    main_rows_current = rebuild_main_rows;
    active_delta = std::move(rebuilt_active_delta);
    frozen_delta.reset();
    frozen_trigger_decision.reset();
    pending_active_train = rebuilt_pending_active_train;
    pending_active_train_begin = rebuilt_pending_active_train_begin;
    sliding_window_doc_ids = std::move(rebuilt_sliding_window_doc_ids);
    last_online_pq_stats = OnlinePQUpdateStats{};

    const double wall_total_ms = total_timer.ElapsedMillis();
    const double total_ms = whitening_ms + whitening_transform_ms + main_build_ms + main_add_ms;
    rebuild_ms_total += total_ms;
    total_global_rebuild_ms += total_ms;
    pending_global_rebuild_ms += total_ms;
    global_rebuild_count++;
    last_global_rebuild_rows = seen_rows;
    last_global_rebuild_main_rows = main_rows_current;

    GlobalRebuildEventRecord event;
    event.base_rows = seen_rows;
    event.old_main_rows = old_main_rows;
    event.new_main_rows = rebuild_main_rows;
    event.active_seed_rows = active_seed_rows;
    event.rebuild_count = global_rebuild_count;
    event.max_count = config.global_rebuild_max_count;
    event.main_nlist = trigger.main_nlist;
    event.main_non_empty_lists = trigger.main_non_empty_lists;
    event.main_max_list_size = trigger.main_max_list_size;
    event.main_avg_non_empty_list_size = trigger.main_avg_non_empty_list_size;
    event.trigger_main_rows = trigger.main_rows_trigger;
    event.trigger_imbalance = trigger.imbalance_trigger;
    event.trigger_main_rows_since_last_rebuild = trigger.main_rows_since_last_rebuild;
    event.trigger_main_rows_threshold = config.global_rebuild_force_main_rows;
    event.trigger_imbalance_threshold = config.global_rebuild_main_imbalance_ratio;
    event.threshold = config.global_rebuild_main_imbalance_ratio;
    event.main_imbalance_ratio = trigger.main_imbalance_ratio;
    event.whitening_ms = whitening_ms;
    event.whitening_transform_ms = whitening_transform_ms;
    event.main_build_ms = main_build_ms;
    event.main_add_ms = main_add_ms;
    event.delta_seed_ms = delta_seed_ms;
    event.total_ms = total_ms;
    event.wall_total_ms = wall_total_ms;
    event.reason = trigger.reason;
    global_rebuild_events.push_back(event);

    std::cout << "[GLOBAL REBUILD] done: base_rows=" << seen_rows
              << ", reason=" << trigger.reason
              << ", count=" << global_rebuild_count << "/" << config.global_rebuild_max_count
              << ", main_rows_since_last=" << trigger.main_rows_since_last_rebuild
              << ", main_rows_threshold=" << config.global_rebuild_force_main_rows
              << ", imbalance=" << trigger.main_imbalance_ratio
              << ", imbalance_threshold=" << config.global_rebuild_main_imbalance_ratio
              << ", old_main_rows=" << old_main_rows
              << ", new_main_rows=" << rebuild_main_rows
              << ", active_seed_rows=" << active_seed_rows
              << ", whitening_ms=" << whitening_ms
              << ", whitening_transform_ms=" << whitening_transform_ms
              << ", main_build_ms=" << main_build_ms
              << ", main_add_ms=" << main_add_ms
              << ", delta_seed_ms=" << delta_seed_ms
              << ", total_ms=" << total_ms
              << ", wall_total_ms=" << wall_total_ms << std::endl;
    return Status::OK();
  };

  std::string config_name = std::filesystem::path(config_path).stem().string();

  std::string dataset_name = dataset_label;
  std::string metric_name = "default";
  
  if (base_dataset_path) {
    std::filesystem::path p(*base_dataset_path);
    std::vector<std::string> parts;
    for (const auto& part : p) {
      parts.push_back(part.string());
    }
  
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
  
  std::filesystem::path results_dir =
      std::filesystem::path("result") / dataset_name / metric_name / config_name;
  std::error_code ec;
  std::filesystem::create_directories(results_dir, ec);
  const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

  const bool collect_snapshots = config.enable_streaming && total_stream_rows > 0;
  const uint32_t snapshot_span =
      collect_snapshots ? std::max(1u, config.snapshot_interval > 0 ? config.snapshot_interval
                                                                     : total_stream_rows)
                        : 0;
  std::vector<SnapshotRecord> snapshots;
  std::vector<MergeEventRecord> merge_events;
  uint32_t last_snapshot_active_rows = stream_start_idx;
  std::optional<EvalMetrics> pre_stream_metrics;
  uint32_t inserted_rows = 0;

  uint64_t eval_seq = 0;
  uint32_t last_insert_begin = stream_start_idx;
  uint32_t last_insert_end = stream_start_idx;

  auto evaluate_rows = [&](uint32_t active_rows,
                           const char* stage,
                           uint32_t new_begin,
                           uint32_t new_end) -> Result<EvalMetrics> {
    auto res = EvaluateState(config,
                             X_whitened,
                             active_rows,
                             new_begin,
                             new_end,
                             main_rows_current,
                             Q,
                             Q_whitened,
                             whitening,
                             whiten_version,
                             main_ivf,
                             main_versions,
                             frozen_delta,
                             active_delta,
                             params);
    if (!res.ok()) {
      return res.status();
    }
    EvalMetrics m = res.value();
    m.rebuild_ms = rebuild_ms_total;
    const uint64_t eval_id = ++eval_seq;
    std::cout << "[EVAL] #" << eval_id << " stage=" << stage << ", base_rows=" << active_rows
              << ", recall@" << config.topk << "=" << m.recall
              << ", latency_ms=" << m.avg_query_ms << ", qps=" << m.query_qps << std::endl;
    return m;
  };

  auto write_snapshot = [&](uint32_t active_rows,
                            const EvalMetrics& metrics,
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
                            const OnlinePQUpdateStats& pq_stats,
                            bool active_window_ready,
                            bool will_commit_merge,
                            std::vector<MinibatchRecord> minibatches) -> Status {
    if (!collect_snapshots) {
      return Status::OK();
    }
    SnapshotRecord snap;
    snap.base_rows = active_rows;
    snap.main_rows = main_rows_current;
    snap.frozen_delta_docs = frozen_delta.has_value() ? frozen_delta->rows : 0;
    snap.delta_rows = active_rows > main_rows_initial ? active_rows - main_rows_initial : 0;
    snap.active_delta_docs = active_delta.has_value() ? active_delta->rows : 0;
    snap.snapshot_rows = active_rows >= last_snapshot_active_rows ? active_rows - last_snapshot_active_rows : 0;
    snap.recall = metrics.recall;
    snap.recall_new = metrics.recall_new;
    snap.recall_old = metrics.recall_old;
    snap.gt_new_ratio = metrics.gt_new_ratio;
    snap.gt_new_total = metrics.gt_new_total;
    snap.gt_old_total = metrics.gt_old_total;
    snap.hit_new_total = metrics.hit_new_total;
    snap.hit_old_total = metrics.hit_old_total;
    snap.latency_ms = metrics.avg_query_ms;
    snap.end_to_end_overhead_ms = metrics.end_to_end_overhead_ms;
    snap.avg_search_ms = metrics.avg_search_ms;
    snap.avg_scanned = metrics.scanned_avg;
    snap.query_qps = metrics.query_qps;
    snap.update_ms = update_ms;
    snap.update_whitening_ms = update_whitening_ms;
    snap.update_insert_ms = update_insert_ms;
    snap.update_record_build_ms = update_record_build_ms;
    snap.update_insert_encode_ms = update_insert_encode_ms;
    snap.update_insert_commit_ms = update_insert_commit_ms;
    snap.update_onlinepq_maintenance_ms = update_onlinepq_maintenance_ms;
    snap.update_delete_ms = update_delete_ms;
    snap.update_codebook_update_ms = update_codebook_update_ms;
    snap.update_reencode_ms = update_reencode_ms;
    snap.query_eval_ms = metrics.query_eval_ms;
    snap.merge_compute_ms = pending_merge_compute_ms;
    snap.global_rebuild_ms = pending_global_rebuild_ms;
    snap.snapshot_total_ms =
        snap.update_ms + snap.merge_compute_ms + snap.global_rebuild_ms;
    snap.update_throughput_vecps =
        (snap.snapshot_rows > 0 && snap.snapshot_total_ms > 0.0)
            ? (static_cast<double>(snap.snapshot_rows) / (snap.snapshot_total_ms / 1000.0))
            : 0.0;
    const uint32_t streamed_rows =
        active_rows >= stream_start_idx ? active_rows - stream_start_idx : 0;
    const double cumulative_maintenance_ms =
        total_update_ms + total_merge_compute_ms + total_global_rebuild_ms;
    snap.amortized_update_throughput_vecps =
        (streamed_rows > 0 && cumulative_maintenance_ms > 0.0)
            ? (1000.0 * static_cast<double>(streamed_rows) / cumulative_maintenance_ms)
            : 0.0;
    snap.nqe_batch = pq_stats.nqe_batch;
    snap.qe_ratio = pq_stats.qe_ratio;
    snap.codebook_drift = pq_stats.codebook_drift_l2;
    snap.pq_updated = pq_stats.updated_codebook;
    snap.in_warmup = pq_stats.in_warmup;
    snap.warmup_batches_left = pq_stats.warmup_batches_left;
    snap.gt_probed_rate = metrics.gt_probed_rate;
    snap.recall_on_probed_gt = metrics.recall_on_probed_gt;
    snap.exact_recall_on_probed_candidates = metrics.exact_recall_on_probed_candidates;
    snap.avg_pq_rank_loss = metrics.avg_pq_rank_loss;
    snap.miss_not_probed = metrics.miss_not_probed;
    snap.miss_probed_filtered_by_pq = metrics.miss_probed_filtered_by_pq;
    snap.pq_rank_loss_count = metrics.pq_rank_loss_count;
    snap.rerank_topk_main_total = metrics.rerank_topk_main_total;
    snap.rerank_topk_delta_total = metrics.rerank_topk_delta_total;
    snap.rerank_topk_main_ratio = metrics.rerank_topk_main_ratio;
    snap.rerank_topk_delta_ratio = metrics.rerank_topk_delta_ratio;
    snap.rerank_topk_main_avg = metrics.rerank_topk_main_avg;
    snap.rerank_topk_delta_avg = metrics.rerank_topk_delta_avg;
    snap.worst_queries = metrics.worst_queries;
    snap.active_window_ready = active_window_ready;
    snap.will_commit_merge = will_commit_merge;
    if (config.enable_latency_debug && metrics.latency_debug.has_value()) {
      snap.latency_debug = metrics.latency_debug;
      auto main_debug = BuildPartitionDebug("main", main_ivf, main_versions);
      if (!main_debug.ok()) {
        return main_debug.status();
      }
      snap.partition_debug.push_back(main_debug.value());
      if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
        auto frozen_debug =
            BuildPartitionDebug("frozen_delta", frozen_delta->ivf, frozen_delta->versions);
        if (!frozen_debug.ok()) {
          return frozen_debug.status();
        }
        snap.partition_debug.push_back(frozen_debug.value());
      }
      if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
        auto active_debug =
            BuildPartitionDebug("active_delta", active_delta->ivf, active_delta->versions);
        if (!active_debug.ok()) {
          return active_debug.status();
        }
        snap.partition_debug.push_back(active_debug.value());
      }
    }
    snap.minibatches = std::move(minibatches);
    std::cout << "[SNAPSHOT] base_rows=" << snap.base_rows
              << ", snapshot_rows=" << snap.snapshot_rows
              << ", recall@" << config.topk << "=" << snap.recall
              << ", latency_ms=" << snap.latency_ms
              << ", qps=" << snap.query_qps
              << ", throughput=" << snap.update_throughput_vecps
              << ", maintenance_ms=" << snap.snapshot_total_ms
              << ", amortized_update_throughput="
              << snap.amortized_update_throughput_vecps << std::endl;
    snapshots.push_back(std::move(snap));
    last_snapshot_active_rows = active_rows;
    return Status::OK();
  };

  if (config.enable_streaming && rows_after_main > 0) {
    auto pre_res = evaluate_rows(stream_start_idx, "pre_stream", stream_start_idx, stream_start_idx);
    if (!pre_res.ok()) {
      std::cerr << pre_res.status().ToString() << std::endl;
      return 1;
    }
    pre_stream_metrics = pre_res.value();
    if (collect_snapshots) {
      const Status ws = write_snapshot(stream_start_idx,
                                       pre_stream_metrics.value(),
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       last_online_pq_stats,
                                       false,
                                       false,
                                       std::vector<MinibatchRecord>{});
      if (!ws.ok()) {
        std::cerr << ws.ToString() << std::endl;
        return 1;
      }
    }
  }

  if (config.enable_streaming && total_stream_rows > 0) {
    const bool eval_after_each_minibatch = config.enable_miss_diag;
    uint32_t next_snapshot_target = snapshot_span;
    uint32_t minibatch_id = 0;
    std::vector<MinibatchRecord> snapshot_minibatches;

    while (next_insert_idx < nx) {
      auto activate_status = ActivatePendingDeltaFromSubsequentWindow(next_insert_idx);
      if (!activate_status.ok()) {
        std::cerr << activate_status.status().ToString() << std::endl;
        return 1;
      }
      if (activate_status.value() > 0.0) {
        total_update_ms += activate_status.value();
        total_update_insert_ms += activate_status.value();
        pending_update_ms += activate_status.value();
        pending_update_insert_ms += activate_status.value();
      }
      if (!frozen_delta.has_value() && active_delta.has_value()) {
        auto trigger_res = EvaluateMergeTriggerDecision(active_delta.value(), last_online_pq_stats);
        if (!trigger_res.ok()) {
          std::cerr << trigger_res.status().ToString() << std::endl;
          return 1;
        }
        if (trigger_res.value().should_trigger) {
          Status freeze_status = FreezeActiveDeltaWindow(next_insert_idx, &trigger_res.value());
          if (!freeze_status.ok()) {
            std::cerr << freeze_status.ToString() << std::endl;
            return 1;
          }
        }
      }

      uint32_t chunk = std::min<uint32_t>(insert_step, nx - next_insert_idx);
      if (collect_snapshots && inserted_rows < total_stream_rows) {
        const uint32_t rows_until_snapshot = next_snapshot_target - inserted_rows;
        if (rows_until_snapshot > 0) {
          chunk = std::min<uint32_t>(chunk, rows_until_snapshot);
        }
      }
      const uint32_t begin = next_insert_idx;
      const uint32_t end = begin + chunk;

      Timer whitening_timer;
      auto chunk_whiten_res = whitening->TransformBatch(X.middleRows(begin, chunk), whiten_version);
      if (!chunk_whiten_res.ok()) {
        std::cerr << chunk_whiten_res.status().ToString() << std::endl;
        return 1;
      }
      MatrixRM chunk_whitened = std::move(chunk_whiten_res.value());
      if (config.use_cosine) {
        NormalizeRowsL2(&chunk_whitened);
      }
      EnsureWhitenedCapacity(&X_whitened, end, nx, config.dim);
      X_whitened.middleRows(begin, chunk) = chunk_whitened;
      const double step_whitening_ms = whitening_timer.ElapsedMillis();

      last_online_pq_stats = OnlinePQUpdateStats{};
      if (active_delta.has_value()) {
        std::vector<DocId> delete_doc_ids;
        if (use_sliding_window) {
          const uint32_t current_rows = active_delta->rows;
          uint32_t need_delete = 0;
          if (current_rows + chunk > sliding_window_rows) {
            need_delete = current_rows + chunk - sliding_window_rows;
          }
          if (need_delete > static_cast<uint32_t>(sliding_window_doc_ids.size())) {
            need_delete = static_cast<uint32_t>(sliding_window_doc_ids.size());
          }
          delete_doc_ids.reserve(need_delete);
          for (uint32_t i = 0; i < need_delete; ++i) {
            delete_doc_ids.push_back(sliding_window_doc_ids[static_cast<size_t>(i)]);
          }
        }

        Result<OnlinePQUpdateStats> add_res = OnlinePQUpdateStats{};
        if (use_sliding_window) {
          add_res = AddRangeToIndexWithOnlinePQSlidingWindow(active_delta->ivf,
                                                             X_whitened,
                                                             begin,
                                                             end,
                                                             config.dim,
                                                             active_delta->versions,
                                                             delete_doc_ids,
                                                             online_pq_options);
        } else {
          add_res = AddRangeToIndexWithOnlinePQ(active_delta->ivf,
                                                X_whitened,
                                                begin,
                                                end,
                                                config.dim,
                                                active_delta->versions,
                                                online_pq_options);
        }
        if (!add_res.ok()) {
          std::cerr << add_res.status().ToString() << std::endl;
          return 1;
        }
        last_online_pq_stats = add_res.value();
        online_pq_rollup.batches++;
        online_pq_rollup.sum_nqe_batch += last_online_pq_stats.nqe_batch;
        online_pq_rollup.sum_qe_ratio += last_online_pq_stats.qe_ratio;
        online_pq_rollup.sum_codebook_drift += last_online_pq_stats.codebook_drift_l2;
        online_pq_rollup.last_nqe_batch = last_online_pq_stats.nqe_batch;
        online_pq_rollup.last_qe_ratio = last_online_pq_stats.qe_ratio;
        online_pq_rollup.last_codebook_drift = last_online_pq_stats.codebook_drift_l2;
        online_pq_rollup.last_warmup_batches_left = last_online_pq_stats.warmup_batches_left;
        if (last_online_pq_stats.in_warmup) {
          online_pq_rollup.warmup_batches++;
        }
        if (last_online_pq_stats.trigger_update) {
          online_pq_rollup.triggered++;
        }
        if (last_online_pq_stats.updated_codebook) {
          online_pq_rollup.updated++;
        }
        if (last_online_pq_stats.reencoded_batch) {
          online_pq_rollup.reencoded++;
        }
        online_pq_rollup.updated_subspaces += last_online_pq_stats.updated_subspaces;
        online_pq_rollup.updated_codewords += last_online_pq_stats.updated_codewords;
        if (use_sliding_window) {
          const uint32_t deleted_rows = static_cast<uint32_t>(delete_doc_ids.size());
          for (uint32_t i = 0; i < deleted_rows; ++i) {
            sliding_window_doc_ids.pop_front();
          }
          for (uint32_t i = begin; i < end; ++i) {
            sliding_window_doc_ids.push_back(i);
          }
          active_delta->rows = active_delta->rows + chunk - deleted_rows;
        } else {
          active_delta->rows += chunk;
        }
      }
      const double step_insert_ms = last_online_pq_stats.insert_ms;
      const double step_update_ms = step_whitening_ms + step_insert_ms;

      next_insert_idx = end;
      last_insert_begin = begin;
      last_insert_end = end;
      inserted_rows = next_insert_idx - stream_start_idx;
      total_update_ms += step_update_ms;
      total_update_whitening_ms += step_whitening_ms;
      total_update_insert_ms += step_insert_ms;
      total_update_record_build_ms += last_online_pq_stats.record_build_ms;
      total_update_insert_encode_ms += last_online_pq_stats.insert_encode_ms;
      total_update_insert_commit_ms += last_online_pq_stats.insert_commit_ms;
      total_update_onlinepq_maintenance_ms += last_online_pq_stats.maintenance_ms;
      total_update_delete_ms += last_online_pq_stats.delete_ms;
      total_update_codebook_update_ms += last_online_pq_stats.codebook_update_ms;
      total_update_reencode_ms += last_online_pq_stats.reencode_ms;
      pending_update_ms += step_update_ms;
      pending_update_whitening_ms += step_whitening_ms;
      pending_update_insert_ms += step_insert_ms;
      pending_update_record_build_ms += last_online_pq_stats.record_build_ms;
      pending_update_insert_encode_ms += last_online_pq_stats.insert_encode_ms;
      pending_update_insert_commit_ms += last_online_pq_stats.insert_commit_ms;
      pending_update_onlinepq_maintenance_ms += last_online_pq_stats.maintenance_ms;
      pending_update_delete_ms += last_online_pq_stats.delete_ms;
      pending_update_codebook_update_ms += last_online_pq_stats.codebook_update_ms;
      pending_update_reencode_ms += last_online_pq_stats.reencode_ms;

      activate_status = ActivatePendingDeltaFromSubsequentWindow(next_insert_idx);
      if (!activate_status.ok()) {
        std::cerr << activate_status.status().ToString() << std::endl;
        return 1;
      }
      if (activate_status.value() > 0.0) {
        total_update_ms += activate_status.value();
        total_update_insert_ms += activate_status.value();
        pending_update_ms += activate_status.value();
        pending_update_insert_ms += activate_status.value();
      }

      bool can_start_merge = false;
      std::optional<MergeTriggerDecision> post_update_trigger;
      if (!frozen_delta.has_value() && active_delta.has_value()) {
        auto trigger_res = EvaluateMergeTriggerDecision(active_delta.value(), last_online_pq_stats);
        if (!trigger_res.ok()) {
          std::cerr << trigger_res.status().ToString() << std::endl;
          return 1;
        }
        post_update_trigger = trigger_res.value();
        can_start_merge = trigger_res.value().should_trigger;
      }
      if (can_start_merge) {
        Status freeze_status =
            FreezeActiveDeltaWindow(next_insert_idx, post_update_trigger.has_value()
                                                         ? &post_update_trigger.value()
                                                         : nullptr);
        if (!freeze_status.ok()) {
          std::cerr << freeze_status.ToString() << std::endl;
          return 1;
        }
      }

      std::optional<EvalMetrics> batch_metrics;
      if (eval_after_each_minibatch) {
        auto mres = evaluate_rows(next_insert_idx, "minibatch", begin, end);
        if (!mres.ok()) {
          std::cerr << mres.status().ToString() << std::endl;
          return 1;
        }
        batch_metrics = mres.value();

        MinibatchRecord minibatch;
        minibatch.batch_id = ++minibatch_id;
        minibatch.base_rows = next_insert_idx;
        minibatch.stream_rows_total = inserted_rows;
        minibatch.batch_rows = chunk;
        minibatch.snapshot_rows_total =
            next_insert_idx >= last_snapshot_active_rows ? next_insert_idx - last_snapshot_active_rows : 0;
        minibatch.recall = batch_metrics->recall;
        minibatch.recall_new = batch_metrics->recall_new;
        minibatch.recall_old = batch_metrics->recall_old;
        minibatch.gt_new_ratio = batch_metrics->gt_new_ratio;
        minibatch.gt_new_total = batch_metrics->gt_new_total;
        minibatch.gt_old_total = batch_metrics->gt_old_total;
        minibatch.hit_new_total = batch_metrics->hit_new_total;
        minibatch.hit_old_total = batch_metrics->hit_old_total;
        minibatch.latency_ms = batch_metrics->avg_query_ms;
        minibatch.end_to_end_overhead_ms = batch_metrics->end_to_end_overhead_ms;
        minibatch.avg_search_ms = batch_metrics->avg_search_ms;
        minibatch.avg_scanned = batch_metrics->scanned_avg;
        minibatch.query_qps = batch_metrics->query_qps;
        minibatch.update_ms = step_update_ms;
        minibatch.update_whitening_ms = step_whitening_ms;
        minibatch.update_insert_ms = step_insert_ms;
        minibatch.update_record_build_ms = last_online_pq_stats.record_build_ms;
        minibatch.update_insert_encode_ms = last_online_pq_stats.insert_encode_ms;
        minibatch.update_insert_commit_ms = last_online_pq_stats.insert_commit_ms;
        minibatch.update_onlinepq_maintenance_ms = last_online_pq_stats.maintenance_ms;
        minibatch.update_delete_ms = last_online_pq_stats.delete_ms;
        minibatch.update_codebook_update_ms = last_online_pq_stats.codebook_update_ms;
        minibatch.update_reencode_ms = last_online_pq_stats.reencode_ms;
        minibatch.update_throughput_vecps =
            (chunk > 0 && step_update_ms > 0.0)
                ? (static_cast<double>(chunk) / (step_update_ms / 1000.0))
                : 0.0;
        minibatch.query_eval_ms = batch_metrics->query_eval_ms;
        minibatch.nqe_batch = last_online_pq_stats.nqe_batch;
        minibatch.qe_ratio = last_online_pq_stats.qe_ratio;
        minibatch.codebook_drift = last_online_pq_stats.codebook_drift_l2;
        minibatch.pq_updated = last_online_pq_stats.updated_codebook;
        minibatch.in_warmup = last_online_pq_stats.in_warmup;
        minibatch.warmup_batches_left = last_online_pq_stats.warmup_batches_left;
        minibatch.gt_probed_rate = batch_metrics->gt_probed_rate;
        minibatch.recall_on_probed_gt = batch_metrics->recall_on_probed_gt;
        minibatch.exact_recall_on_probed_candidates =
            batch_metrics->exact_recall_on_probed_candidates;
        minibatch.avg_pq_rank_loss = batch_metrics->avg_pq_rank_loss;
        minibatch.miss_not_probed = batch_metrics->miss_not_probed;
        minibatch.miss_probed_filtered_by_pq = batch_metrics->miss_probed_filtered_by_pq;
        minibatch.pq_rank_loss_count = batch_metrics->pq_rank_loss_count;
        snapshot_minibatches.push_back(std::move(minibatch));
      }

      bool active_window_ready = false;
      if (active_delta.has_value()) {
        auto trigger_res = EvaluateMergeTriggerDecision(active_delta.value(), last_online_pq_stats);
        if (!trigger_res.ok()) {
          std::cerr << trigger_res.status().ToString() << std::endl;
          return 1;
        }
        active_window_ready = trigger_res.value().should_trigger;
      }
      const bool should_commit_merge =
          frozen_delta.has_value() && (next_insert_idx == nx || active_window_ready);
      const bool hit_periodic_snapshot =
          collect_snapshots && inserted_rows == next_snapshot_target;
      const bool hit_final_snapshot = collect_snapshots && inserted_rows == total_stream_rows;
      if (hit_periodic_snapshot) {
        while (next_snapshot_target <= inserted_rows) {
          next_snapshot_target += snapshot_span;
        }
      }

      bool snapshot_state_changed = false;
      if (should_commit_merge) {
        Timer merge_commit_timer;
        auto merge_res = merge_frozen_delta_into_main(main_ivf,
                                                      main_versions,
                                                      frozen_delta->ivf,
                                                      frozen_delta->versions,
                                                      X_whitened,
                                                      merge_options);
        if (!merge_res.ok()) {
          std::cerr << merge_res.status().ToString() << std::endl;
          return 1;
        }
        const double merge_commit_ms = merge_commit_timer.ElapsedMillis();
        total_merge_compute_ms += merge_res.value().merge_compute_ms;
        pending_merge_compute_ms += merge_res.value().merge_compute_ms;
        rebuild_ms_total += merge_commit_ms;
        main_rows_current = std::min<uint32_t>(
            next_insert_idx, main_rows_current + merge_res.value().frozen_records);
        auto main_sizes_after_merge_res = main_ivf->GetPartitionSizes(main_versions);
        if (!main_sizes_after_merge_res.ok()) {
          std::cerr << main_sizes_after_merge_res.status().ToString() << std::endl;
          return 1;
        }
        uint32_t main_non_empty_after = 0;
        uint32_t main_max_after = 0;
        double main_avg_non_empty_after = 0.0;
        const double main_imbalance_after_real = ComputeNonEmptyListImbalance(
            main_sizes_after_merge_res.value(),
            &main_non_empty_after,
            &main_max_after,
            &main_avg_non_empty_after);
        MergeEventRecord merge_event;
        merge_event.base_rows = next_insert_idx;
        merge_event.frozen_rows = merge_res.value().frozen_records;
        merge_event.patched_partitions = merge_res.value().patch_partitions;
        merge_event.append_partitions = merge_res.value().append_partitions;
        merge_event.recluster_partitions = merge_res.value().recluster_partitions;
        merge_event.moved_delta_ratio = merge_res.value().moved_delta_ratio;
        merge_event.avg_assignment_dist_ratio = merge_res.value().avg_assignment_dist_ratio;
        merge_event.max_assignment_dist_ratio = merge_res.value().max_assignment_dist_ratio;
        merge_event.imbalance_before = merge_res.value().imbalance_before;
        merge_event.imbalance_after = merge_res.value().imbalance_after;
        merge_event.main_non_empty_lists_after = main_non_empty_after;
        merge_event.main_max_list_after = main_max_after;
        merge_event.main_avg_non_empty_list_after = main_avg_non_empty_after;
        merge_event.main_imbalance_after_real = main_imbalance_after_real;
        merge_event.trigger_mode = config.merge_trigger_mode;
        if (frozen_trigger_decision.has_value()) {
          merge_event.trigger_reason = frozen_trigger_decision->reason;
          merge_event.trigger_rows = frozen_trigger_decision->rows_trigger;
          merge_event.trigger_structure = frozen_trigger_decision->structure_trigger;
          merge_event.trigger_qe_ratio = frozen_trigger_decision->qe_ratio_trigger;
          merge_event.trigger_drift = frozen_trigger_decision->drift_trigger;
          merge_event.trigger_delta_main_ratio =
              frozen_trigger_decision->delta_main_ratio_trigger;
          merge_event.trigger_imbalance = frozen_trigger_decision->imbalance_trigger;
          merge_event.trigger_qe_ratio_value = frozen_trigger_decision->qe_ratio_value;
          merge_event.trigger_drift_value = frozen_trigger_decision->drift_value;
          merge_event.trigger_delta_main_ratio_value =
              frozen_trigger_decision->delta_main_ratio_value;
          merge_event.trigger_imbalance_value = frozen_trigger_decision->imbalance_value;
          merge_event.trigger_active_rows = frozen_trigger_decision->active_rows;
          merge_event.trigger_active_nlist = frozen_trigger_decision->active_nlist;
          merge_event.trigger_active_non_empty_lists =
              frozen_trigger_decision->active_non_empty_lists;
          merge_event.trigger_active_max_list = frozen_trigger_decision->active_max_list_size;
          merge_event.trigger_active_avg_non_empty_list =
              frozen_trigger_decision->active_avg_non_empty_list_size;
        } else {
          merge_event.trigger_reason = "unknown";
        }
        merge_event.merge_compute_ms = merge_res.value().merge_compute_ms;
        merge_event.merge_ms = merge_commit_ms;
        merge_event.codebook_rebuild_ms = merge_res.value().codebook_rebuild_ms;
        merge_events.push_back(merge_event);
        std::cout << "[MERGE] commit done: frozen_rows=" << merge_res.value().frozen_records
                  << ", patched_partitions=" << merge_res.value().patch_partitions
                  << ", append_parts=" << merge_res.value().append_partitions
                  << ", recluster_parts=" << merge_res.value().recluster_partitions
                  << ", moved_delta_ratio=" << merge_res.value().moved_delta_ratio
                  << ", avg_assignment_dist_ratio=" << merge_res.value().avg_assignment_dist_ratio
                  << ", max_assignment_dist_ratio=" << merge_res.value().max_assignment_dist_ratio
                  << ", imbalance_before=" << merge_res.value().imbalance_before
                  << ", imbalance_after=" << merge_res.value().imbalance_after
                  << ", main_imbalance_after_real=" << merge_event.main_imbalance_after_real
                  << ", trigger_reason=" << merge_event.trigger_reason
                  << ", merge_compute_ms=" << merge_res.value().merge_compute_ms
                  << ", codebook_rebuild_ms=" << merge_res.value().codebook_rebuild_ms
                  << ", merge_ms=" << merge_commit_ms << std::endl;
        frozen_delta.reset();
        frozen_trigger_decision.reset();
        snapshot_state_changed = true;
      }

      if (config.enable_global_rebuild && next_insert_idx < nx) {
        auto global_trigger_res = EvaluateGlobalRebuildDecision(next_insert_idx);
        if (!global_trigger_res.ok()) {
          std::cerr << global_trigger_res.status().ToString() << std::endl;
          return 1;
        }
        const GlobalRebuildDecision& global_trigger = global_trigger_res.value();
        std::cout << "[GLOBAL REBUILD] check: base_rows=" << next_insert_idx
                  << ", main_rows_current=" << global_trigger.main_rows_current
                  << ", main_rows_since_last=" << global_trigger.main_rows_since_last_rebuild
                  << ", imbalance=" << global_trigger.main_imbalance_ratio
                  << ", imbalance_threshold=" << config.global_rebuild_main_imbalance_ratio
                  << ", rows_trigger=" << std::boolalpha << global_trigger.main_rows_trigger
                  << ", imbalance_trigger=" << global_trigger.imbalance_trigger
                  << ", should_trigger=" << global_trigger.should_trigger
                  << ", reason=" << global_trigger.reason << std::noboolalpha << std::endl;
        if (global_trigger.should_trigger) {
          Status rebuild_status = RunGlobalRebuild(global_trigger);
          if (!rebuild_status.ok()) {
            std::cerr << rebuild_status.ToString() << std::endl;
            return 1;
          }
          snapshot_state_changed = true;
        }
      }

      if (hit_periodic_snapshot || hit_final_snapshot) {
        EvalMetrics snapshot_metrics;
        if (batch_metrics.has_value() && !snapshot_state_changed) {
          snapshot_metrics = batch_metrics.value();
        } else {
          auto sres = evaluate_rows(next_insert_idx, "snapshot", last_insert_begin, last_insert_end);
          if (!sres.ok()) {
            std::cerr << sres.status().ToString() << std::endl;
            return 1;
          }
          snapshot_metrics = sres.value();
        }
        const Status ws = write_snapshot(next_insert_idx,
                                         snapshot_metrics,
                                         pending_update_ms,
                                         pending_update_whitening_ms,
                                         pending_update_insert_ms,
                                         pending_update_record_build_ms,
                                         pending_update_insert_encode_ms,
                                         pending_update_insert_commit_ms,
                                         pending_update_onlinepq_maintenance_ms,
                                         pending_update_delete_ms,
                                         pending_update_codebook_update_ms,
                                         pending_update_reencode_ms,
                                         last_online_pq_stats,
                                         active_window_ready,
                                         should_commit_merge,
                                         eval_after_each_minibatch
                                             ? std::move(snapshot_minibatches)
                                             : std::vector<MinibatchRecord>{});
        if (!ws.ok()) {
          std::cerr << ws.ToString() << std::endl;
          return 1;
        }
        snapshot_minibatches.clear();
        pending_update_ms = 0.0;
        pending_update_whitening_ms = 0.0;
        pending_update_insert_ms = 0.0;
        pending_update_record_build_ms = 0.0;
        pending_update_insert_encode_ms = 0.0;
        pending_update_insert_commit_ms = 0.0;
        pending_update_onlinepq_maintenance_ms = 0.0;
        pending_update_delete_ms = 0.0;
        pending_update_codebook_update_ms = 0.0;
        pending_update_reencode_ms = 0.0;
        pending_merge_compute_ms = 0.0;
        pending_global_rebuild_ms = 0.0;
      }
    }
  }

  auto final_res = evaluate_rows(next_insert_idx, "final", last_insert_begin, last_insert_end);
  if (!final_res.ok()) {
    std::cerr << final_res.status().ToString() << std::endl;
    return 1;
  }
  EvalMetrics final_metrics = final_res.value();
  const double online_update_total_ms =
      total_update_ms + total_merge_compute_ms + total_global_rebuild_ms;
  final_metrics.update_total_ms = online_update_total_ms;
  final_metrics.update_per_vector_ms =
      inserted_rows > 0 ? online_update_total_ms / static_cast<double>(inserted_rows) : 0.0;
  final_metrics.update_throughput_vecps =
      (inserted_rows > 0 && online_update_total_ms > 0.0)
          ? (1000.0 * static_cast<double>(inserted_rows) / online_update_total_ms)
          : 0.0;
  const double online_avg_nqe =
      online_pq_rollup.batches > 0
          ? online_pq_rollup.sum_nqe_batch / static_cast<double>(online_pq_rollup.batches)
          : 0.0;
  const double online_avg_qe_ratio =
      online_pq_rollup.batches > 0
          ? online_pq_rollup.sum_qe_ratio / static_cast<double>(online_pq_rollup.batches)
          : 1.0;
  const double online_avg_drift =
      online_pq_rollup.batches > 0
          ? online_pq_rollup.sum_codebook_drift / static_cast<double>(online_pq_rollup.batches)
          : 0.0;

  std::cout << "[ONLINE EVAL] "
            << "Recall@" << config.topk << " = " << final_metrics.recall
            << " (nprobe=" << params.nprobe
            << ", exact_rerank=" << std::boolalpha << config.exact_rerank_enable
            << ", rerank_candidates_per_route=" << config.exact_rerank_candidates_per_route
            << ")" << std::endl;
  std::cout << "Latency(no_merge_wall)=" << final_metrics.avg_query_ms << "ms; "
            << "End-to-end overhead=" << final_metrics.end_to_end_overhead_ms << "ms; "
            << "Search p50=" << final_metrics.search_p50 << "ms, p99=" << final_metrics.search_p99
            << "ms; "
            << "Total p50=" << final_metrics.total_p50 << "ms, p99=" << final_metrics.total_p99
            << "ms; "
            << "Build/Rebuild=" << rebuild_ms_total << "ms; "
            << "Update maintenance total=" << final_metrics.update_total_ms
            << "ms (apply=" << total_update_ms
            << ", merge_compute=" << total_merge_compute_ms
            << ", global_rebuild=" << total_global_rebuild_ms
            << "), per_vec=" << final_metrics.update_per_vector_ms << "ms; "
            << "Scanned avg=" << final_metrics.scanned_avg << ", p50=" << final_metrics.scanned_p50
            << ", p99=" << final_metrics.scanned_p99 << ", max=" << final_metrics.scanned_max
            << "; Query QPS=" << final_metrics.query_qps
            << ", Update throughput=" << final_metrics.update_throughput_vecps << " vec/s"
            << std::endl;
  std::cout << "[GLOBAL REBUILD] enabled=" << std::boolalpha << config.enable_global_rebuild
            << ", count=" << global_rebuild_count << "/" << config.global_rebuild_max_count
            << ", trigger_main_imbalance=" << config.global_rebuild_main_imbalance_ratio
            << ", trigger_main_rows=" << config.global_rebuild_force_main_rows
            << ", cooldown_rows=" << config.global_rebuild_cooldown_rows << std::endl;
  std::cout << "[ONLINE PQ] enabled=" << std::boolalpha << online_pq_options.enable
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
  std::cout << "[QUERY ROUTING] main_queries=" << final_metrics.main_route_queries
            << ", frozen_delta_queries=" << final_metrics.frozen_delta_route_queries
            << ", active_delta_queries=" << final_metrics.active_delta_route_queries << std::endl;
  if (config.enable_miss_diag) {
    std::cout << "[MISS DIAG] gt_probed_rate=" << final_metrics.gt_probed_rate
              << ", recall_on_probed_gt=" << final_metrics.recall_on_probed_gt
              << ", exact_recall_on_probed_candidates="
              << final_metrics.exact_recall_on_probed_candidates
              << ", miss_not_probed=" << final_metrics.miss_not_probed
              << ", miss_probed_filtered_by_pq=" << final_metrics.miss_probed_filtered_by_pq
              << ", avg_pq_rank_loss=" << final_metrics.avg_pq_rank_loss << std::endl;
  } else {
    std::cout << "[MISS DIAG] disabled by config(enable_miss_diag=false)" << std::endl;
  }
  if (config.enable_rerank_source_diag) {
    std::cout << "[RERANK SOURCE DIAG] main_total=" << final_metrics.rerank_topk_main_total
              << ", delta_total=" << final_metrics.rerank_topk_delta_total
              << ", main_ratio=" << final_metrics.rerank_topk_main_ratio
              << ", delta_ratio=" << final_metrics.rerank_topk_delta_ratio
              << ", main_avg_topk=" << final_metrics.rerank_topk_main_avg
              << ", delta_avg_topk=" << final_metrics.rerank_topk_delta_avg << std::endl;
  } else {
    std::cout << "[RERANK SOURCE DIAG] disabled by config(enable_rerank_source_diag=false)"
              << std::endl;
  }

  std::string file_name = "online_eval.json";
  std::filesystem::path result_path = results_dir / file_name;
  std::ofstream ofs(result_path);
  if (!ofs) {
    std::cerr << "Failed to write results to " << result_path << std::endl;
    return 1;
  }


  std::vector<double> recall_values;
  std::vector<double> qps_values;
  std::vector<double> latency_values;
  std::vector<double> e2e_latency_values;
  std::vector<double> update_throughput_values;
  std::vector<double> amortized_update_throughput_values;
  for (const auto& snap : snapshots) {
    recall_values.push_back(snap.recall);
    qps_values.push_back(snap.query_qps);
    latency_values.push_back(snap.latency_ms);
    e2e_latency_values.push_back(snap.end_to_end_overhead_ms);
    if (snap.update_throughput_vecps > 0.0) {
      update_throughput_values.push_back(snap.update_throughput_vecps);
    }
    if (snap.amortized_update_throughput_vecps > 0.0) {
      amortized_update_throughput_values.push_back(snap.amortized_update_throughput_vecps);
    }
  }
  if (recall_values.empty()) {
    recall_values.push_back(final_metrics.recall);
    qps_values.push_back(final_metrics.query_qps);
    latency_values.push_back(final_metrics.avg_query_ms);
    e2e_latency_values.push_back(final_metrics.end_to_end_overhead_ms);
    if (final_metrics.update_throughput_vecps > 0.0) {
      update_throughput_values.push_back(final_metrics.update_throughput_vecps);
      amortized_update_throughput_values.push_back(final_metrics.update_throughput_vecps);
    }
  }
  const auto recall_summary = SummarizeSeries(recall_values);
  const auto qps_summary = SummarizeSeries(qps_values);
  const auto latency_summary = SummarizeSeries(latency_values);
  const auto e2e_latency_summary = SummarizeSeries(e2e_latency_values);
  const auto update_throughput_summary = SummarizeSeries(update_throughput_values);
  const auto amortized_update_throughput_summary =
      SummarizeSeries(amortized_update_throughput_values);
  const double total_maintenance_ms =
      total_update_ms + total_merge_compute_ms + total_global_rebuild_ms;

  uint64_t summary_gt_new_total = 0;
  uint64_t summary_gt_old_total = 0;
  uint64_t summary_hit_new_total = 0;
  uint64_t summary_hit_old_total = 0;
  if (!snapshots.empty()) {
    for (const auto& snap : snapshots) {
      summary_gt_new_total += snap.gt_new_total;
      summary_gt_old_total += snap.gt_old_total;
      summary_hit_new_total += snap.hit_new_total;
      summary_hit_old_total += snap.hit_old_total;
    }
  } else {
    summary_gt_new_total = final_metrics.gt_new_total;
    summary_gt_old_total = final_metrics.gt_old_total;
    summary_hit_new_total = final_metrics.hit_new_total;
    summary_hit_old_total = final_metrics.hit_old_total;
  }
  const uint64_t summary_gt_total = summary_gt_new_total + summary_gt_old_total;
  const double summary_recall_new = summary_gt_new_total > 0
      ? static_cast<double>(summary_hit_new_total) / static_cast<double>(summary_gt_new_total)
      : 0.0;
  const double summary_recall_old = summary_gt_old_total > 0
      ? static_cast<double>(summary_hit_old_total) / static_cast<double>(summary_gt_old_total)
      : 0.0;
  const double summary_gt_new_ratio = summary_gt_total > 0
      ? static_cast<double>(summary_gt_new_total) / static_cast<double>(summary_gt_total)
      : 0.0;

  const bool debug_output_enabled =
      config.enable_miss_diag || config.enable_rerank_source_diag || config.enable_latency_debug;
  if (!debug_output_enabled) {
    ofs << "{\n";
    ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
    ofs << "  \"timestamp\": " << ts << ",\n";
    ofs << "  \"params\": {\n";
    ofs << "    \"topk\": " << config.topk << ",\n";
    ofs << "    \"nprobe\": " << params.nprobe << ",\n";
    ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
    ofs << "    \"omp_max_threads\": " << RuntimeMaxThreads() << ",\n";
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
    ofs << "    \"snapshot_span\": " << snapshot_span << "\n";
    ofs << "  },\n";
    ofs << "  \"metrics\": {\n";
    ofs << "    \"recall@" << config.topk << "\": " << final_metrics.recall << ",\n";
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
    ofs << "    \"update_merge_compute_ms\": " << total_merge_compute_ms << ",\n";
    ofs << "    \"update_global_rebuild_ms\": " << total_global_rebuild_ms << ",\n";
    ofs << "    \"initial_used_prebuilt_index\": "
        << (used_prebuilt_index ? "true" : "false") << ",\n";
    ofs << "    \"initial_index_load_ms\": " << init_load_ms << ",\n";
    ofs << "    \"initial_build_aligned_ms\": " << init_rebuild_ms << ",\n";
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

    std::cout << "Saved metrics to " << result_path << std::endl;
    return 0;
  }

  ofs << "{\n";
  ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
  ofs << "  \"timestamp\": " << ts << ",\n";
  ofs << "  \"params\": {\n";
  ofs << "    \"topk\": " << config.topk << ",\n";
  ofs << "    \"nprobe\": " << params.nprobe << ",\n";
  ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
  ofs << "    \"omp_max_threads\": " << RuntimeMaxThreads() << ",\n";
  ofs << "    \"use_cosine\": " << (config.use_cosine ? "true" : "false") << ",\n";
  ofs << "    \"main_query_only\": " << (config.main_query_only ? "true" : "false") << ",\n";
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
  ofs << "    \"avg_query_ms\": " << final_metrics.avg_query_ms << ",\n";
  ofs << "    \"end_to_end_overhead_ms\": " << final_metrics.end_to_end_overhead_ms << ",\n";
  ofs << "    \"query_qps\": " << final_metrics.query_qps << ",\n";
  ofs << "    \"recall@" << config.topk << "\": " << final_metrics.recall << ",\n";
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
  ofs << "    \"enabled\": " << (online_pq_options.enable ? "true" : "false") << ",\n";
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

  std::cout << "Saved metrics to " << result_path << std::endl;
  return 0;
}
