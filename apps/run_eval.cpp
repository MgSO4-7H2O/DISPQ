#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "index/merge.h"
#include "search/exact_search.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

constexpr uint32_t kDeltaKMeansIterationsDefault = 10;
constexpr uint32_t kWorstQueryDiagCount = 10;
constexpr uint32_t kSlowQueryDebugCount = 5;
constexpr double kMergeScoreAlphaDefault = 1.0;
constexpr double kMergeScoreBetaDefault = 0.05;
constexpr double kMergeScoreThresholdDefault = 1.0;

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

struct DeltaShard {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint32_t rows{0};
  uint32_t shard_id{0};
};

struct EvalMetrics {
  double recall{0.0};
  double whitening_p50{0.0};
  double whitening_p99{0.0};
  double avg_whiten_ms{0.0};
  double avg_search_ms{0.0};
  double search_p50{0.0};
  double search_p99{0.0};
  double total_p50{0.0};
  double total_p99{0.0};
  double avg_query_ms{0.0};
  double qps{0.0};
  double rebuild_ms{0.0};
  double scanned_avg{0.0};
  double scanned_p50{0.0};
  double scanned_p99{0.0};
  double scanned_max{0.0};
  double update_total_ms{0.0};
  double update_per_vector_ms{0.0};
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
  std::vector<std::string> worst_queries;
  std::optional<LatencyDebugMetrics> latency_debug;
};

struct MinibatchRecord {
  uint32_t batch_id{0};
  uint32_t base_rows{0};
  uint32_t stream_rows_total{0};
  uint32_t batch_rows{0};
  uint32_t snapshot_rows_total{0};
  double recall{0.0};
  double avg_search_ms{0.0};
  double avg_scanned{0.0};
  double qps{0.0};
  double update_ms{0.0};
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
  double avg_search_ms{0.0};
  double avg_scanned{0.0};
  double qps{0.0};
  double update_ms{0.0};
  double query_eval_ms{0.0};
  double snapshot_total_ms{0.0};
  double snapshot_qps{0.0};
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

struct SearchRoute {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint8_t from_new{0};
  std::string name;
};

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

  AlignedVector<VectorRecord> records;
  records.reserve(static_cast<size_t>(end - begin));
  for (uint32_t i = begin; i < end; ++i) {
    VectorRecord rec;
    rec.doc_id = i;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<int64_t>(i)).transpose();
    records.push_back(std::move(rec));
  }
  return ivf->Add(records);
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
  records.reserve(static_cast<size_t>(end - begin));
  for (uint32_t i = begin; i < end; ++i) {
    VectorRecord rec;
    rec.doc_id = i;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<int64_t>(i)).transpose();
    records.push_back(std::move(rec));
  }
  return ivf->AddWithOnlinePQ(records, options);
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
  records.reserve(static_cast<size_t>(end - begin));
  for (uint32_t i = begin; i < end; ++i) {
    VectorRecord rec;
    rec.doc_id = i;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<int64_t>(i)).transpose();
    records.push_back(std::move(rec));
  }
  return ivf->AddWithOnlinePQSlidingWindow(records, delete_doc_ids, options);
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

Result<SearchResult> MergeTopK(const std::vector<SearchResult>& partial_results,
                               uint32_t topk,
                               bool exact_rerank_enable,
                               uint32_t rerank_candidates_per_route,
                               Eigen::Ref<const Eigen::VectorXf> query_whitened,
                               const MatrixRM& base_whitened) {
  SearchResult out;
  if (partial_results.empty() || topk == 0) {
    return out;
  }
  std::unordered_map<DocId, Candidate> best_by_doc;
  uint64_t scanned = 0;
  for (const auto& part : partial_results) {
    scanned += part.scanned_candidates;
    size_t route_take = part.topk.size();
    if (exact_rerank_enable) {
      route_take = std::min(route_take, static_cast<size_t>(rerank_candidates_per_route));
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
    for (auto& cand : merged) {
      if (cand.doc_id >= static_cast<DocId>(base_whitened.rows())) {
        return Status::InvalidArgument("MergeTopK: doc id out of range for exact rerank");
      }
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

  auto main_records_res = main_ivf->SnapshotRecords(main_versions);
  if (!main_records_res.ok()) {
    return main_records_res.status();
  }

  std::vector<DocId> searchable_doc_ids;
  searchable_doc_ids.reserve(main_records_res.value().size() +
                             (frozen_delta.has_value() ? frozen_delta->rows : 0u) +
                             (active_delta.has_value() ? active_delta->rows : 0u));
  std::unordered_set<DocId> searchable_seen;
  searchable_seen.reserve(searchable_doc_ids.capacity() * 2 + 1);
  auto append_unique = [&](const AlignedVector<VectorRecord>& records) -> Status {
    for (const auto& rec : records) {
      if (rec.doc_id >= seen_rows) {
        return Status::InvalidArgument("EvaluateState: route doc_id exceeds seen_rows");
      }
      if (searchable_seen.insert(rec.doc_id).second) {
        searchable_doc_ids.push_back(rec.doc_id);
      }
    }
    return Status::OK();
  };

  Status append_main = append_unique(main_records_res.value());
  if (!append_main.ok()) {
    return append_main;
  }
  if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
    auto frozen_records_res = frozen_delta->ivf->SnapshotRecords(frozen_delta->versions);
    if (!frozen_records_res.ok()) {
      return frozen_records_res.status();
    }
    Status append_frozen = append_unique(frozen_records_res.value());
    if (!append_frozen.ok()) {
      return append_frozen;
    }
  }
  if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
    auto active_records_res = active_delta->ivf->SnapshotRecords(active_delta->versions);
    if (!active_records_res.ok()) {
      return active_records_res.status();
    }
    Status append_active = append_unique(active_records_res.value());
    if (!append_active.ok()) {
      return append_active;
    }
  }
  if (searchable_doc_ids.empty()) {
    return Status::InvalidArgument("EvaluateState: no searchable docs");
  }

  MatrixRM searchable_db(searchable_doc_ids.size(), base_whitened.cols());
  for (size_t i = 0; i < searchable_doc_ids.size(); ++i) {
    searchable_db.row(static_cast<Eigen::Index>(i)) =
        base_whitened.row(static_cast<Eigen::Index>(searchable_doc_ids[i]));
  }

  auto gt_res = ExactSearchBatch(queries_whitened, searchable_db, config.topk);
  if (!gt_res.ok()) {
    return gt_res.status();
  }
  std::vector<std::vector<DocId>> ground_truth = gt_res.value();
  for (auto& row : ground_truth) {
    for (auto& did : row) {
      const size_t idx = static_cast<size_t>(did);
      if (idx < searchable_doc_ids.size()) {
        did = searchable_doc_ids[idx];
      }
    }
  }

  std::vector<SearchRoute> routes;
  routes.push_back(SearchRoute{main_ivf, main_versions, 0, "main"});
  if (!config.main_query_only) {
    if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
      routes.push_back(SearchRoute{frozen_delta->ivf, frozen_delta->versions, 1, "frozen_delta"});
    }
    if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
      routes.push_back(SearchRoute{active_delta->ivf, active_delta->versions, 1, "active_delta"});
    }
  }

  const uint32_t nq = static_cast<uint32_t>(queries_raw.rows());
  const uint32_t route_count = static_cast<uint32_t>(routes.size());
  const bool enable_exact_rerank = config.exact_rerank_enable;
  const uint32_t route_topk =
      enable_exact_rerank
          ? std::max(params.topk, config.exact_rerank_candidates_per_route)
          : params.topk;
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

  Timer wall_timer;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
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
    queries_whitened_runtime.row(i) = qbuf.transpose();
    whitening_ms[static_cast<size_t>(i)] = whiten_elapsed;
  }

  if (failed.load()) {
    return error_status;
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
  for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
    for (int64_t ri = 0; ri < static_cast<int64_t>(route_count); ++ri) {
      if (failed.load()) {
        continue;
      }
      Eigen::VectorXf q = queries_whitened_runtime.row(qi).transpose();
      Timer stimer;
      auto sres = routes[static_cast<size_t>(ri)].ivf->Search(
          q,
          route_topk,
          params.nprobe,
          routes[static_cast<size_t>(ri)].versions,
          routes[static_cast<size_t>(ri)].from_new);
      const double elapsed = stimer.ElapsedMillis();
      route_search_ms[static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri)] = elapsed;
      if (!sres.ok()) {
        std::lock_guard<std::mutex> lock(error_mu);
        if (!failed.exchange(true)) {
          error_status = sres.status();
        }
        continue;
      }
      route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)] = sres.value();
      route_scanned_candidates[static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri)] =
          static_cast<double>(sres.value().scanned_candidates);
    }
  }

  if (failed.load()) {
    return error_status;
  }

  uint64_t rerank_topk_main_total = 0;
  uint64_t rerank_topk_delta_total = 0;
  for (uint32_t qi = 0; qi < nq; ++qi) {
    Eigen::VectorXf q = queries_whitened_runtime.row(static_cast<Eigen::Index>(qi)).transpose();
    Timer merge_timer;
    auto merged_res = MergeTopK(route_results[static_cast<size_t>(qi)],
                                params.topk,
                                enable_exact_rerank,
                                config.exact_rerank_candidates_per_route,
                                q,
                                base_whitened);
    if (!merged_res.ok()) {
      return merged_res.status();
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
          rerank_topk_delta_total++;
        } else {
          rerank_topk_main_total++;
        }
      }
    }
    predictions[static_cast<size_t>(qi)] = std::move(row);
    search_ms[static_cast<size_t>(qi)] = q_search_ms;
    total_ms[static_cast<size_t>(qi)] = whitening_ms[static_cast<size_t>(qi)] + q_search_ms;
    scanned_counts[static_cast<size_t>(qi)] = static_cast<double>(merged.scanned_candidates);
  }

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

  const double wall_elapsed_ms = wall_timer.ElapsedMillis();

  auto recall_res = RecallAtK(ground_truth, predictions, config.topk);
  if (!recall_res.ok()) {
    return recall_res.status();
  }
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
  double total_latency_sum_ms = 0.0;
  if (nq > 0) {
    avg_whiten =
        std::accumulate(whitening_ms.begin(), whitening_ms.end(), 0.0) / static_cast<double>(nq);
    avg_search =
        std::accumulate(search_ms.begin(), search_ms.end(), 0.0) / static_cast<double>(nq);
    avg_total = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / static_cast<double>(nq);
    total_latency_sum_ms = std::accumulate(total_ms.begin(), total_ms.end(), 0.0);
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

  const double qps = total_latency_sum_ms > 0.0
                         ? (1000.0 * static_cast<double>(nq) / total_latency_sum_ms)
                         : 0.0;

  EvalMetrics metrics;
  metrics.recall = recall_res.value();
  metrics.whitening_p50 = whiten_summary.value().p50_ms;
  metrics.whitening_p99 = whiten_summary.value().p99_ms;
  metrics.avg_whiten_ms = avg_whiten;
  metrics.avg_search_ms = avg_search;
  metrics.search_p50 = search_summary.value().p50_ms;
  metrics.search_p99 = search_summary.value().p99_ms;
  metrics.total_p50 = total_summary.value().p50_ms;
  metrics.total_p99 = total_summary.value().p99_ms;
  metrics.avg_query_ms = avg_total;
  metrics.qps = qps;
  metrics.scanned_avg = scanned_avg;
  metrics.scanned_p50 = scanned_p50;
  metrics.scanned_p99 = scanned_p99;
  metrics.scanned_max = scanned_max;
  metrics.query_eval_ms = wall_elapsed_ms;
  metrics.query_count = nq;
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
  const std::string config_path = (argc > 1) ? argv[1] : "configs/base.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  std::optional<std::string> dataset_spec =
      (argc > 2) ? std::optional<std::string>(argv[2]) : std::nullopt;
  std::optional<std::string> query_spec =
      (argc > 3) ? std::optional<std::string>(argv[3]) : std::nullopt;
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

  if (config.max_queries > 0 && nq > config.max_queries) {
    Q = Q.topRows(config.max_queries);
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

  auto whitening = CreateWhiteningModel();
  auto main_ivf = CreateIVFIndex();
  VersionId whiten_version = 0;
  VersionSet main_versions{};
  MatrixRM X_whitened;
  MatrixRM Q_whitened;
  uint32_t main_rows_current = main_rows_initial;
  uint32_t next_insert_idx = stream_start_idx;

  Timer init_timer;
  auto whiten_version_res = whitening->Fit(X.topRows(main_rows_initial));
  if (!whiten_version_res.ok()) {
    std::cerr << whiten_version_res.status().ToString() << std::endl;
    return 1;
  }
  whiten_version = whiten_version_res.value();

  auto xb_res = whitening->TransformBatch(X, whiten_version);
  if (!xb_res.ok()) {
    std::cerr << xb_res.status().ToString() << std::endl;
    return 1;
  }
  X_whitened = xb_res.value();
  auto qb_res = whitening->TransformBatch(Q, whiten_version);
  if (!qb_res.ok()) {
    std::cerr << qb_res.status().ToString() << std::endl;
    return 1;
  }
  Q_whitened = qb_res.value();

  std::vector<DocId> main_ids(static_cast<size_t>(main_rows_initial));
  std::iota(main_ids.begin(), main_ids.end(), 0);
  MatrixRM main_train = X_whitened.topRows(main_rows_initial);
  auto main_version_res = main_ivf->Build(main_train, main_ids, ivf_params, 0);
  if (!main_version_res.ok()) {
    std::cerr << main_version_res.status().ToString() << std::endl;
    return 1;
  }
  main_versions = VersionSet{whiten_version, main_version_res.value()};
  Status add_main =
      AddRangeToIndex(main_ivf, X_whitened, 0, main_rows_initial, config.dim, main_versions);
  if (!add_main.ok()) {
    std::cerr << add_main.ToString() << std::endl;
    return 1;
  }
  std::optional<DeltaShard> active_delta;
  std::optional<DeltaShard> frozen_delta;
  std::deque<DocId> sliding_window_doc_ids;
  uint32_t next_delta_shard_id = 2;
  bool pending_active_train = false;
  uint32_t pending_active_train_begin = 0;
  MergeOptions merge_options;
  merge_options.alpha = kMergeScoreAlphaDefault;
  merge_options.beta = kMergeScoreBetaDefault;
  merge_options.recluster_threshold = kMergeScoreThresholdDefault;
  if (config.enable_streaming && rows_after_main > 0) {
    // Train delta with the reserved window, then preload the same window as existing delta docs.
    MatrixRM delta_train = X_whitened.middleRows(main_rows_initial, delta_train_rows);
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

  auto ActivatePendingDeltaFromSubsequentWindow = [&](uint32_t end_row) -> Status {
    if (!pending_active_train || active_delta.has_value()) {
      return Status::OK();
    }
    if (delta_train_rows == 0) {
      return Status::InvalidArgument(
          "ActivatePendingDeltaFromSubsequentWindow: delta_train_rows is 0");
    }
    if (end_row <= pending_active_train_begin) {
      return Status::OK();
    }
    const uint32_t available = end_row - pending_active_train_begin;
    if (available < delta_train_rows && end_row < nx) {
      return Status::OK();
    }
    const uint32_t train_rows = std::min<uint32_t>(delta_train_rows, available);
    if (train_rows == 0) {
      return Status::OK();
    }
    const uint32_t train_begin = pending_active_train_begin;
    MatrixRM delta_train = X_whitened.middleRows(train_begin, train_rows);
    IVFParams delta_params = ivf_params;
    delta_params.nlist = std::max(1u, delta_ivf_nlist);
    delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;
    auto active_res =
        BuildDeltaShard(delta_train, delta_params, whiten_version, next_delta_shard_id++);
    if (!active_res.ok()) {
      return active_res.status();
    }
    DeltaShard shard = active_res.value();
    const Status add_delta_seed = AddRangeToIndex(shard.ivf,
                                                  X_whitened,
                                                  train_begin,
                                                  train_begin + train_rows,
                                                  config.dim,
                                                  shard.versions);
    if (!add_delta_seed.ok()) {
      return add_delta_seed;
    }
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
    return Status::OK();
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
  double rebuild_ms_total = init_timer.ElapsedMillis();

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

  std::filesystem::path results_dir = std::filesystem::path("result") / dataset_label;
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

  auto evaluate_rows = [&](uint32_t active_rows) -> Result<EvalMetrics> {
    auto res = EvaluateState(config,
                             X_whitened,
                             active_rows,
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
    return m;
  };

  auto write_snapshot = [&](uint32_t active_rows,
                            const EvalMetrics& metrics,
                            double update_ms,
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
    snap.avg_search_ms = metrics.avg_search_ms;
    snap.avg_scanned = metrics.scanned_avg;
    snap.qps = metrics.qps;
    snap.update_ms = update_ms;
    snap.query_eval_ms = metrics.query_eval_ms;
    snap.snapshot_total_ms = update_ms;
    snap.snapshot_qps =
        (snap.snapshot_rows > 0 && snap.snapshot_total_ms > 0.0)
            ? (static_cast<double>(snap.snapshot_rows) / (snap.snapshot_total_ms / 1000.0))
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
    snapshots.push_back(std::move(snap));
    last_snapshot_active_rows = active_rows;
    return Status::OK();
  };

  uint32_t inserted_rows = 0;
  double total_update_ms = 0.0;
  double pending_update_ms = 0.0;

  if (config.enable_streaming && rows_after_main > 0) {
    auto pre_res = evaluate_rows(stream_start_idx);
    if (!pre_res.ok()) {
      std::cerr << pre_res.status().ToString() << std::endl;
      return 1;
    }
    pre_stream_metrics = pre_res.value();
    if (collect_snapshots) {
      const Status ws = write_snapshot(stream_start_idx,
                                       pre_stream_metrics.value(),
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
      Status activate_status = ActivatePendingDeltaFromSubsequentWindow(next_insert_idx);
      if (!activate_status.ok()) {
        std::cerr << activate_status.ToString() << std::endl;
        return 1;
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

      Timer update_timer;
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
      const double step_update_ms = update_timer.ElapsedMillis();

      next_insert_idx = end;
      inserted_rows = next_insert_idx - stream_start_idx;
      total_update_ms += step_update_ms;
      pending_update_ms += step_update_ms;

      activate_status = ActivatePendingDeltaFromSubsequentWindow(next_insert_idx);
      if (!activate_status.ok()) {
        std::cerr << activate_status.ToString() << std::endl;
        return 1;
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
        auto mres = evaluate_rows(next_insert_idx);
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
        minibatch.avg_search_ms = batch_metrics->avg_search_ms;
        minibatch.avg_scanned = batch_metrics->scanned_avg;
        minibatch.qps = batch_metrics->qps;
        minibatch.update_ms = step_update_ms;
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
      if (hit_periodic_snapshot || hit_final_snapshot) {
        EvalMetrics snapshot_metrics;
        if (batch_metrics.has_value()) {
          snapshot_metrics = batch_metrics.value();
        } else {
          auto sres = evaluate_rows(next_insert_idx);
          if (!sres.ok()) {
            std::cerr << sres.status().ToString() << std::endl;
            return 1;
          }
          snapshot_metrics = sres.value();
        }
        const Status ws = write_snapshot(next_insert_idx,
                                         snapshot_metrics,
                                         pending_update_ms,
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
      }

      if (should_commit_merge) {
        Timer merge_commit_timer;
        auto merge_res = merge_frozen_delta_into_main(main_ivf,
                                                      main_versions,
                                                      frozen_delta->ivf,
                                                      frozen_delta->versions,
                                                      merge_options);
        if (!merge_res.ok()) {
          std::cerr << merge_res.status().ToString() << std::endl;
          return 1;
        }
        const double merge_commit_ms = merge_commit_timer.ElapsedMillis();
        rebuild_ms_total += merge_commit_ms;
        main_rows_current = std::min<uint32_t>(
            next_insert_idx, main_rows_current + merge_res.value().frozen_records);
        MergeEventRecord merge_event;
        merge_event.base_rows = next_insert_idx;
        merge_event.frozen_rows = merge_res.value().frozen_records;
        merge_event.patched_partitions = merge_res.value().patch_partitions;
        merge_event.append_partitions = merge_res.value().append_partitions;
        merge_event.recluster_partitions = merge_res.value().recluster_partitions;
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
        merge_event.merge_ms = merge_commit_ms;
        merge_event.codebook_rebuild_ms = merge_res.value().codebook_rebuild_ms;
        merge_events.push_back(merge_event);
        std::cout << "[MERGE] commit done: frozen_rows=" << merge_res.value().frozen_records
                  << ", patched_partitions=" << merge_res.value().patch_partitions
                  << ", append_parts=" << merge_res.value().append_partitions
                  << ", recluster_parts=" << merge_res.value().recluster_partitions
                  << ", trigger_reason=" << merge_event.trigger_reason
                  << ", codebook_rebuild_ms=" << merge_res.value().codebook_rebuild_ms
                  << ", merge_ms=" << merge_commit_ms << std::endl;
        frozen_delta.reset();
        frozen_trigger_decision.reset();
      }
    }
  }

  auto final_res = evaluate_rows(next_insert_idx);
  if (!final_res.ok()) {
    std::cerr << final_res.status().ToString() << std::endl;
    return 1;
  }
  EvalMetrics final_metrics = final_res.value();
  final_metrics.update_total_ms = total_update_ms;
  final_metrics.update_per_vector_ms =
      inserted_rows > 0 ? total_update_ms / static_cast<double>(inserted_rows) : 0.0;
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
  std::cout << "Avg query=" << final_metrics.avg_query_ms << "ms; "
            << "Search p50=" << final_metrics.search_p50 << "ms, p99=" << final_metrics.search_p99
            << "ms; "
            << "Total p50=" << final_metrics.total_p50 << "ms, p99=" << final_metrics.total_p99
            << "ms; "
            << "Build/Rebuild=" << rebuild_ms_total << "ms; "
            << "Update total=" << final_metrics.update_total_ms
            << "ms, per_vec=" << final_metrics.update_per_vector_ms << "ms; "
            << "Scanned avg=" << final_metrics.scanned_avg << ", p50=" << final_metrics.scanned_p50
            << ", p99=" << final_metrics.scanned_p99 << ", max=" << final_metrics.scanned_max
            << "; QPS=" << final_metrics.qps << std::endl;
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

  ofs << "{\n";
  ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
  ofs << "  \"timestamp\": " << ts << ",\n";
  ofs << "  \"params\": {\n";
  ofs << "    \"topk\": " << config.topk << ",\n";
  ofs << "    \"nprobe\": " << params.nprobe << ",\n";
  ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
  ofs << "    \"main_query_only\": " << (config.main_query_only ? "true" : "false") << ",\n";
  ofs << "    \"enable_streaming\": " << (config.enable_streaming ? "true" : "false") << ",\n";
  ofs << "    \"main_index_rows_initial\": " << main_rows_initial << ",\n";
  ofs << "    \"main_rows_final\": " << main_rows_current << ",\n";
  ofs << "    \"rows_after_main\": " << rows_after_main << ",\n";
  ofs << "    \"delta_train_window\": " << config.delta_train_window << ",\n";
  ofs << "    \"delta_train_rows\": " << delta_train_rows << ",\n";
  ofs << "    \"delta_ivf_nlist\": " << config.delta_ivf_nlist << ",\n";
  ofs << "    \"delta_ivf_nlist_resolved\": " << delta_ivf_nlist << ",\n";
  ofs << "    \"merge_trigger_mode\": \"" << config.merge_trigger_mode << "\",\n";
  ofs << "    \"merge_trigger_rows\": " << config.merge_trigger_rows << ",\n";
  ofs << "    \"merge_trigger_rows_resolved\": " << merge_trigger_rows << ",\n";
  ofs << "    \"merge_trigger_qe_ratio\": " << config.merge_trigger_qe_ratio << ",\n";
  ofs << "    \"merge_trigger_drift\": " << config.merge_trigger_drift << ",\n";
  ofs << "    \"merge_trigger_delta_main_ratio\": " << config.merge_trigger_delta_main_ratio
      << ",\n";
  ofs << "    \"merge_trigger_imbalance_ratio\": " << config.merge_trigger_imbalance_ratio
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
  ofs << "    \"qps\": " << final_metrics.qps << ",\n";
  ofs << "    \"recall@" << config.topk << "\": " << final_metrics.recall << ",\n";
  ofs << "    \"build_rebuild_ms\": " << rebuild_ms_total << ",\n";
  ofs << "    \"update_total_ms\": " << final_metrics.update_total_ms << ",\n";
  ofs << "    \"update_per_vector_ms\": " << final_metrics.update_per_vector_ms << ",\n";
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
  ofs << "  \"pre_stream_metrics\": ";
  if (pre_stream_metrics.has_value()) {
    ofs << "{\n";
    ofs << "    \"base_rows\": " << stream_start_idx << ",\n";
    ofs << "    \"recall\": " << pre_stream_metrics->recall << ",\n";
    ofs << "    \"avg_query_ms\": " << pre_stream_metrics->avg_query_ms << ",\n";
    ofs << "    \"avg_search_ms\": " << pre_stream_metrics->avg_search_ms << ",\n";
    ofs << "    \"qps\": " << pre_stream_metrics->qps << ",\n";
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
    ofs << "      \"codebook_rebuild_ms\": " << ev.codebook_rebuild_ms << ",\n";
    ofs << "      \"merge_ms\": " << ev.merge_ms << "\n";
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
    ofs << "      \"base_rows\": " << snap.base_rows << ",\n";
    ofs << "      \"main_rows\": " << snap.main_rows << ",\n";
    ofs << "      \"frozen_delta_docs\": " << snap.frozen_delta_docs << ",\n";
    ofs << "      \"delta_rows\": " << snap.delta_rows << ",\n";
    ofs << "      \"active_delta_docs\": " << snap.active_delta_docs << ",\n";
    ofs << "      \"snapshot_rows\": " << snap.snapshot_rows << ",\n";
    ofs << "      \"recall\": " << snap.recall << ",\n";
    ofs << "      \"avg_search_ms\": " << snap.avg_search_ms << ",\n";
    ofs << "      \"avg_scanned\": " << snap.avg_scanned << ",\n";
    ofs << "      \"qps\": " << snap.qps << ",\n";
    ofs << "      \"update_ms\": " << snap.update_ms << ",\n";
    ofs << "      \"query_eval_ms\": " << snap.query_eval_ms << ",\n";
    ofs << "      \"snapshot_total_ms\": " << snap.snapshot_total_ms << ",\n";
    ofs << "      \"snapshot_qps\": " << snap.snapshot_qps << ",\n";
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
      ofs << "          \"avg_search_ms\": " << mb.avg_search_ms << ",\n";
      ofs << "          \"avg_scanned\": " << mb.avg_scanned << ",\n";
      ofs << "          \"qps\": " << mb.qps << ",\n";
      ofs << "          \"update_ms\": " << mb.update_ms << ",\n";
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
