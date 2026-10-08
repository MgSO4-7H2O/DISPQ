#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "eval_memory.h"
#include "index/ivf.h"
#include "index/merge.h"
#include "search/rerank.h"
#include "whitening/whitening.h"

namespace {

using namespace ann;
using Clock = std::chrono::steady_clock;
constexpr uint32_t kConcurrentTransformBlockRows = 65536;
constexpr double kTimelineBucketMs = 200.0;

struct Route {
  std::shared_ptr<IVFIndex> index;
  VersionSet versions;
  uint32_t rows{0};
  uint32_t id{0};
};

struct QueryRecord {
  uint32_t query_id{0};
  uint64_t version{0};
  double latency_ms{0.0};
  double completion_ms{0.0};
  std::vector<DocId> returned_topk;
};

struct InsertTimelineRecord {
  uint64_t batch{0};
  uint64_t version{0};
  uint32_t begin_row{0};
  uint32_t end_row{0};
  double start_ms{0.0};
  double end_ms{0.0};
  double transform_ms{0.0};
  double insert_ms{0.0};
  double batch_ms{0.0};
};

enum class TimelineEventType { kMerge, kRebuild };

struct TimelineEvent {
  TimelineEventType type{TimelineEventType::kMerge};
  uint64_t id{0};
  double request_ms{0.0};
  double start_ms{0.0};
  double compute_done_ms{0.0};
  double commit_wait_start_ms{0.0};
  double commit_lock_acquired_ms{0.0};
  double commit_done_ms{0.0};
  double end_ms{0.0};
  uint32_t rows{0};
  uint32_t main_rows{0};
  double prepare_publish_us{0.0};
  double commit_validation_us{0.0};
  double commit_docmap_us{0.0};
  double commit_partition_swap_us{0.0};
};

struct RuntimeState {
  std::shared_ptr<IVFIndex> main;
  VersionSet main_versions;
  std::optional<Route> frozen;
  std::optional<Route> active;
  uint64_t version{0};
  uint32_t committed_rows{0};
  uint32_t main_rows{0};
  bool active_freeze_pending{false};
};

double Percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double pos = quantile * (values.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

Status TransformBaseRangeToIndexData(
    const ann::eval_memory::BaseVectorSource& source,
    uint32_t begin,
    uint32_t end,
    bool use_whitening,
    VersionId version,
    const std::shared_ptr<WhiteningModel>& whitening,
    bool use_cosine,
    MatrixRM* index_data) {
  for (uint32_t block_begin = begin; block_begin < end;) {
    const uint32_t rows = std::min<uint32_t>(kConcurrentTransformBlockRows,
                                             end - block_begin);
    auto block = ann::eval_memory::TransformSourceRangeToIndexSpace(
        source, block_begin, rows, use_whitening, version, whitening, use_cosine);
    if (!block.ok()) return block.status();
    index_data->middleRows(static_cast<Eigen::Index>(block_begin),
                           static_cast<Eigen::Index>(rows)) = block.value();
    block_begin += rows;
  }
  return Status::OK();
}

std::vector<std::vector<std::vector<DocId>>> ComputeIncrementalPrefixGroundTruth(
    const MatrixRM& queries,
    const MatrixRM& database,
    uint32_t topk,
    const std::vector<uint32_t>& frontiers) {
  const size_t query_count = static_cast<size_t>(queries.rows());
  std::vector<std::vector<std::vector<DocId>>> gt(
      frontiers.size(), std::vector<std::vector<DocId>>(query_count));
  const uint32_t final_frontier = frontiers.empty() ? 0 : frontiers.back();
  std::vector<float> database_norms(final_frontier);
  for (uint32_t row = 0; row < final_frontier; ++row)
    database_norms[row] = database.row(row).squaredNorm();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t qi = 0; qi < queries.rows(); ++qi) {
    const Eigen::VectorXf query = queries.row(qi).transpose();
    const float query_norm = query.squaredNorm();
    const size_t heap_capacity = std::min<size_t>(topk, final_frontier);
    std::vector<std::pair<float, DocId>> heap;
    std::vector<std::pair<float, DocId>> ordered;
    heap.reserve(heap_capacity);
    ordered.reserve(heap_capacity);
    uint32_t scanned_rows = 0;

    for (size_t version = 0; version < frontiers.size(); ++version) {
      const uint32_t frontier = frontiers[version];
      for (uint32_t row = scanned_rows; row < frontier; ++row) {
        const float dot = database.row(row).dot(query);
        const std::pair<float, DocId> candidate{
            query_norm + database_norms[row] - 2.0f * dot, row};
        if (heap.size() < topk) {
          heap.push_back(candidate);
          std::push_heap(heap.begin(), heap.end());
        } else if (!heap.empty() && candidate < heap.front()) {
          std::pop_heap(heap.begin(), heap.end());
          heap.back() = candidate;
          std::push_heap(heap.begin(), heap.end());
        }
      }
      scanned_rows = frontier;

      ordered.assign(heap.begin(), heap.end());
      std::sort(ordered.begin(), ordered.end());
      auto& query_ids = gt[version][static_cast<size_t>(qi)];
      query_ids.resize(topk, 0);
      for (size_t i = 0; i < ordered.size(); ++i)
        query_ids[i] = ordered[i].second;
    }
  }
  return gt;
}

Result<std::shared_ptr<IVFIndex>> BuildIndex(const MatrixRM& data,
                                             uint32_t begin,
                                             uint32_t end,
                                             IVFParams params,
                                             VersionSet* versions) {
  params.nlist = std::max(1u, std::min(params.nlist, end - begin));
  std::vector<DocId> ids(end - begin);
  std::iota(ids.begin(), ids.end(), begin);
  auto index = CreateIVFIndex();
  auto build = index->Build(data.middleRows(begin, end - begin), ids, params, 0);
  if (!build.ok()) return build.status();
  *versions = VersionSet{versions->whiten_version, build.value()};
  AlignedVector<VectorRecord> records(end - begin);
  for (uint32_t row = begin; row < end; ++row) {
    auto& record = records[row - begin];
    record.doc_id = row;
    record.dim = static_cast<uint32_t>(data.cols());
    record.versions = *versions;
    record.x = data.row(row).transpose();
  }
  const Status add = index->AddBatch(records, true);
  if (!add.ok()) return add;
  return index;
}

Status AddOnline(const std::shared_ptr<IVFIndex>& index,
                 const MatrixRM& data,
                 uint32_t begin,
                 uint32_t end,
                 VersionSet versions,
                 const OnlinePQUpdateOptions& options,
                 uint32_t threads,
                 bool sliding_window,
                 const std::vector<DocId>& delete_ids,
                 OnlinePQUpdateStats* stats) {
  AlignedVector<VectorRecord> records(end - begin);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int64_t offset = 0; offset < static_cast<int64_t>(records.size()); ++offset) {
    const uint32_t row = begin + static_cast<uint32_t>(offset);
    auto& record = records[static_cast<size_t>(offset)];
    record.doc_id = row;
    record.dim = static_cast<uint32_t>(data.cols());
    record.versions = versions;
    record.x = data.row(row).transpose();
  }
  Result<OnlinePQUpdateStats> result = OnlinePQUpdateStats{};
  if (sliding_window) {
    AlignedVector<VectorRecord> deleted;
    deleted.reserve(delete_ids.size());
    for (DocId id : delete_ids) {
      VectorRecord record;
      record.doc_id = id;
      record.dim = static_cast<uint32_t>(data.cols());
      record.versions = versions;
      record.x = data.row(id).transpose();
      deleted.push_back(std::move(record));
    }
    result = index->AddWithOnlinePQSlidingWindowRecords(records, deleted, options);
  } else {
    result = index->AddWithOnlinePQ(records, options);
  }
  if (!result.ok()) return result.status();
  *stats = result.value();
  return Status::OK();
}

void WriteString(std::ostream& out, const std::string& value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  out.write(reinterpret_cast<const char*>(&size), sizeof(size));
  out.write(value.data(), size);
}

std::string ReadString(std::istream& in) {
  uint32_t size = 0;
  in.read(reinterpret_cast<char*>(&size), sizeof(size));
  std::string value(size, '\0');
  in.read(value.data(), size);
  return value;
}

template <typename T>
void WriteValue(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
bool ReadValue(std::istream& in, T* value) {
  return static_cast<bool>(in.read(reinterpret_cast<char*>(value), sizeof(*value)));
}

Result<std::vector<std::vector<std::vector<DocId>>>> LoadGroundTruth(
    const std::string& path,
    const std::string& base_id,
    const std::string& query_id,
    uint32_t query_count,
    uint32_t topk,
    uint32_t initial_rows,
    uint32_t batch_size,
    const std::vector<uint32_t>& frontiers) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Status::IOError("Unable to open GT artifact: " + path);
  char magic[8]{};
  in.read(magic, sizeof(magic));
  uint32_t nq = 0, k = 0, initial = 0, batch = 0, versions = 0;
  if (std::string(magic, sizeof(magic)) != "DISPGT01" ||
      ReadString(in) != base_id || ReadString(in) != query_id ||
      !ReadValue(in, &nq) || !ReadValue(in, &k) || !ReadValue(in, &initial) ||
      !ReadValue(in, &batch) || !ReadValue(in, &versions) || nq != query_count ||
      k != topk || initial != initial_rows || batch != batch_size ||
      versions != frontiers.size()) {
    return Status::InvalidArgument("GT artifact metadata mismatch");
  }
  std::vector<std::vector<std::vector<DocId>>> gt(versions,
      std::vector<std::vector<DocId>>(query_count));
  for (uint32_t v = 0; v < versions; ++v) {
    uint32_t frontier = 0;
    if (!ReadValue(in, &frontier) || frontier != frontiers[v])
      return Status::InvalidArgument("GT artifact frontier mismatch");
    for (uint32_t q = 0; q < query_count; ++q) {
      gt[v][q].resize(topk);
      in.read(reinterpret_cast<char*>(gt[v][q].data()), topk * sizeof(DocId));
      if (!in) return Status::IOError("Truncated GT artifact");
    }
  }
  return gt;
}

Status SaveGroundTruth(const std::string& path,
                       const std::string& base_id,
                       const std::string& query_id,
                       uint32_t topk,
                       uint32_t initial_rows,
                       uint32_t batch_size,
                       const std::vector<uint32_t>& frontiers,
                       const MatrixRM& queries,
                       const MatrixRM& database) {
  const auto gt = ComputeIncrementalPrefixGroundTruth(queries, database, topk, frontiers);
  std::ofstream out(path, std::ios::binary);
  if (!out) return Status::IOError("Unable to write GT artifact: " + path);
  out.write("DISPGT01", 8);
  WriteString(out, base_id);
  WriteString(out, query_id);
  const uint32_t query_count = static_cast<uint32_t>(queries.rows());
  const uint32_t version_count = static_cast<uint32_t>(frontiers.size());
  WriteValue(out, query_count);
  WriteValue(out, topk);
  WriteValue(out, initial_rows);
  WriteValue(out, batch_size);
  WriteValue(out, version_count);
  for (size_t version = 0; version < frontiers.size(); ++version) {
    WriteValue(out, frontiers[version]);
    for (const auto& ids : gt[version])
      out.write(reinterpret_cast<const char*>(ids.data()), topk * sizeof(DocId));
  }
  return out ? Status::OK() : Status::IOError("Failed writing GT artifact");
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ann;
#ifdef _OPENMP
  const int runtime_max_threads = std::max(1, omp_get_max_threads());
#else
  const int runtime_max_threads = 1;
#endif
  if (argc < 3 || argc > 5) {
    std::cerr << "Usage: run_concurrent_eval <config.json> <dataset_dir_or_base_file> "
                 "[query_or_dir] [summary.json]\n";
    return 2;
  }
  auto config_result = LoadConfigFromJson(argv[1]);
  if (!config_result.ok()) { std::cerr << config_result.status().ToString() << '\n'; return 1; }
  const Config config = config_result.value();
  if (!config.concurrent_workload_enable) {
    std::cerr << "concurrent_workload_enable must be true for this executable\n";
    return 1;
  }
  auto base_path_result = ResolveFvecsPath(argv[2], "_base.fvecs");
  if (!base_path_result.ok()) { std::cerr << base_path_result.status().ToString() << '\n'; return 1; }
  const std::string base_path = base_path_result.value();
  std::string query_path;
  if (argc >= 4) {
    auto result = ResolveFvecsPath(argv[3], "_query.fvecs");
    if (!result.ok()) { std::cerr << result.status().ToString() << '\n'; return 1; }
    query_path = result.value();
  } else {
    const std::filesystem::path base(base_path);
    auto result = ResolveFvecsPath(base.parent_path().string(), "_query.fvecs");
    if (!result.ok()) { std::cerr << result.status().ToString() << '\n'; return 1; }
    query_path = result.value();
  }
  auto source_result = ann::eval_memory::MakeFvecsBaseSource(base_path);
  if (!source_result.ok()) { std::cerr << source_result.status().ToString() << '\n'; return 1; }
  auto base_source = std::move(source_result.value());
  auto query_result = LoadFvecs(query_path);
  if (!query_result.ok()) {
    std::cerr << query_result.status().ToString() << '\n';
    return 1;
  }
  MatrixRM queries = std::move(query_result.value());
  if (base_source.rows == 0) { std::cerr << "Base dataset contains no vectors\n"; return 1; }
  if (config.max_queries > 0 && queries.rows() > config.max_queries)
    queries.conservativeResize(config.max_queries, Eigen::NoChange);
  const uint32_t nx = base_source.rows;
  const uint32_t nq = static_cast<uint32_t>(queries.rows());
  if (nq == 0 || static_cast<Eigen::Index>(base_source.dim) != queries.cols()) {
    std::cerr << "Empty queries or base/query dimension mismatch\n";
    return 1;
  }

  MatrixRM index_data(static_cast<Eigen::Index>(nx),
                      static_cast<Eigen::Index>(base_source.dim));
  MatrixRM query_data = queries;
  auto whitening = CreateWhiteningModel();
  VersionId whiten_version = 0;
  uint32_t main_rows = config.enable_streaming ? config.main_index_rows : nx;
  if (main_rows == 0 && config.enable_streaming) main_rows = std::max(1u, nx / 2);
  if (main_rows >= nx && nx > 1) main_rows = nx - 1;
  const uint32_t insert_step = config.streaming_mode == "batch" ||
                                   (config.streaming_mode == "streaming" && config.streaming_use_stream_batch_size)
                                   ? config.stream_batch_size : 1u;
  uint32_t delta_rows = 0;
  if (config.enable_streaming && nx > main_rows) {
    const uint64_t wanted = static_cast<uint64_t>(config.delta_train_window) * insert_step;
    delta_rows = std::min<uint32_t>(nx - main_rows, static_cast<uint32_t>(wanted));
  }
  const uint32_t initial_rows = main_rows + delta_rows;
  const uint32_t batch_size = insert_step;
  const auto whitening_fit_start = Clock::now();
  if (config.use_whitening) {
    auto fit_rows = base_source.LoadRange(0, main_rows);
    if (!fit_rows.ok()) { std::cerr << fit_rows.status().ToString() << '\n'; return 1; }
    auto fit = whitening->Fit(fit_rows.value());
    if (!fit.ok()) { std::cerr << fit.status().ToString() << '\n'; return 1; }
    whiten_version = fit.value();
  }
  std::cout << "[INIT] whitening fit done, elapsed_ms="
            << std::chrono::duration<double, std::milli>(Clock::now() - whitening_fit_start).count()
            << std::endl;

  const auto initial_transform_start = Clock::now();
  const Status initial_transform = TransformBaseRangeToIndexData(
      base_source, 0, initial_rows, config.use_whitening, whiten_version,
      whitening, config.use_cosine, &index_data);
  if (!initial_transform.ok()) {
    std::cerr << initial_transform.ToString() << '\n';
    return 1;
  }
  std::cout << "[INIT] initial base transform done, rows=" << initial_rows
            << ", elapsed_ms="
            << std::chrono::duration<double, std::milli>(Clock::now() - initial_transform_start).count()
            << std::endl;

  if (config.use_whitening) {
    auto transformed_queries = whitening->TransformBatch(query_data, whiten_version);
    if (!transformed_queries.ok()) {
      std::cerr << transformed_queries.status().ToString() << '\n';
      return 1;
    }
    query_data = std::move(transformed_queries.value());
  }
  if (config.use_cosine) {
    for (Eigen::Index i = 0; i < query_data.rows(); ++i) {
      const float norm = query_data.row(i).norm();
      if (norm > 0.0f) query_data.row(i) /= norm;
    }
  }

  std::vector<uint32_t> frontiers{initial_rows};
  for (uint32_t end = initial_rows; end < nx; end += batch_size)
    frontiers.push_back(std::min(nx, end + batch_size));
  const std::string base_identity = base_path;
  const std::string query_identity = query_path;
  std::vector<std::vector<std::vector<DocId>>> gt;
  const std::string gt_path = config.concurrent_gt_path.empty()
                                  ? std::string(argv[1]) + ".concurrent.gt"
                                  : config.concurrent_gt_path;
  if (config.concurrent_gt_mode == "prepare") {
    const Status remaining_transform = TransformBaseRangeToIndexData(
        base_source, initial_rows, nx, config.use_whitening, whiten_version,
        whitening, config.use_cosine, &index_data);
    if (!remaining_transform.ok()) {
      std::cerr << remaining_transform.ToString() << '\n';
      return 1;
    }
    const Status status = SaveGroundTruth(gt_path, base_identity, query_identity, config.topk,
                                          initial_rows, batch_size, frontiers, query_data, index_data);
    if (!status.ok()) { std::cerr << status.ToString() << '\n'; return 1; }
    std::cout << "GT artifact written: " << gt_path << '\n';
    return 0;
  }
  if (config.concurrent_gt_mode == "load") {
    const auto gt_load_start = Clock::now();
    auto result = LoadGroundTruth(gt_path, base_identity, query_identity, nq, config.topk,
                                  initial_rows, batch_size, frontiers);
    if (!result.ok()) { std::cerr << result.status().ToString() << '\n'; return 1; }
    gt = std::move(result.value());
    std::cout << "[INIT] GT loaded, elapsed_ms="
              << std::chrono::duration<double, std::milli>(Clock::now() - gt_load_start).count()
              << std::endl;
  }

  IVFParams ivf_params;
  ivf_params.nlist = config.ivf_nlist;
  ivf_params.dim = static_cast<uint32_t>(index_data.cols());
  ivf_params.pq.enable = config.pq_enable;
  ivf_params.pq.M = config.pq_m;
  ivf_params.pq.nbits = config.pq_nbits;
  ivf_params.pq.residual = config.pq_residual;
  ivf_params.pq_codebook_dimension_major = config.pq_codebook_dimension_major;
  ivf_params.pq_codes_subquantizer_major = config.pq_codes_subquantizer_major;
  ivf_params.defer_pq_stats_to_add = true;
  RuntimeState state;
  state.committed_rows = initial_rows;
  state.main_rows = main_rows;
  state.main_versions.whiten_version = whiten_version;
  const auto main_build_start = Clock::now();
  auto main = BuildIndex(index_data, 0, main_rows, ivf_params, &state.main_versions);
  if (!main.ok()) { std::cerr << main.status().ToString() << '\n'; return 1; }
  state.main = main.value();
  std::cout << "[INIT] main index built, rows=" << main_rows << ", elapsed_ms="
            << std::chrono::duration<double, std::milli>(Clock::now() - main_build_start).count()
            << std::endl;
  const uint32_t delta_nlist = config.delta_ivf_nlist ? config.delta_ivf_nlist : config.ivf_nlist;
  const auto active_build_start = Clock::now();
  if (delta_rows > 0) {
    IVFParams delta_params = ivf_params;
    delta_params.nlist = delta_nlist;
    delta_params.kmeans_iterations = 10;
    VersionSet versions{whiten_version, 0};
    auto delta = BuildIndex(index_data, main_rows, initial_rows, delta_params, &versions);
    if (!delta.ok()) { std::cerr << delta.status().ToString() << '\n'; return 1; }
    state.active = Route{delta.value(), versions, delta_rows, 1};
  }
  std::cout << "[INIT] initial active built, rows=" << delta_rows << ", elapsed_ms="
            << std::chrono::duration<double, std::milli>(Clock::now() - active_build_start).count()
            << std::endl;
  std::deque<DocId> active_window_ids;
  if (config.online_pq_update_scheme == "sliding_window") {
    for (uint32_t row = main_rows; row < initial_rows; ++row) active_window_ids.push_back(row);
  }

  OnlinePQUpdateOptions online_options;
  online_options.enable = config.online_pq_enable;
  online_options.qe_ratio_threshold = config.online_pq_qe_ratio_threshold;
  online_options.ema_alpha = config.online_pq_ema_alpha;
  online_options.nqe_eps = config.online_pq_eps;
  online_options.warmup_enable = config.online_pq_warmup_enable;
  online_options.warmup_batches = config.online_pq_warmup_batches;
  online_options.force_update_interval = config.online_pq_force_update_interval;
  online_options.partial_top_alpha = config.online_pq_partial_top_alpha;
  online_options.partial_alpha = config.online_pq_alpha;
  online_options.partial_top_lambda = config.online_pq_partial_top_lambda;
  online_options.partial_lambda = config.online_pq_lambda;
  online_options.reencode_batch_after_update = config.online_pq_reencode_batch;

  std::shared_mutex workload_mutex;
  std::mutex workload_entry_mutex;
  std::condition_variable workload_entry_cv;
  bool rebuild_pending = false;
  std::mutex state_mutex;
  std::mutex records_mutex;
  std::mutex version_mutex;
  std::mutex maintenance_mutex;
  std::condition_variable maintenance_cv;
  std::condition_variable state_cv;
  bool maintenance_stop = false;
  bool merge_requested = false;
  bool rebuild_requested = false;
  std::atomic<bool> stop_queries{false};
  std::atomic<bool> start_workers{false};
  std::atomic<uint64_t> next_query{0};
  std::vector<QueryRecord> query_records;
  std::vector<InsertTimelineRecord> insert_timeline_records;
  std::vector<TimelineEvent> timeline_events;
  std::vector<double> insert_batch_ms;
  std::atomic<uint64_t> inserted_vectors{0}, committed_batches{0}, merge_count{0}, rebuild_count{0};
  double merge_total_ms = 0.0, merge_compute_ms = 0.0, merge_commit_ms = 0.0;
  double merge_codebook_rebuild_ms = 0.0, rebuild_total_ms = 0.0;
  uint64_t merge_patch_partitions = 0, merge_append_partitions = 0;
  uint64_t merge_recluster_partitions = 0;
  double max_insert_lag_ms = 0.0, active_rows_over_trigger = 0.0;
  uint64_t insert_backpressure_events = 0;
  double insert_backpressure_total_ms = 0.0, insert_backpressure_max_ms = 0.0;
  std::atomic<uint64_t> main_queries{0}, frozen_queries{0}, active_queries{0};
  std::atomic<uint32_t> current_committed{initial_rows};
  std::atomic<uint64_t> current_version{0};
  std::atomic<bool> failed{false};
  std::string failure;
  std::mutex failure_mutex;
  auto report_failure = [&](const std::string& message) {
    if (!failed.exchange(true)) {
      {
        std::lock_guard<std::mutex> lock(failure_mutex);
        failure = message;
      }
      state_cv.notify_all();
      maintenance_cv.notify_all();
      workload_entry_cv.notify_all();
    }
  };
  MergeOptions merge_options;
  merge_options.alpha = config.merge_score_alpha;
  merge_options.beta = config.merge_score_beta;
  merge_options.recluster_threshold = config.merge_score_threshold;
  merge_options.assignment_mode = config.merge_assignment_mode;
  merge_options.assignment_top_r = config.merge_assignment_top_r;
  merge_options.assignment_gamma = config.merge_assignment_gamma;
  merge_options.assignment_hard_cap_ratio = config.merge_assignment_hard_cap_ratio;
  merge_options.assignment_lambda = config.merge_assignment_lambda;
  const uint32_t trigger_rows = config.merge_trigger_rows ? config.merge_trigger_rows : delta_rows;
  uint32_t last_global_rebuild_rows = initial_rows;
  uint32_t last_global_rebuild_main_rows = main_rows;
  Clock::time_point measurement_start;
  auto timeline_ms = [&](Clock::time_point time) {
    return std::chrono::duration<double, std::milli>(time - measurement_start).count();
  };
  auto request_maintenance = [&](bool merge, bool rebuild) {
    if (!merge && !rebuild) return;
    {
      std::lock_guard<std::mutex> lock(maintenance_mutex);
      merge_requested = merge_requested || merge;
      rebuild_requested = rebuild_requested || rebuild;
    }
    maintenance_cv.notify_one();
  };
  auto promote_active_to_frozen = [&] {
    state.frozen = std::move(state.active);
    state.active.reset();
    state.active_freeze_pending = false;
    active_window_ids.clear();
  };
  bool was_backpressured = false;
  auto insert_admission_gate = [&] {
    was_backpressured = false;
    bool promoted = false;
    {
      std::unique_lock<std::mutex> state_lock(state_mutex);
      if (state.active_freeze_pending && state.frozen && !failed.load()) {
        was_backpressured = true;
        ++insert_backpressure_events;
        const uint64_t batch = committed_batches.load() + 1;
        std::cout << "[BACKPRESSURE] begin batch=" << batch
                  << " active_rows=" << (state.active ? state.active->rows : 0)
                  << " frozen_rows=" << state.frozen->rows << std::endl;
        const auto wait_start = Clock::now();
        state_cv.wait(state_lock, [&] {
          return failed.load() || !state.active_freeze_pending || !state.frozen;
        });
        const double wait_ms = std::chrono::duration<double, std::milli>(
                                   Clock::now() - wait_start).count();
        insert_backpressure_total_ms += wait_ms;
        insert_backpressure_max_ms = std::max(insert_backpressure_max_ms, wait_ms);
        std::cout << "[BACKPRESSURE] end batch=" << batch
                  << " wait_ms=" << wait_ms << std::endl;
      }
      if (failed.load()) return false;
      if (state.active_freeze_pending && !state.frozen && state.active) {
        promote_active_to_frozen();
        promoted = true;
      }
    }
    if (promoted) request_maintenance(true, false);
    return !failed.load();
  };

#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_max_active_levels(1);
#endif
  std::thread maintenance([&] {
#ifdef _OPENMP
    omp_set_num_threads(std::max(1u, config.concurrent_maintenance_threads));
#endif
    for (;;) {
      std::unique_lock<std::mutex> wait_lock(maintenance_mutex);
      maintenance_cv.wait(wait_lock, [&] {
        return merge_requested || rebuild_requested || maintenance_stop || failed.load();
      });
      if ((maintenance_stop || failed.load()) && !merge_requested && !rebuild_requested) return;
      const bool rebuild = rebuild_requested;
      rebuild_requested = false;
      merge_requested = false;
      wait_lock.unlock();
      if (rebuild) {
        const auto total_start = Clock::now();
        const uint64_t rebuild_id = rebuild_count.load() + 1;
        uint32_t rebuild_rows = 0;
        uint32_t rebuild_main_rows = 0;
        Clock::time_point rebuild_requested;
        {
          std::lock_guard<std::mutex> state_lock(state_mutex);
          rebuild_rows = state.committed_rows;
          rebuild_main_rows = state.main_rows;
        }
        {
          std::lock_guard<std::mutex> entry_lock(workload_entry_mutex);
          rebuild_pending = true;
          rebuild_requested = Clock::now();
        }
        std::cout << "[REBUILD] requested id=" << rebuild_id
                  << " committed_rows=" << rebuild_rows
                  << " main_rows=" << rebuild_main_rows << std::endl;
        const auto barrier_wait_start = Clock::now();
        std::unique_lock<std::shared_mutex> barrier(workload_mutex);
        const auto barrier_acquired = Clock::now();
        const double barrier_wait_ms = std::chrono::duration<double, std::milli>(
                                           barrier_acquired - barrier_wait_start).count();
#ifdef _OPENMP
        omp_set_num_threads(runtime_max_threads);
#endif
        std::cout << "[REBUILD] barrier_acquired id=" << rebuild_id
                  << " wait_ms=" << barrier_wait_ms
                  << " threads=" << runtime_max_threads << std::endl;
        {
          std::lock_guard<std::mutex> state_lock(state_mutex);
          rebuild_rows = state.committed_rows;
        }
        VersionSet versions{whiten_version, 0};
        const auto build_start = Clock::now();
        auto replacement = BuildIndex(index_data, 0, rebuild_rows, ivf_params, &versions);
        const double build_ms = std::chrono::duration<double, std::milli>(
                                    Clock::now() - build_start).count();
        if (!replacement.ok()) {
          report_failure(replacement.status().ToString());
        } else {
          std::lock_guard<std::mutex> state_lock(state_mutex);
          state.main = replacement.value();
          state.main_versions = versions;
          state.main_rows = rebuild_rows;
          state.active.reset();
          state.frozen.reset();
          state.active_freeze_pending = false;
          ++rebuild_count;
          last_global_rebuild_rows = rebuild_rows;
          last_global_rebuild_main_rows = rebuild_rows;
        }
        const auto rebuild_end = Clock::now();
        if (replacement.ok()) {
          TimelineEvent event;
          event.type = TimelineEventType::kRebuild;
          event.id = rebuild_id;
          event.request_ms = timeline_ms(rebuild_requested);
          event.start_ms = timeline_ms(barrier_acquired);
          event.end_ms = timeline_ms(rebuild_end);
          event.rows = rebuild_rows;
          event.main_rows = rebuild_main_rows;
          timeline_events.push_back(event);
        }
#ifdef _OPENMP
        omp_set_num_threads(std::max(1u, config.concurrent_maintenance_threads));
#endif
        barrier.unlock();
        {
          std::lock_guard<std::mutex> entry_lock(workload_entry_mutex);
          rebuild_pending = false;
        }
        workload_entry_cv.notify_all();
        const double rebuild_ms = std::chrono::duration<double, std::milli>(
                                      Clock::now() - total_start).count();
        rebuild_total_ms += rebuild_ms;
        if (replacement.ok()) {
          std::cout << "[REBUILD] done id=" << rebuild_id
                    << " build_ms=" << build_ms
                    << " total_ms=" << rebuild_ms << std::endl;
        }
      } else {
        std::optional<Route> frozen;
        std::shared_ptr<IVFIndex> main_index;
        VersionSet main_versions;
        uint32_t main_rows_for_merge = 0;
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          frozen = state.frozen;
          main_index = state.main;
          main_versions = state.main_versions;
          main_rows_for_merge = state.main_rows;
        }
        if (frozen) {
          const auto begin = Clock::now();
          const double merge_start_ms = std::chrono::duration<double, std::milli>(begin - measurement_start).count();
          std::cout << "[MERGE] start id=" << frozen->id
                    << " frozen_rows=" << frozen->rows
                    << " main_rows=" << main_rows_for_merge
                    << " elapsed_ms="
                    << std::chrono::duration<double, std::milli>(Clock::now() - measurement_start).count()
                    << std::endl;
          std::shared_lock<std::shared_mutex> access(workload_mutex);
          auto snapshot = freeze_delta(frozen->index, frozen->versions, index_data);
          bool merge_succeeded = false;
          double event_merge_compute_ms = 0.0;
          double event_codebook_rebuild_ms = 0.0;
          double event_merge_commit_ms = 0.0;
          MergeProfiling event_profiling;
          if (!snapshot.ok()) report_failure(snapshot.status().ToString());
          else {
            auto merged = merge_frozen_delta_into_main(main_index, main_versions,
                                                        snapshot.value(), index_data,
                                                        merge_options);
            if (!merged.ok()) report_failure(merged.status().ToString());
            else {
              merge_succeeded = true;
              event_merge_compute_ms = merged.value().merge_compute_ms;
              event_codebook_rebuild_ms = merged.value().codebook_rebuild_ms;
              event_merge_commit_ms = merged.value().profiling.commit_us / 1000.0;
              event_profiling = merged.value().profiling;
              ++merge_count;
              merge_compute_ms += event_merge_compute_ms;
              merge_codebook_rebuild_ms += event_codebook_rebuild_ms;
              merge_commit_ms += event_merge_commit_ms;
              merge_patch_partitions += merged.value().patch_partitions;
              merge_append_partitions += merged.value().append_partitions;
              merge_recluster_partitions += merged.value().recluster_partitions;
              std::lock_guard<std::mutex> lock(state_mutex);
              if (state.frozen && state.frozen->id == frozen->id) {
                state.main_rows += frozen->rows;
                state.frozen.reset();
              }
            }
          }
          const auto merge_end = Clock::now();
          const double merge_ms = std::chrono::duration<double, std::milli>(merge_end - begin).count();
          merge_total_ms += merge_ms;
          if (merge_succeeded) {
            TimelineEvent event;
            event.type = TimelineEventType::kMerge;
            event.id = frozen->id;
            event.request_ms = merge_start_ms;
            event.start_ms = merge_start_ms;
            event.compute_done_ms = timeline_ms(event_profiling.compute_done_at);
            event.commit_wait_start_ms = timeline_ms(event_profiling.commit_wait_start_at);
            event.commit_lock_acquired_ms = timeline_ms(event_profiling.commit_lock_acquired_at);
            event.commit_done_ms = timeline_ms(event_profiling.commit_done_at);
            event.end_ms = timeline_ms(merge_end);
            event.rows = frozen->rows;
            event.main_rows = main_rows_for_merge;
            event.prepare_publish_us = event_profiling.prepare_publish_us;
            event.commit_validation_us = event_profiling.commit_validation_us;
            event.commit_docmap_us = event_profiling.commit_docmap_us;
            event.commit_partition_swap_us = event_profiling.commit_partition_swap_us;
            timeline_events.push_back(event);
            uint32_t merged_main_rows = 0;
            {
              std::lock_guard<std::mutex> lock(state_mutex);
              merged_main_rows = state.main_rows;
            }
            std::cout << "[MERGE] done id=" << frozen->id
                      << " frozen_rows=" << frozen->rows
                      << " total_ms=" << merge_ms
                      << " compute_ms=" << event_merge_compute_ms
                      << " codebook_ms=" << event_codebook_rebuild_ms
                      << " commit_ms=" << event_merge_commit_ms
                      << " main_rows=" << merged_main_rows << std::endl;
          }
        }
      }
      state_cv.notify_all();
      maintenance_cv.notify_all();
    }
  });

  std::vector<std::thread> query_workers;
  query_workers.reserve(config.concurrent_query_workers);
  for (uint32_t worker = 0; worker < config.concurrent_query_workers; ++worker) {
    query_workers.emplace_back([&] {
#ifdef _OPENMP
      omp_set_num_threads(std::max(1u, config.concurrent_query_threads_per_request));
      omp_set_dynamic(0);
      omp_set_max_active_levels(1);
#endif
      while (!start_workers.load(std::memory_order_acquire)) std::this_thread::yield();
      while (!stop_queries.load(std::memory_order_relaxed)) {
        const auto dispatch = Clock::now();
        const uint32_t qid = static_cast<uint32_t>(next_query.fetch_add(1) % nq);
        std::shared_lock<std::shared_mutex> workload_access;
        {
          std::unique_lock<std::mutex> entry_lock(workload_entry_mutex);
          workload_entry_cv.wait(entry_lock, [&] {
            return !rebuild_pending || failed.load() ||
                   stop_queries.load(std::memory_order_relaxed);
          });
          if (failed.load() || stop_queries.load(std::memory_order_relaxed)) break;
          workload_access = std::shared_lock<std::shared_mutex>(workload_mutex);
        }
        std::shared_ptr<IVFIndex> main_index;
        VersionSet main_versions;
        std::optional<Route> frozen, active;
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          main_index = state.main; main_versions = state.main_versions;
          frozen = state.frozen; active = state.active;
        }
        std::vector<Candidate> candidates;
        auto collect = [&](const std::shared_ptr<IVFIndex>& index, VersionSet versions,
                           uint8_t from_new, uint32_t candidate_count) {
          if (!index) return Status::OK();
          auto result = index->Search(query_data.row(qid).transpose(), candidate_count,
                                      config.nprobe, versions, from_new);
          if (!result.ok()) return result.status();
          candidates.insert(candidates.end(), result.value().topk.begin(), result.value().topk.end());
          return Status::OK();
        };
        const uint32_t default_candidates = config.exact_rerank_candidates_per_route
                                                ? config.exact_rerank_candidates_per_route
                                                : config.topk;
        const uint32_t main_candidates = config.main_exact_rerank_candidates
                                             ? config.main_exact_rerank_candidates
                                             : default_candidates;
        const uint32_t frozen_candidates = config.frozen_exact_rerank_candidates
                                               ? config.frozen_exact_rerank_candidates
                                               : default_candidates;
        const uint32_t active_candidates = config.active_exact_rerank_candidates
                                               ? config.active_exact_rerank_candidates
                                               : default_candidates;
        Status status = collect(main_index, main_versions, 0,
                                config.exact_rerank_enable ? main_candidates : config.topk);
        if (status.ok() && frozen)
          status = collect(frozen->index, frozen->versions, 1,
                           config.exact_rerank_enable ? frozen_candidates : config.topk);
        if (status.ok() && active)
          status = collect(active->index, active->versions, 1,
                           config.exact_rerank_enable ? active_candidates : config.topk);
        if (!status.ok()) { report_failure(status.ToString()); stop_queries = true; break; }
        {
          std::vector<Candidate> unique;
          std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.doc_id == b.doc_id) {
              if (a.approx_dist != b.approx_dist) return a.approx_dist < b.approx_dist;
              return a.from_new < b.from_new;
            }
            return a.doc_id < b.doc_id;
          });
          for (const Candidate& candidate : candidates) {
            if (std::none_of(unique.begin(), unique.end(), [&](const Candidate& v) { return v.doc_id == candidate.doc_id; }))
              unique.push_back(candidate);
          }
          std::sort(unique.begin(), unique.end(), [](const Candidate& a, const Candidate& b) {
            if (a.approx_dist != b.approx_dist) return a.approx_dist < b.approx_dist;
            return a.doc_id < b.doc_id;
          });
          if (config.exact_rerank_enable && !unique.empty()) {
            const auto vector_for = [&](DocId id) -> Result<Eigen::VectorXf> {
              if (id >= static_cast<DocId>(index_data.rows())) return Status::NotFound("rerank vector not found");
              return Eigen::VectorXf(index_data.row(id).transpose());
            };
            auto rerank = RerankL2(query_data.row(qid).transpose(), &unique, vector_for);
            if (!rerank.ok()) { report_failure(rerank.status().ToString()); stop_queries = true; break; }
            std::sort(unique.begin(), unique.end(), [](const Candidate& a, const Candidate& b) {
              if (a.rerank_dist != b.rerank_dist) return a.rerank_dist < b.rerank_dist;
              if (a.approx_dist != b.approx_dist) return a.approx_dist < b.approx_dist;
              return a.doc_id < b.doc_id;
            });
          }
          if (unique.size() > config.topk) unique.resize(config.topk);
          QueryRecord record;
          record.query_id = qid;
          {
            std::lock_guard<std::mutex> commit_lock(version_mutex);
            record.version = current_version.load(std::memory_order_acquire);
          }
          const auto completion = Clock::now();
          record.latency_ms = std::chrono::duration<double, std::milli>(completion - dispatch).count();
          record.completion_ms = std::chrono::duration<double, std::milli>(completion - measurement_start).count();
          for (const Candidate& candidate : unique) record.returned_topk.push_back(candidate.doc_id);
          std::lock_guard<std::mutex> lock(records_mutex);
          query_records.push_back(std::move(record));
        }
        ++main_queries;
        if (frozen) ++frozen_queries;
        if (active) ++active_queries;
      }
    });
  }

  measurement_start = Clock::now();
  std::cout << "[WORKLOAD] concurrent workload start" << std::endl;
  start_workers.store(true, std::memory_order_release);
  const double period_ms = config.concurrent_target_insert_vecps > 0.0
                               ? 1000.0 * batch_size / config.concurrent_target_insert_vecps : 0.0;
  auto next_deadline = measurement_start;
  bool first_batch = true;
  uint32_t next_row = initial_rows;
  double insert_wall_ms = 0.0;
  const auto insert_stream_start = Clock::now();
  uint32_t next_shard_id = 2;
  while (next_row < nx && !failed.load()) {
    if (!insert_admission_gate()) break;
    if (was_backpressured) next_deadline = Clock::now();
    if (period_ms > 0.0 && !first_batch) {
      next_deadline += std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double, std::milli>(period_ms));
      std::this_thread::sleep_until(next_deadline);
      max_insert_lag_ms = std::max(max_insert_lag_ms,
          std::max(0.0, std::chrono::duration<double, std::milli>(Clock::now() - next_deadline).count()));
    }
    first_batch = false;
    if (failed.load()) break;

    const uint32_t end = std::min(nx, next_row + batch_size);
    const uint32_t batch_begin = next_row;
    const uint64_t batch_number = committed_batches.load() + 1;
    const auto batch_start = Clock::now();
    const auto transform_start = Clock::now();
    const Status batch_transform = TransformBaseRangeToIndexData(
        base_source, next_row, end, config.use_whitening, whiten_version,
        whitening, config.use_cosine, &index_data);
    if (!batch_transform.ok()) {
      report_failure(batch_transform.ToString());
      break;
    }
    const double transform_ms = std::chrono::duration<double, std::milli>(Clock::now() - transform_start).count();
    const auto insert_start = Clock::now();
    OnlinePQUpdateStats update_stats;
    {
      std::shared_lock<std::shared_mutex> access;
      {
        std::unique_lock<std::mutex> entry_lock(workload_entry_mutex);
        workload_entry_cv.wait(entry_lock, [&] {
          return !rebuild_pending || failed.load();
        });
        if (failed.load()) break;
        access = std::shared_lock<std::shared_mutex>(workload_mutex);
      }
      std::lock_guard<std::mutex> commit_lock(version_mutex);
      std::shared_ptr<IVFIndex> active_index;
      VersionSet active_versions;
      bool create_active;
      std::optional<Route> new_active;
      std::vector<DocId> delete_ids;
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        create_active = !state.active;
        if (!create_active) {
          active_index = state.active->index;
          active_versions = state.active->versions;
        }
      }
      if (create_active) {
        IVFParams params = ivf_params;
        params.nlist = delta_nlist;
        params.kmeans_iterations = 10;
        VersionSet versions{whiten_version, 0};
        auto built = BuildIndex(index_data, next_row, end, params, &versions);
        if (!built.ok()) { report_failure(built.status().ToString()); break; }
        new_active = Route{built.value(), versions, end - next_row, next_shard_id++};
      } else {
        const bool sliding_window = config.online_pq_update_scheme == "sliding_window";
        if (sliding_window && state.active->rows + end - next_row >
                                  config.online_pq_sliding_window_size) {
          const uint32_t delete_count = std::min<uint32_t>(
              state.active->rows + end - next_row - config.online_pq_sliding_window_size,
              static_cast<uint32_t>(active_window_ids.size()));
          for (uint32_t i = 0; i < delete_count; ++i)
            delete_ids.push_back(active_window_ids[i]);
        }
        const Status inserted = AddOnline(active_index, index_data, next_row, end,
                                          active_versions, online_options,
                                          config.concurrent_insert_threads, sliding_window,
                                          delete_ids, &update_stats);
        if (!inserted.ok()) { report_failure(inserted.ToString()); break; }
        for (size_t i = 0; i < delete_ids.size(); ++i) active_window_ids.pop_front();
        for (uint32_t row = next_row; row < end; ++row) active_window_ids.push_back(row);
      }
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (new_active) {
          state.active = std::move(new_active);
          active_window_ids.clear();
          for (uint32_t row = next_row; row < end; ++row) active_window_ids.push_back(row);
        }
        else if (state.active)
          state.active->rows += end - next_row - static_cast<uint32_t>(delete_ids.size());
        state.committed_rows = end;
        ++state.version;
        current_committed.store(end, std::memory_order_release);
        current_version.store(state.version, std::memory_order_release);
      }
    }
    const double insert_ms = std::chrono::duration<double, std::milli>(Clock::now() - insert_start).count();
    const auto batch_end = Clock::now();
    const double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
    insert_timeline_records.push_back({
        batch_number, current_version.load(std::memory_order_acquire), batch_begin, end,
        std::chrono::duration<double, std::milli>(batch_start - measurement_start).count(),
        std::chrono::duration<double, std::milli>(batch_end - measurement_start).count(),
        transform_ms, insert_ms, batch_ms});
    insert_batch_ms.push_back(batch_ms);
    inserted_vectors += end - next_row;
    ++committed_batches;
    next_row = end;
    bool trigger = false;
    bool rows_trigger = false;
    bool qe_trigger = false;
    bool drift_trigger = false;
    bool ratio_trigger = false;
    bool imbalance_trigger = false;
    bool request_merge = false;
    bool request_rebuild = false;
    uint32_t log_active_rows = 0;
    uint32_t trigger_active_rows = 0;
    uint32_t log_frozen_rows = 0;
    uint64_t log_version = 0;
    bool log_pending = false;
    bool log_frozen = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (state.active) {
        trigger_active_rows = state.active->rows;
        rows_trigger = trigger_rows > 0 && state.active->rows >= trigger_rows;
        qe_trigger = config.merge_trigger_qe_ratio > 0.0 &&
                                update_stats.qe_ratio >= config.merge_trigger_qe_ratio;
        drift_trigger = config.merge_trigger_drift > 0.0 &&
                                   update_stats.codebook_drift_l2 >= config.merge_trigger_drift;
        ratio_trigger = config.merge_trigger_delta_main_ratio > 0.0 &&
            static_cast<double>(state.active->rows) / std::max(1u, state.main_rows) >=
                config.merge_trigger_delta_main_ratio;
        if (config.merge_trigger_imbalance_ratio > 0.0) {
          auto sizes = state.active->index->GetPartitionSizes(state.active->versions);
          if (sizes.ok() && !sizes.value().empty()) {
            uint64_t total = 0;
            uint32_t nonempty = 0, max_size = 0;
            for (uint32_t size : sizes.value()) {
              total += size;
              if (size > 0) ++nonempty;
              max_size = std::max(max_size, size);
            }
            const double average = nonempty ? static_cast<double>(total) / nonempty : 0.0;
            imbalance_trigger = average > 0.0 &&
                max_size / average >= config.merge_trigger_imbalance_ratio;
          }
        }
        const bool state_trigger = qe_trigger || drift_trigger || ratio_trigger || imbalance_trigger;
        if (config.merge_trigger_mode == "rows") trigger = rows_trigger;
        else if (config.merge_trigger_mode == "qe_ratio") trigger = qe_trigger;
        else if (config.merge_trigger_mode == "drift") trigger = drift_trigger;
        else if (config.merge_trigger_mode == "delta_main_ratio") trigger = ratio_trigger;
        else if (config.merge_trigger_mode == "imbalance") trigger = imbalance_trigger;
        else if (config.merge_trigger_mode == "state") trigger = state_trigger;
        else trigger = rows_trigger || state_trigger;
        trigger = trigger && config.enable_merge;
        if (state.frozen && rows_trigger)
          active_rows_over_trigger = std::max<double>(active_rows_over_trigger,
                                                       state.active->rows - trigger_rows);
        if (trigger) {
          if (!state.frozen) {
            promote_active_to_frozen();
            request_merge = true;
          } else {
            state.active_freeze_pending = true;
          }
        }
      }
      log_active_rows = state.active ? state.active->rows : 0;
      log_frozen_rows = state.frozen ? state.frozen->rows : 0;
      log_pending = state.active_freeze_pending;
      log_frozen = state.frozen.has_value();
      log_version = state.version;
      bool global_trigger = config.enable_global_rebuild &&
                            config.global_rebuild_max_count > rebuild_count.load();
      if (global_trigger && config.global_rebuild_force_main_rows > 0 &&
          state.main_rows >= last_global_rebuild_main_rows + config.global_rebuild_force_main_rows) {
        global_trigger = true;
      } else if (global_trigger && config.global_rebuild_main_imbalance_ratio > 0.0) {
        auto sizes = state.main->GetPartitionSizes(state.main_versions);
        global_trigger = false;
        if (sizes.ok() && !sizes.value().empty()) {
          uint64_t total = 0;
          uint32_t nonempty = 0, max_size = 0;
          for (uint32_t size : sizes.value()) {
            total += size;
            if (size > 0) ++nonempty;
            max_size = std::max(max_size, size);
          }
          const double average = nonempty ? static_cast<double>(total) / nonempty : 0.0;
          global_trigger = average > 0.0 &&
              max_size / average >= config.global_rebuild_main_imbalance_ratio;
        }
      } else {
        global_trigger = false;
      }
      if (global_trigger && config.global_rebuild_cooldown_rows > 0 &&
          state.committed_rows - last_global_rebuild_rows < config.global_rebuild_cooldown_rows) {
        global_trigger = false;
      }
      if (global_trigger) {
        request_rebuild = true;
      }
    }
    if (trigger) {
      std::cout << "[TRIGGER] batch=" << batch_number
                << " version=" << log_version
                << " active_rows=" << trigger_active_rows
                << " rows=" << (rows_trigger ? "true" : "false")
                << " qe=" << (qe_trigger ? "true" : "false")
                << " drift=" << (drift_trigger ? "true" : "false")
                << " ratio=" << (ratio_trigger ? "true" : "false")
                << " imbalance=" << (imbalance_trigger ? "true" : "false")
                << " frozen=" << (log_frozen ? "true" : "false") << std::endl;
    }
    std::cout << "[INSERT] batch=" << batch_number
              << " version=" << log_version
              << " rows=[" << batch_begin << "," << end << ")"
              << " active_rows=" << log_active_rows
              << " frozen_rows=" << log_frozen_rows
              << " transform_ms=" << transform_ms
              << " insert_ms=" << insert_ms
              << " batch_ms=" << batch_ms
              << " elapsed_ms="
              << std::chrono::duration<double, std::milli>(Clock::now() - measurement_start).count()
              << " pending=" << (log_pending ? "true" : "false") << std::endl;
    request_maintenance(request_merge, request_rebuild);
  }
  if (!failed.load()) insert_admission_gate();
  insert_wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - insert_stream_start).count();
  {
    std::unique_lock<std::mutex> lock(maintenance_mutex);
    maintenance_cv.notify_one();
  }
  {
    std::unique_lock<std::mutex> lock(maintenance_mutex);
    maintenance_cv.wait(lock, [&] {
      std::lock_guard<std::mutex> state_lock(state_mutex);
      return !state.frozen.has_value() || failed.load();
    });
    maintenance_stop = true;
  }
  maintenance_cv.notify_one();
  maintenance.join();
  stop_queries = true;
  workload_entry_cv.notify_all();
  for (auto& worker : query_workers) worker.join();
  const double measurement_wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - measurement_start).count();
  if (failed) { std::cerr << failure << '\n'; return 1; }

  std::vector<double> latencies;
  double recall = 0.0;
  for (const QueryRecord& record : query_records) {
    latencies.push_back(record.latency_ms);
    if (!gt.empty()) {
      const uint32_t version = static_cast<uint32_t>(std::min<uint64_t>(record.version, gt.size() - 1));
      const auto& expected = gt[version][record.query_id];
      uint32_t hits = 0;
      for (DocId id : record.returned_topk)
        if (std::find(expected.begin(), expected.end(), id) != expected.end()) ++hits;
      recall += static_cast<double>(hits) / config.topk;
    }
  }
  if (!query_records.empty() && !gt.empty()) recall /= query_records.size();
  const std::string summary_path = argc == 5 ? argv[4] : "concurrent_summary.json";
  std::ofstream summary(summary_path);
  summary << "{\n"
          << "  \"query_workers\": " << config.concurrent_query_workers << ",\n"
          << "  \"query_threads_per_request\": " << config.concurrent_query_threads_per_request << ",\n"
          << "  \"insert_threads\": " << config.concurrent_insert_threads << ",\n"
          << "  \"maintenance_threads\": " << config.concurrent_maintenance_threads << ",\n"
          << "  \"target_insert_vecps\": " << config.concurrent_target_insert_vecps << ",\n"
          << "  \"stream_batch_size\": " << config.stream_batch_size << ",\n"
          << "  \"query_count\": " << nq << ",\n"
          << "  \"topk\": " << config.topk << ",\n"
          << "  \"nprobe\": " << config.nprobe << ",\n"
          << "  \"initial_rows\": " << initial_rows << ",\n"
          << "  \"final_committed_rows\": " << current_committed.load() << ",\n"
          << "  \"measurement_wall_ms\": " << measurement_wall_ms << ",\n"
          << "  \"completed_queries\": " << query_records.size() << ",\n"
          << "  \"qps\": " << (measurement_wall_ms > 0 ? query_records.size() * 1000.0 / measurement_wall_ms : 0.0) << ",\n"
          << "  \"recall_at_k\": ";
  if (gt.empty()) summary << "null,\n";
  else summary << recall << ",\n";
  summary
          << "  \"latency_mean_ms\": " << (latencies.empty() ? 0.0 : std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size()) << ",\n"
          << "  \"latency_p50_ms\": " << Percentile(latencies, .50) << ",\n"
          << "  \"latency_p95_ms\": " << Percentile(latencies, .95) << ",\n"
          << "  \"latency_p99_ms\": " << Percentile(latencies, .99) << ",\n"
          << "  \"latency_p999_ms\": " << Percentile(latencies, .999) << ",\n"
          << "  \"latency_max_ms\": " << (latencies.empty() ? 0.0 : *std::max_element(latencies.begin(), latencies.end())) << ",\n"
          << "  \"main_route_queries\": " << main_queries << ",\n"
          << "  \"frozen_route_queries\": " << frozen_queries << ",\n"
          << "  \"active_route_queries\": " << active_queries << ",\n"
          << "  \"inserted_vectors\": " << inserted_vectors << ",\n"
          << "  \"committed_batches\": " << committed_batches << ",\n"
          << "  \"actual_insert_vecps\": " << (insert_wall_ms > 0 ? inserted_vectors * 1000.0 / insert_wall_ms : 0.0) << ",\n"
          << "  \"insert_backpressure_events\": " << insert_backpressure_events << ",\n"
          << "  \"insert_backpressure_total_ms\": " << insert_backpressure_total_ms << ",\n"
          << "  \"insert_backpressure_max_ms\": " << insert_backpressure_max_ms << ",\n"
          << "  \"insert_backpressure_ratio\": "
          << (measurement_wall_ms > 0 ? insert_backpressure_total_ms / measurement_wall_ms : 0.0) << ",\n"
          << "  \"insert_batch_mean_ms\": " << (insert_batch_ms.empty() ? 0.0 : std::accumulate(insert_batch_ms.begin(), insert_batch_ms.end(), 0.0) / insert_batch_ms.size()) << ",\n"
          << "  \"insert_batch_p99_ms\": " << Percentile(insert_batch_ms, .99) << ",\n"
          << "  \"max_insert_lag_ms\": " << max_insert_lag_ms << ",\n"
          << "  \"merge_count\": " << merge_count << ",\n"
          << "  \"merge_total_ms\": " << merge_total_ms << ",\n"
          << "  \"merge_compute_ms\": " << merge_compute_ms << ",\n"
          << "  \"merge_codebook_rebuild_ms\": " << merge_codebook_rebuild_ms << ",\n"
          << "  \"merge_commit_ms\": " << merge_commit_ms << ",\n"
          << "  \"merge_patch_partitions\": " << merge_patch_partitions << ",\n"
          << "  \"merge_append_partitions\": " << merge_append_partitions << ",\n"
          << "  \"merge_recluster_partitions\": " << merge_recluster_partitions << ",\n"
          << "  \"maintenance_busy_ratio\": " << (measurement_wall_ms > 0 ? (merge_total_ms + rebuild_total_ms) / measurement_wall_ms : 0.0) << ",\n"
          << "  \"active_rows_over_trigger\": " << active_rows_over_trigger << ",\n"
          << "  \"global_rebuild_count\": " << rebuild_count << ",\n"
          << "  \"global_rebuild_total_ms\": " << rebuild_total_ms << "\n}\n";
  if (!summary) {
    std::cerr << "Failed writing summary file: " << summary_path << '\n';
    return 1;
  }

  const std::filesystem::path summary_file(summary_path);
  const std::string summary_stem = summary_file.stem().string();
  const auto timeline_path = [&](const std::string& suffix) {
    return summary_file.parent_path() / (summary_stem + suffix);
  };
  const auto timeseries_path = timeline_path("_timeseries.csv");
  const auto insert_path = timeline_path("_insert.csv");
  const auto events_path = timeline_path("_events.csv");

  struct QueryBucket {
    std::vector<double> latencies;
  };
  const size_t bucket_count = measurement_wall_ms > 0.0
      ? static_cast<size_t>(std::ceil(measurement_wall_ms / kTimelineBucketMs)) : 0;
  std::vector<QueryBucket> query_buckets(bucket_count);
  for (const QueryRecord& record : query_records) {
    if (bucket_count == 0) break;
    const size_t bucket = std::min(
        static_cast<size_t>(record.completion_ms / kTimelineBucketMs), bucket_count - 1);
    query_buckets[bucket].latencies.push_back(record.latency_ms);
  }

  std::ofstream timeseries(timeseries_path);
  if (!timeseries) {
    std::cerr << "Failed to open timeline CSV: " << timeseries_path << '\n';
    return 1;
  }
  timeseries << std::setprecision(12)
             << "window_start_ms,window_end_ms,query_count,query_qps,"
                "query_latency_mean_ms,query_latency_p50_ms,query_latency_p95_ms,"
                "query_latency_p99_ms,query_latency_max_ms\n";
  for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
    const double window_start_ms = bucket * kTimelineBucketMs;
    // The final bucket uses the measured remainder of the workload duration.
    const double window_end_ms = std::min(measurement_wall_ms,
                                          window_start_ms + kTimelineBucketMs);
    const double window_seconds = (window_end_ms - window_start_ms) / 1000.0;
    const auto& bucket_latencies = query_buckets[bucket].latencies;
    const double latency_mean = bucket_latencies.empty() ? 0.0
        : std::accumulate(bucket_latencies.begin(), bucket_latencies.end(), 0.0) /
              bucket_latencies.size();
    const double latency_max = bucket_latencies.empty() ? 0.0
        : *std::max_element(bucket_latencies.begin(), bucket_latencies.end());
    timeseries << window_start_ms << ',' << window_end_ms << ','
               << bucket_latencies.size() << ','
               << (window_seconds > 0.0 ? bucket_latencies.size() / window_seconds : 0.0) << ','
               << latency_mean << ',' << Percentile(bucket_latencies, .50) << ','
               << Percentile(bucket_latencies, .95) << ','
               << Percentile(bucket_latencies, .99) << ',' << latency_max << '\n';
  }
  timeseries.close();
  if (!timeseries) {
    std::cerr << "Failed writing timeline CSV: " << timeseries_path << '\n';
    return 1;
  }

  std::ofstream inserts(insert_path);
  if (!inserts) {
    std::cerr << "Failed to open insert timeline CSV: " << insert_path << '\n';
    return 1;
  }
  inserts << std::setprecision(12)
          << "batch,version,begin_row,end_row,start_ms,end_ms,transform_ms,insert_ms,batch_ms\n";
  for (const InsertTimelineRecord& record : insert_timeline_records) {
    inserts << record.batch << ',' << record.version << ','
            << record.begin_row << ',' << record.end_row << ','
            << record.start_ms << ',' << record.end_ms << ','
            << record.transform_ms << ',' << record.insert_ms << ','
            << record.batch_ms << '\n';
  }
  inserts.close();
  if (!inserts) {
    std::cerr << "Failed writing insert timeline CSV: " << insert_path << '\n';
    return 1;
  }

  std::ofstream events(events_path);
  if (!events) {
    std::cerr << "Failed to open event timeline CSV: " << events_path << '\n';
    return 1;
  }
  events << std::setprecision(12)
         << "type,id,request_ms,start_ms,compute_done_ms,commit_wait_start_ms,"
            "commit_lock_acquired_ms,commit_done_ms,end_ms,rows,main_rows,"
            "prepare_publish_us,commit_validation_us,commit_docmap_us,commit_partition_swap_us\n";
  for (const TimelineEvent& event : timeline_events) {
    events << (event.type == TimelineEventType::kMerge ? "merge" : "rebuild")
           << ',' << event.id << ',' << event.request_ms << ',' << event.start_ms << ','
           << event.compute_done_ms << ',' << event.commit_wait_start_ms << ','
           << event.commit_lock_acquired_ms << ',' << event.commit_done_ms << ','
           << event.end_ms << ',' << event.rows << ',' << event.main_rows << ','
           << event.prepare_publish_us << ',' << event.commit_validation_us << ','
           << event.commit_docmap_us << ',' << event.commit_partition_swap_us << '\n';
  }
  events.close();
  if (!events) {
    std::cerr << "Failed writing event timeline CSV: " << events_path << '\n';
    return 1;
  }
  std::cout << "[CONCURRENT] summary=" << summary_path << ", completed_queries="
            << query_records.size() << ", qps="
            << (measurement_wall_ms > 0 ? query_records.size() * 1000.0 / measurement_wall_ms : 0.0)
            << ", recall_at_k=";
  if (gt.empty()) std::cout << "n/a\n";
  else std::cout << recall << '\n';
  return summary ? 0 : 1;
}
