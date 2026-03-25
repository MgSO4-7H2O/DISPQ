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
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "search/exact_search.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

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
};

struct SnapshotRecord {
  uint32_t base_rows{0};
  uint32_t main_rows{0};
  uint32_t delta_rows{0};
  uint32_t active_delta_docs{0};
  double recall{0.0};
  double avg_search_ms{0.0};
  double avg_scanned{0.0};
  double qps{0.0};
  double update_ms{0.0};
  double nqe_batch{0.0};
  double qe_ratio{1.0};
  double codebook_drift{0.0};
  bool pq_updated{false};
  bool in_warmup{false};
  uint32_t warmup_batches_left{0};
};

struct SearchRoute {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint8_t from_new{0};
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

Result<DeltaShard> BuildEmptyShard(Eigen::Ref<const MatrixRM> train_data,
                                   const IVFParams& ivf_params,
                                   VersionId whiten_version,
                                   uint32_t shard_id,
                                   Eigen::Ref<const MatrixRM> shared_centroids,
                                   const std::optional<PQRuntimeState>& main_pq_state) {
  if (train_data.rows() == 0 || train_data.cols() == 0) {
    return Status::InvalidArgument("BuildEmptyShard: empty training data");
  }
  if (shared_centroids.rows() == 0 || shared_centroids.cols() != train_data.cols()) {
    return Status::InvalidArgument("BuildEmptyShard: invalid shared_centroids");
  }
  auto ivf = CreateIVFIndex();
  std::vector<DocId> ids(static_cast<size_t>(train_data.rows()));
  std::iota(ids.begin(), ids.end(), 0);
  IVFParams delta_params = ivf_params;
  delta_params.use_fixed_routing_centroids = true;
  delta_params.fixed_routing_centroids = shared_centroids;
  if (main_pq_state.has_value() && main_pq_state->use_pq && !main_pq_state->codebooks.empty()) {
    delta_params.use_fixed_pq_codebooks = true;
    delta_params.fixed_pq_codebooks = main_pq_state->codebooks;
    delta_params.fixed_pq_counts = main_pq_state->counts;
    delta_params.fixed_pq_baseline_nqe = main_pq_state->nqe_baseline;
    delta_params.fixed_pq_ema_nqe = main_pq_state->nqe_ema;
  }
  auto version_res = ivf->Build(train_data, ids, delta_params, 0);
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

SearchResult MergeTopK(const std::vector<SearchResult>& partial_results, uint32_t topk) {
  SearchResult out;
  if (partial_results.empty() || topk == 0) {
    return out;
  }
  std::unordered_map<DocId, Candidate> best_by_doc;
  uint64_t scanned = 0;
  for (const auto& part : partial_results) {
    scanned += part.scanned_candidates;
    for (const auto& cand : part.topk) {
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
  if (merged.size() > topk) {
    std::nth_element(merged.begin(),
                     merged.begin() + static_cast<int64_t>(topk),
                     merged.end(),
                     [](const Candidate& a, const Candidate& b) {
                       return a.approx_dist < b.approx_dist;
                     });
    merged.resize(topk);
  }
  std::sort(merged.begin(), merged.end(),
            [](const Candidate& a, const Candidate& b) { return a.approx_dist < b.approx_dist; });
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

  const uint32_t active_docs = (active_delta.has_value() ? active_delta->rows : 0u);
  if (active_docs > seen_rows) {
    return Status::InvalidArgument("EvaluateState: invalid active docs");
  }
  const uint32_t searchable_rows = main_rows + active_docs;
  MatrixRM searchable_db(searchable_rows, base_whitened.cols());
  std::vector<DocId> searchable_doc_ids(static_cast<size_t>(searchable_rows));
  if (main_rows > 0) {
    searchable_db.topRows(main_rows) = base_whitened.topRows(main_rows);
    for (uint32_t i = 0; i < main_rows; ++i) {
      searchable_doc_ids[static_cast<size_t>(i)] = i;
    }
  }
  if (active_docs > 0) {
    const uint32_t active_begin = seen_rows - active_docs;
    searchable_db.middleRows(main_rows, active_docs) =
        base_whitened.middleRows(active_begin, active_docs);
    for (uint32_t i = 0; i < active_docs; ++i) {
      searchable_doc_ids[static_cast<size_t>(main_rows + i)] = active_begin + i;
    }
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
  routes.push_back(SearchRoute{main_ivf, main_versions, 0});
  if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
    routes.push_back(SearchRoute{active_delta->ivf, active_delta->versions, 1});
  }

  const uint32_t nq = static_cast<uint32_t>(queries_raw.rows());
  const uint32_t route_count = static_cast<uint32_t>(routes.size());
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
          params.topk,
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
    }
  }

  if (failed.load()) {
    return error_status;
  }

  for (uint32_t qi = 0; qi < nq; ++qi) {
    SearchResult merged = MergeTopK(route_results[static_cast<size_t>(qi)], params.topk);
    double q_search_ms = 0.0;
    for (uint32_t ri = 0; ri < route_count; ++ri) {
      q_search_ms = std::max(
          q_search_ms,
          route_search_ms[static_cast<size_t>(qi) * route_count + static_cast<size_t>(ri)]);
    }
    std::vector<DocId> row;
    row.reserve(merged.topk.size());
    for (const auto& cand : merged.topk) {
      row.push_back(cand.doc_id);
    }
    predictions[static_cast<size_t>(qi)] = std::move(row);
    search_ms[static_cast<size_t>(qi)] = q_search_ms;
    total_ms[static_cast<size_t>(qi)] = whitening_ms[static_cast<size_t>(qi)] + q_search_ms;
    scanned_counts[static_cast<size_t>(qi)] = static_cast<double>(merged.scanned_candidates);
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

  auto percentile = [](std::vector<double> values, double q) -> double {
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
  };

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
    scanned_p50 = percentile(scanned_counts, 0.50);
    scanned_p99 = percentile(scanned_counts, 0.99);
    scanned_max = *std::max_element(scanned_counts.begin(), scanned_counts.end());
  }

  const double qps =
      wall_elapsed_ms > 0.0 ? (static_cast<double>(nq) / (wall_elapsed_ms / 1000.0)) : 0.0;

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
  const uint32_t total_stream_rows = nx > main_rows_initial ? nx - main_rows_initial : 0;
  std::cout << "[INFO] main_rows=" << main_rows_initial
            << ", stream_rows=" << total_stream_rows
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
  MatrixRM main_routing_centroids;

  MatrixRM X_whitened;
  MatrixRM Q_whitened;
  uint32_t main_rows_current = main_rows_initial;
  uint32_t next_insert_idx = main_rows_initial;

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
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  if (!centroids_res.ok()) {
    std::cerr << centroids_res.status().ToString() << std::endl;
    return 1;
  }
  main_routing_centroids = centroids_res.value();

  std::optional<PQRuntimeState> main_pq_state;
  if (config.pq_enable && config.pq_residual) {
    auto pq_state_res = main_ivf->GetPQRuntimeState(main_versions);
    if (!pq_state_res.ok()) {
      std::cerr << pq_state_res.status().ToString() << std::endl;
      return 1;
    }
    if (pq_state_res.value().use_pq) {
      main_pq_state = pq_state_res.value();
    }
  }
  double rebuild_ms_total = init_timer.ElapsedMillis();

  std::optional<DeltaShard> active_delta;
  if (config.enable_streaming && total_stream_rows > 0) {
    auto active_res = BuildEmptyShard(main_train,
                                      ivf_params,
                                      whiten_version,
                                      1,
                                      main_routing_centroids,
                                      main_pq_state);
    if (!active_res.ok()) {
      std::cerr << active_res.status().ToString() << std::endl;
      return 1;
    }
    active_delta = active_res.value();
  }

  OnlinePQUpdateOptions online_pq_options;
  online_pq_options.enable = config.online_pq_enable && config.pq_enable && config.pq_residual;
  online_pq_options.qe_ratio_threshold = config.online_pq_qe_ratio_threshold;
  online_pq_options.ema_alpha = config.online_pq_ema_alpha;
  online_pq_options.nqe_eps = config.online_pq_eps;
  online_pq_options.warmup_enable = config.online_pq_warmup_enable;
  online_pq_options.warmup_batches = config.online_pq_warmup_batches;
  online_pq_options.partial_top_alpha = config.online_pq_partial_top_alpha;
  online_pq_options.partial_alpha = config.online_pq_alpha;
  online_pq_options.partial_top_lambda = config.online_pq_partial_top_lambda;
  online_pq_options.partial_lambda = config.online_pq_lambda;
  online_pq_options.reencode_batch_after_update = config.online_pq_reencode_batch;

  OnlinePQRollup online_pq_rollup;
  OnlinePQUpdateStats last_online_pq_stats;

  std::filesystem::path results_dir = std::filesystem::path("result") / dataset_label;
  std::error_code ec;
  std::filesystem::create_directories(results_dir, ec);
  const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

  const bool collect_snapshots = config.enable_streaming && total_stream_rows > 0 &&
                                 config.snapshot_interval > 0;
  std::vector<SnapshotRecord> snapshots;

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
                            const OnlinePQUpdateStats& pq_stats) {
    if (!collect_snapshots) {
      return;
    }
    SnapshotRecord snap;
    snap.base_rows = active_rows;
    snap.main_rows = main_rows_current;
    snap.delta_rows = active_rows > main_rows_initial ? active_rows - main_rows_initial : 0;
    snap.active_delta_docs = active_delta.has_value() ? active_delta->rows : 0;
    snap.recall = metrics.recall;
    snap.avg_search_ms = metrics.avg_search_ms;
    snap.avg_scanned = metrics.scanned_avg;
    snap.qps = metrics.qps;
    snap.update_ms = update_ms;
    snap.nqe_batch = pq_stats.nqe_batch;
    snap.qe_ratio = pq_stats.qe_ratio;
    snap.codebook_drift = pq_stats.codebook_drift_l2;
    snap.pq_updated = pq_stats.updated_codebook;
    snap.in_warmup = pq_stats.in_warmup;
    snap.warmup_batches_left = pq_stats.warmup_batches_left;
    snapshots.push_back(std::move(snap));
  };

  uint32_t inserted_rows = 0;
  double total_update_ms = 0.0;
  double pending_update_ms = 0.0;

  if (collect_snapshots) {
    auto mres = evaluate_rows(next_insert_idx);
    if (!mres.ok()) {
      std::cerr << mres.status().ToString() << std::endl;
      return 1;
    }
    write_snapshot(next_insert_idx, mres.value(), 0.0, last_online_pq_stats);
  }

  if (config.enable_streaming && total_stream_rows > 0) {
    const bool use_stream_batch_size =
        config.streaming_mode == "batch" ||
        (config.streaming_mode == "streaming" && config.streaming_use_stream_batch_size);
    const uint32_t insert_step = use_stream_batch_size ? std::max(1u, config.stream_batch_size) : 1u;
    uint32_t next_snapshot_target = std::max(1u, config.snapshot_interval);

    while (next_insert_idx < nx) {
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
        auto add_res = AddRangeToIndexWithOnlinePQ(active_delta->ivf,
                                                   X_whitened,
                                                   begin,
                                                   end,
                                                   config.dim,
                                                   active_delta->versions,
                                                   online_pq_options);
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
        active_delta->rows += chunk;
      }
      const double step_update_ms = update_timer.ElapsedMillis();

      next_insert_idx = end;
      inserted_rows = next_insert_idx - main_rows_initial;
      total_update_ms += step_update_ms;
      pending_update_ms += step_update_ms;

      const bool hit_periodic_snapshot =
          collect_snapshots && inserted_rows == next_snapshot_target;
      const bool hit_final_snapshot = collect_snapshots && inserted_rows == total_stream_rows;
      if (hit_periodic_snapshot) {
        while (next_snapshot_target <= inserted_rows) {
          next_snapshot_target += config.snapshot_interval;
        }
      }
      if (hit_periodic_snapshot || hit_final_snapshot) {
        auto mres = evaluate_rows(next_insert_idx);
        if (!mres.ok()) {
          std::cerr << mres.status().ToString() << std::endl;
          return 1;
        }
        write_snapshot(next_insert_idx, mres.value(), pending_update_ms, last_online_pq_stats);
        pending_update_ms = 0.0;
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
            << " (nprobe=" << params.nprobe << ")" << std::endl;
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
  ofs << "    \"enable_streaming\": " << (config.enable_streaming ? "true" : "false") << ",\n";
  ofs << "    \"main_index_rows_initial\": " << main_rows_initial << ",\n";
  ofs << "    \"main_rows_final\": " << main_rows_current << ",\n";
  ofs << "    \"stream_rows\": " << total_stream_rows << ",\n";
  ofs << "    \"streaming_mode\": \""
      << (config.enable_streaming ? config.streaming_mode : "offline") << "\",\n";
  ofs << "    \"stream_batch_size\": " << config.stream_batch_size << ",\n";
  ofs << "    \"streaming_use_stream_batch_size\": "
      << (config.streaming_use_stream_batch_size ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_enable\": " << (config.online_pq_enable ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_qe_ratio_threshold\": " << config.online_pq_qe_ratio_threshold << ",\n";
  ofs << "    \"online_pq_ema_alpha\": " << config.online_pq_ema_alpha << ",\n";
  ofs << "    \"online_pq_eps\": " << config.online_pq_eps << ",\n";
  ofs << "    \"online_pq_warmup_enable\": "
      << (config.online_pq_warmup_enable ? "true" : "false") << ",\n";
  ofs << "    \"online_pq_warmup_batches\": " << config.online_pq_warmup_batches << ",\n";
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
  ofs << "    \"scanned_avg\": " << final_metrics.scanned_avg << "\n";
  ofs << "  },\n";
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
  ofs << "  \"snapshots\": [\n";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto& snap = snapshots[i];
    ofs << "    {\n";
    ofs << "      \"base_rows\": " << snap.base_rows << ",\n";
    ofs << "      \"main_rows\": " << snap.main_rows << ",\n";
    ofs << "      \"delta_rows\": " << snap.delta_rows << ",\n";
    ofs << "      \"active_delta_docs\": " << snap.active_delta_docs << ",\n";
    ofs << "      \"recall\": " << snap.recall << ",\n";
    ofs << "      \"avg_search_ms\": " << snap.avg_search_ms << ",\n";
    ofs << "      \"avg_scanned\": " << snap.avg_scanned << ",\n";
    ofs << "      \"qps\": " << snap.qps << ",\n";
    ofs << "      \"update_ms\": " << snap.update_ms << ",\n";
    ofs << "      \"nqe_batch\": " << snap.nqe_batch << ",\n";
    ofs << "      \"qe_ratio\": " << snap.qe_ratio << ",\n";
    ofs << "      \"codebook_drift\": " << snap.codebook_drift << ",\n";
    ofs << "      \"pq_updated\": " << (snap.pq_updated ? "true" : "false") << ",\n";
    ofs << "      \"in_warmup\": " << (snap.in_warmup ? "true" : "false") << ",\n";
    ofs << "      \"warmup_batches_left\": " << snap.warmup_batches_left << "\n";
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
