#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>

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

}  // namespace

struct EvalMetrics {
  double recall{0.0};
  double whitening_p50{0.0};
  double whitening_p99{0.0};
  double avg_whiten_ms{0.0};
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
};

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1) ? argv[1] : "configs/base.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  std::optional<std::string> dataset_spec = (argc > 2) ? std::optional<std::string>(argv[2]) : std::nullopt;
  std::optional<std::string> query_spec = (argc > 3) ? std::optional<std::string>(argv[3]) : std::nullopt;
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
    } else {
      dataset_label = spec_path.filename().string();
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
    MatrixRM limited = Q.topRows(config.max_queries);
    Q = limited;
    nq = config.max_queries;
  }

  if (!config.use_whitening) {
    std::cout << "[WARN] use_whitening is false in config; this eval will still use whitening."
              << std::endl;
  }

  SearchParams base_params;
  base_params.topk = config.topk;
  base_params.nprobe = config.nprobe;
  base_params.use_whitening = false;
  base_params.enable_dual_route = config.enable_dual_route;

  auto run_experiment = [&]() -> Result<EvalMetrics> {
    auto ivf = CreateIVFIndex();
    auto searcher_res = CreateHybridSearcher(config);
    if (!searcher_res.ok()) {
      return searcher_res.status();
    }
    std::unique_ptr<HybridSearcher> searcher = std::move(searcher_res.value());
    std::vector<DocId> ids(nx);
    std::iota(ids.begin(), ids.end(), 0);

    MatrixRM X_index;
    std::shared_ptr<WhiteningModel> whitening = CreateWhiteningModel();
    VersionId whiten_version = 0;
    double rebuild_ms = 0.0;
    VersionSet versions;

    Timer rebuild_timer;
    auto version_res = whitening->Fit(X);
    if (!version_res.ok()) {
      return version_res.status();
    }
    whiten_version = version_res.value();
    auto base_batch = whitening->TransformBatch(X, whiten_version);
    if (!base_batch.ok()) {
      return base_batch.status();
    }
    X_index = base_batch.value();

    IVFParams ivf_params;
    ivf_params.nlist = std::max(1u, config.ivf_nlist);
    ivf_params.dim = config.dim;
    ivf_params.pq.enable = config.pq_enable;
    ivf_params.pq.M = config.pq_m;
    ivf_params.pq.nbits = config.pq_nbits;
    ivf_params.pq.residual = config.pq_residual;
    auto ivf_version_res = ivf->Build(X_index, ids, ivf_params, 0);
    if (!ivf_version_res.ok()) {
      return ivf_version_res.status();
    }
    versions = VersionSet{whiten_version, ivf_version_res.value()};
    AlignedVector<VectorRecord> records;
    records.reserve(static_cast<size_t>(X_index.rows()));
    const int64_t list_count = std::max<int64_t>(1, static_cast<int64_t>(ivf_params.nlist));
    for (int64_t i = 0; i < X_index.rows(); ++i) {
      VectorRecord rec;
      rec.doc_id = ids[static_cast<size_t>(i)];
      rec.dim = config.dim;
      rec.versions = versions;
      rec.ivf_id = static_cast<uint32_t>(i % list_count);
      rec.x = X_index.row(i).transpose();
      records.push_back(std::move(rec));
    }
    Status add_status = ivf->Add(records);
    if (!add_status.ok()) {
      return add_status;
    }
    Status index_status = searcher->SetIndex(ivf, versions);
    if (!index_status.ok()) {
      return index_status;
    }
    rebuild_ms = rebuild_timer.ElapsedMillis();

    auto query_batch = whitening->TransformBatch(Q, whiten_version);
    if (!query_batch.ok()) {
      return query_batch.status();
    }
    MatrixRM Q_whitened = query_batch.value();
    auto gt_res = ExactSearchBatch(Q_whitened, X_index, config.topk);
    if (!gt_res.ok()) {
      return gt_res.status();
    }
    const auto& ground_truth = gt_res.value();

    SearchParams params = base_params;
    params.use_whitening = false;

    std::vector<std::vector<DocId>> predictions(nq);
    std::vector<double> whitening_ms(nq, 0.0);
    std::vector<double> search_ms(nq, 0.0);
    std::vector<double> total_ms(nq, 0.0);
    std::vector<double> scanned_counts(nq, 0.0);
    std::atomic<bool> failed{false};
    std::mutex error_mu;
    Status error_status;

    Timer wall_timer;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(nq); ++i) {
      if (failed.load()) {
        continue;
      }
      Eigen::VectorXf qvec = Q.row(i).transpose();
      double whiten_elapsed = 0.0;
      const Eigen::VectorXf* query_ptr = &qvec;
      static thread_local Eigen::VectorXf tls_whiten_buf;
      if (tls_whiten_buf.size() != qvec.size()) {
        tls_whiten_buf.resize(qvec.size());
      }
      Timer wtimer;
      auto wstatus = whitening->Transform(qvec, whiten_version, tls_whiten_buf);
      whiten_elapsed = wtimer.ElapsedMillis();
      if (!wstatus.ok()) {
        std::lock_guard<std::mutex> lock(error_mu);
        if (!failed.exchange(true)) {
          error_status = wstatus.status();
        }
        continue;
      }
      query_ptr = &tls_whiten_buf;
      Timer search_timer;
      auto search_res = searcher->Search(*query_ptr, params);
      double search_elapsed = search_timer.ElapsedMillis();
      if (!search_res.ok()) {
        std::lock_guard<std::mutex> lock(error_mu);
        if (!failed.exchange(true)) {
          error_status = search_res.status();
        }
        continue;
      }
      std::vector<DocId> row;
      row.reserve(search_res.value().topk.size());
      for (const auto& cand : search_res.value().topk) {
        row.push_back(cand.doc_id);
      }
      predictions[i] = std::move(row);
      whitening_ms[i] = whiten_elapsed;
      search_ms[i] = search_elapsed;
      total_ms[i] = whiten_elapsed + search_elapsed;
      scanned_counts[i] = static_cast<double>(search_res.value().scanned_candidates);
    }

    double wall_elapsed_ms = wall_timer.ElapsedMillis();

    if (failed.load()) {
      return error_status;
    }

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
    double avg_total = 0.0;
    if (nq > 0) {
      avg_whiten = std::accumulate(whitening_ms.begin(), whitening_ms.end(), 0.0) /
                   static_cast<double>(nq);
      avg_total = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / static_cast<double>(nq);
    }
    auto percentile = [](std::vector<double> values, double q) -> double {
      if (values.empty()) {
        return 0.0;
      }
      std::sort(values.begin(), values.end());
      double idx = q * (values.size() - 1);
      size_t lo = static_cast<size_t>(std::floor(idx));
      size_t hi = static_cast<size_t>(std::ceil(idx));
      double frac = idx - lo;
      if (hi >= values.size()) {
        hi = values.size() - 1;
      }
      return values[lo] + (values[hi] - values[lo]) * frac;
    };
    double scanned_avg = 0.0;
    double scanned_p50 = 0.0;
    double scanned_p99 = 0.0;
    double scanned_max = 0.0;
    if (nq > 0) {
      scanned_avg = std::accumulate(scanned_counts.begin(), scanned_counts.end(), 0.0) /
                    static_cast<double>(nq);
      scanned_p50 = percentile(scanned_counts, 0.50);
      scanned_p99 = percentile(scanned_counts, 0.99);
      scanned_max = *std::max_element(scanned_counts.begin(), scanned_counts.end());
    }
    double qps =
        (wall_elapsed_ms > 0.0) ? (static_cast<double>(nq) / (wall_elapsed_ms / 1000.0)) : 0.0;

    EvalMetrics metrics;
    metrics.recall = recall_res.value();
    metrics.whitening_p50 = whiten_summary.value().p50_ms;
    metrics.whitening_p99 = whiten_summary.value().p99_ms;
    metrics.avg_whiten_ms = avg_whiten;
    metrics.search_p50 = search_summary.value().p50_ms;
    metrics.search_p99 = search_summary.value().p99_ms;
    metrics.total_p50 = total_summary.value().p50_ms;
    metrics.total_p99 = total_summary.value().p99_ms;
    metrics.avg_query_ms = avg_total;
    metrics.qps = qps;
    metrics.rebuild_ms = rebuild_ms;
    metrics.scanned_avg = scanned_avg;
    metrics.scanned_p50 = scanned_p50;
    metrics.scanned_p99 = scanned_p99;
    metrics.scanned_max = scanned_max;
    return metrics;
  };

  auto save_metrics = [&](const EvalMetrics& metrics) {
    std::filesystem::path results_dir = std::filesystem::path("result") / dataset_label;
    std::error_code ec;
    std::filesystem::create_directories(results_dir, ec);
    const auto now = std::chrono::system_clock::now();
    const auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string file_name = "topk" + std::to_string(config.topk) + "_np" +
                            std::to_string(base_params.nprobe) + "_nl" +
                            std::to_string(config.ivf_nlist) + "_zca_" +
                            std::to_string(ts) + ".json";
    std::filesystem::path result_path = results_dir / file_name;
    std::ofstream ofs(result_path);
    if (ofs) {
      ofs << "{\n";
      ofs << "  \"dataset\": \"" << dataset_label << "\",\n";
      ofs << "  \"timestamp\": " << ts << ",\n";
      ofs << "  \"use_whitening\": true,\n";
      ofs << "  \"params\": {\n";
      ofs << "    \"topk\": " << config.topk << ",\n";
      ofs << "    \"nprobe\": " << base_params.nprobe << ",\n";
      ofs << "    \"nlist\": " << config.ivf_nlist << ",\n";
      ofs << "    \"use_whitening\": true\n";
      ofs << "  },\n";
      ofs << "  \"metrics\": {\n";
      ofs << "    \"avg_query_ms\": " << metrics.avg_query_ms << ",\n";
      ofs << "    \"qps\": " << metrics.qps << ",\n";
      ofs << "    \"recall@" << config.topk << "\": " << metrics.recall << ",\n";
      ofs << "    \"rebuild_ms\": " << metrics.rebuild_ms << "\n";
      ofs << "  }\n";
      ofs << "}\n";
      std::cout << "Saved metrics to " << result_path << std::endl;
    } else {
      std::cerr << "Failed to write results to " << result_path << std::endl;
    }
  };

  auto metrics_res = run_experiment();
  if (!metrics_res.ok()) {
    std::cerr << metrics_res.status().ToString() << std::endl;
    return 1;
  }
  const EvalMetrics& metrics = metrics_res.value();
  std::cout << "[FULL BUILD] "
            << "Recall@" << config.topk << " = " << metrics.recall
            << " (nprobe=" << base_params.nprobe << ")" << std::endl;
  std::cout << "Avg query=" << metrics.avg_query_ms << "ms; "
            << "Search p50=" << metrics.search_p50 << "ms, p99=" << metrics.search_p99 << "ms; "
            << "Total p50=" << metrics.total_p50 << "ms, p99=" << metrics.total_p99 << "ms; "
            << "Whitening p50=" << metrics.whitening_p50 << "ms, p99=" << metrics.whitening_p99
            << "ms, avg=" << metrics.avg_whiten_ms << "ms; "
            << "Rebuild=" << metrics.rebuild_ms << "ms; "
            << "Scanned avg=" << metrics.scanned_avg << ", p50=" << metrics.scanned_p50
            << ", p99=" << metrics.scanned_p99 << ", max=" << metrics.scanned_max << "; "
            << "QPS=" << metrics.qps << std::endl;

  save_metrics(metrics);

  return 0;
}
