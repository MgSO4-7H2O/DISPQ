#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/config.h"
#include "common/dataset.h"
#include "common/result.h"
#include "common/status.h"
#include "common/timer.h"
#include "common/types.h"
#include "eval_memory.h"
#include "eval/eval_counter.h"
#include "eval/eval_output.h"
#include "eval/metrics.h"
#include "eval/eval_type.h"
#include "index/ivf.h"
#include "index/merge.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;
using namespace ann::eval::run_eval_ms;

namespace {

constexpr uint32_t kDeltaKMeansIterationsDefault = 10;
constexpr uint32_t kAddBlockRows = 65536;
constexpr uint32_t kWhitenedMinGrowthRows = 65536;
constexpr size_t kGroundTruthBlockTargetBytes = static_cast<size_t>(64) << 20;

void TrimAllocatorRetainedMemory(const char* stage) {
#ifdef __GLIBC__
  const int released = malloc_trim(0);
  std::cout << "[MALLOC_TRIM] stage=" << stage
            << ", released=" << released << std::endl;
#else
  (void)stage;
#endif
}

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

double PercentileValueSorted(const std::vector<double>& values, double q) {
  if (values.empty()) return 0.0;
  const double idx = q * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(idx)];
}

DistributionStatsLite SummarizeValues(const std::vector<double>& values) {
  DistributionStatsLite out;
  if (values.empty()) return out;
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  out.avg = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
            static_cast<double>(sorted.size());
  out.p5 = PercentileValueSorted(sorted, 0.05);
  out.p50 = PercentileValueSorted(sorted, 0.50);
  out.p95 = PercentileValueSorted(sorted, 0.95);
  out.p99 = PercentileValueSorted(sorted, 0.99);
  out.min = sorted.front();
  out.max = sorted.back();
  return out;
}

float SquaredL2FromNormDot(float a_norm, float b_norm, float dot) {
  const float dist = a_norm + b_norm - 2.0f * dot;
  return dist >= 0.0f ? dist : 0.0f;
}

Result<std::string> ResolveFileWithSuffix(const std::string& path_or_dir,
                                          const std::string& required_suffix) {
  namespace fs = std::filesystem;
  fs::path candidate(path_or_dir);
  std::error_code ec;
  if (!fs::exists(candidate, ec)) {
    return Status::NotFound("Path does not exist: " + path_or_dir);
  }
  auto ends_with = [](const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  if (fs::is_regular_file(candidate, ec)) {
    const std::string name = candidate.filename().string();
    if (!required_suffix.empty() && !ends_with(name, required_suffix)) {
      return Status::InvalidArgument("Expected a file ending with " +
                                     required_suffix + ": " + name);
    }
    return candidate.string();
  }
  if (fs::is_directory(candidate, ec)) {
    for (const auto& entry : fs::directory_iterator(candidate, ec)) {
      if (ec) break;
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      if (required_suffix.empty() || ends_with(name, required_suffix)) {
        return entry.path().string();
      }
    }
    return Status::NotFound("No file ending with " + required_suffix +
                            " found under " + path_or_dir);
  }
  return Status::InvalidArgument("Unsupported path type: " + path_or_dir);
}

uint32_t ResolveRouteExactRerankCandidates(const Config& config,
                                           const SearchRoute& route) {
  uint32_t candidates = config.exact_rerank_candidates_per_route;

  if (route.name == "main" && config.main_exact_rerank_candidates > 0) {
    candidates = config.main_exact_rerank_candidates;
  } else if (route.name == "active_delta" &&
             config.active_exact_rerank_candidates > 0) {
    candidates = config.active_exact_rerank_candidates;
  } else if (route.name == "frozen_delta" &&
             config.frozen_exact_rerank_candidates > 0) {
    candidates = config.frozen_exact_rerank_candidates;
  }

  return std::max<uint32_t>(1u, candidates);
}

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

bool ExactDocLess(const ExactDocCandidate& a, const ExactDocCandidate& b) {
  if (a.dist != b.dist) return a.dist < b.dist;
  return a.doc_id < b.doc_id;
}

bool MaxHeapWorse(const ExactDocCandidate& a, const ExactDocCandidate& b) {
  return ExactDocLess(a, b);
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::optional<std::string> ExtractStringField(const std::string& obj,
                                              const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch m;
  if (std::regex_search(obj, m, re)) return m[1].str();
  return std::nullopt;
}

std::optional<uint32_t> ExtractUintField(const std::string& obj,
                                         const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
  std::smatch m;
  if (std::regex_search(obj, m, re)) {
    return static_cast<uint32_t>(std::stoul(m[1].str()));
  }
  return std::nullopt;
}

std::optional<int32_t> ExtractIntOrNullField(const std::string& obj,
                                             const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+|null)");
  std::smatch m;
  if (std::regex_search(obj, m, re)) {
    if (m[1].str() == "null") return std::nullopt;
    return static_cast<int32_t>(std::stol(m[1].str()));
  }
  return std::nullopt;
}

Result<Workload> LoadWorkloadJson(const std::string& path) {
  Workload wl;
  wl.path = path;

  std::string text;
  try {
    text = ReadTextFile(path);
  } catch (const std::exception& e) {
    return Status::IOError(e.what());
  }

  if (auto v = ExtractStringField(text, "dataset")) wl.dataset = *v;
  if (auto v = ExtractUintField(text, "num_vectors")) wl.num_vectors = *v;
  if (auto v = ExtractUintField(text, "dim")) wl.dim = *v;

  const size_t ops_pos = text.find("\"operations\"");
  if (ops_pos == std::string::npos) {
    return Status::InvalidArgument("workload JSON does not contain operations");
  }

  const size_t arr_begin = text.find('[', ops_pos);
  if (arr_begin == std::string::npos) {
    return Status::InvalidArgument("workload JSON operations is not an array");
  }

  int depth = 0;
  size_t arr_end = std::string::npos;
  for (size_t i = arr_begin; i < text.size(); ++i) {
    if (text[i] == '[') depth++;
    if (text[i] == ']') {
      depth--;
      if (depth == 0) {
        arr_end = i;
        break;
      }
    }
  }

  if (arr_end == std::string::npos) {
    return Status::InvalidArgument("unterminated operations array");
  }

  const std::string arr = text.substr(arr_begin + 1, arr_end - arr_begin - 1);
  std::regex obj_re("\\{[^\\{\\}]*\\}");
  auto begin = std::sregex_iterator(arr.begin(), arr.end(), obj_re);
  auto end = std::sregex_iterator();

  uint32_t ordinal = 1;
  for (auto it = begin; it != end; ++it) {
    const std::string obj = it->str();

    auto op_name = ExtractStringField(obj, "operation");
    if (!op_name) op_name = ExtractStringField(obj, "op");
    if (!op_name) continue;

    WorkloadOp op;
    op.operation = *op_name;
    op.op_id = ExtractUintField(obj, "op_id").value_or(ordinal);
    op.source_op_id = ExtractUintField(obj, "source_op_id").value_or(0);
    op.start = ExtractUintField(obj, "start").value_or(0);
    op.end = ExtractUintField(obj, "end").value_or(op.start);
    op.active_rows = ExtractUintField(obj, "active_rows").value_or(op.end);

    if (auto r = ExtractIntOrNullField(obj, "round")) op.round = *r;
    if (auto c = ExtractIntOrNullField(obj, "cluster")) op.cluster = *c;

    wl.operations.push_back(op);
    ordinal++;
  }

  if (wl.operations.empty()) {
    return Status::InvalidArgument("no operations parsed from workload JSON");
  }

  return wl;
}

void NormalizeRowsL2(MatrixRM* X) {
  if (!X) return;
  for (Eigen::Index i = 0; i < X->rows(); ++i) {
    const float norm = X->row(i).norm();
    if (norm > 0.0f) X->row(i) /= norm;
  }
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
  if (total_rows == 0) return 0;
  if (!config.enable_streaming) return total_rows;

  uint32_t main_rows = config.main_index_rows;
  if (main_rows == 0) main_rows = std::max<uint32_t>(1, total_rows / 2);
  if (total_rows > 1 && main_rows >= total_rows) main_rows = total_rows - 1;

  return std::max<uint32_t>(1, main_rows);
}

uint32_t ResolveDeltaTrainRows(const Config& config, uint32_t rows_after_main) {
  if (!config.enable_streaming || rows_after_main == 0) return 0;

  uint64_t requested =
      static_cast<uint64_t>(config.delta_train_window) *
      static_cast<uint64_t>(std::max<uint32_t>(1, config.stream_batch_size));

  if (requested == 0) requested = std::min<uint32_t>(rows_after_main, 1);

  return static_cast<uint32_t>(
      std::min<uint64_t>(rows_after_main, requested));
}

Status AddRangeToIndex(const std::shared_ptr<IVFIndex>& ivf,
                       const MatrixRM& x_whitened,
                       uint32_t begin,
                       uint32_t end,
                       uint32_t dim,
                       const VersionSet& versions,
                       const std::vector<int>* precomputed_assignments = nullptr) {
  if (!ivf) return Status::InvalidArgument("AddRangeToIndex: null ivf");

  if (begin > end || end > static_cast<uint32_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddRangeToIndex: invalid range");
  }
  if (precomputed_assignments != nullptr &&
      precomputed_assignments->size() != static_cast<size_t>(end - begin)) {
    return Status::InvalidArgument("AddRangeToIndex: assignment size mismatch");
  }

  if (begin == end) return Status::OK();

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
      rec.x = x_whitened.row(static_cast<Eigen::Index>(i)).transpose();
      records.push_back(std::move(rec));
    }

    const int* block_assignments =
        precomputed_assignments != nullptr
            ? precomputed_assignments->data() + (block_begin - begin)
            : nullptr;
    const Status add_status =
        ivf->AddBatch(records, block_end == end, block_assignments);
    if (!add_status.ok()) return add_status;
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

  OnlinePQUpdateStats empty;
  if (begin == end) return empty;

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
    rec.x = x_whitened.row(static_cast<Eigen::Index>(i)).transpose();
    records[static_cast<size_t>(offset)] = std::move(rec);
  }
  const double record_build_ms = record_timer.ElapsedMillis();

  auto add_res = ivf->AddWithOnlinePQ(records, options);
  if (!add_res.ok()) return add_res.status();
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
  if (!version_res.ok()) return version_res.status();

  DeltaShard shard;
  shard.ivf = ivf;
  shard.versions = VersionSet{whiten_version, version_res.value()};
  shard.rows = 0;
  shard.shard_id = shard_id;

  return shard;
}

size_t ResolveGroundTruthBlockRows(Eigen::Index nq,
                                   Eigen::Index dim,
                                   size_t total_docs) {
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

Result<std::vector<std::vector<DocId>>> ExactSearchPrefixBlockwise(
    Eigen::Ref<const MatrixRM> queries,
    Eigen::Ref<const MatrixRM> database,
    uint32_t seen_rows,
    uint32_t topk) {
  if (queries.cols() == 0 || database.cols() == 0) {
    return Status::InvalidArgument("ExactSearchPrefixBlockwise: empty matrices");
  }
  if (queries.cols() != database.cols()) {
    return Status::InvalidArgument("ExactSearchPrefixBlockwise: dimension mismatch");
  }
  if (queries.rows() == 0) {
    return Status::InvalidArgument("ExactSearchPrefixBlockwise: no queries");
  }
  if (seen_rows == 0 || seen_rows > static_cast<uint32_t>(database.rows())) {
    return Status::InvalidArgument("ExactSearchPrefixBlockwise: invalid seen_rows");
  }
  if (topk == 0) {
    return Status::InvalidArgument("ExactSearchPrefixBlockwise: topk must be positive");
  }

  const size_t limit = std::min<size_t>(topk, seen_rows);
  std::vector<std::vector<ExactDocCandidate>> heaps(static_cast<size_t>(queries.rows()));
  for (auto& heap : heaps) {
    heap.reserve(limit);
  }

  const Eigen::VectorXf query_norms = queries.rowwise().squaredNorm();
  const size_t block_rows =
      ResolveGroundTruthBlockRows(queries.rows(), queries.cols(), seen_rows);

  for (uint32_t block_begin = 0; block_begin < seen_rows;
       block_begin += static_cast<uint32_t>(block_rows)) {
    const uint32_t block_count =
        std::min<uint32_t>(static_cast<uint32_t>(block_rows), seen_rows - block_begin);
    const auto block = database.middleRows(static_cast<Eigen::Index>(block_begin),
                                           static_cast<Eigen::Index>(block_count));
    const Eigen::VectorXf block_norms = block.rowwise().squaredNorm();
    const MatrixRM dots = queries * block.transpose();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t qi = 0; qi < static_cast<int64_t>(queries.rows()); ++qi) {
      auto& heap = heaps[static_cast<size_t>(qi)];
      const float qnorm = query_norms(static_cast<Eigen::Index>(qi));

      for (uint32_t bi = 0; bi < block_count; ++bi) {
        float dist = qnorm + block_norms(static_cast<Eigen::Index>(bi)) -
                     2.0f * dots(static_cast<Eigen::Index>(qi),
                                  static_cast<Eigen::Index>(bi));

        if (dist < 0.0f) dist = 0.0f;

        ExactDocCandidate cand{dist, block_begin + bi};

        if (heap.size() < limit) {
          heap.push_back(cand);
          std::push_heap(heap.begin(), heap.end(), MaxHeapWorse);
        } else if (ExactDocLess(cand, heap.front())) {
          std::pop_heap(heap.begin(), heap.end(), MaxHeapWorse);
          heap.back() = cand;
          std::push_heap(heap.begin(), heap.end(), MaxHeapWorse);
        }
      }
    }
  }

  std::vector<std::vector<DocId>> out(static_cast<size_t>(queries.rows()));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t qi64 = 0; qi64 < static_cast<int64_t>(heaps.size()); ++qi64) {
    const size_t qi = static_cast<size_t>(qi64);
    auto& heap = heaps[qi];
    std::sort(heap.begin(), heap.end(), ExactDocLess);

    std::vector<DocId> row;
    row.reserve(heap.size());
    for (const auto& cand : heap) row.push_back(cand.doc_id);
    out[qi] = std::move(row);
  }

  return out;
}

bool ApproxCandidateLess(const Candidate& a, const Candidate& b) {
  if (a.approx_dist != b.approx_dist) {
    return a.approx_dist < b.approx_dist;
  }
  return a.doc_id < b.doc_id;
}

bool RerankCandidateLess(const Candidate& a, const Candidate& b) {
  if (a.rerank_dist != b.rerank_dist) {
    return a.rerank_dist < b.rerank_dist;
  }
  if (a.approx_dist != b.approx_dist) {
    return a.approx_dist < b.approx_dist;
  }
  return a.doc_id < b.doc_id;
}

SearchResult MergeTopKPrefix(const std::vector<SearchResult>& partial_results,
                             uint32_t topk,
                             bool exact_rerank_enable,
                             const std::vector<uint32_t>& rerank_candidates_per_route,
                             Eigen::Ref<const Eigen::VectorXf> query_whitened,
                             const MatrixRM& base_whitened,
                             const Eigen::VectorXf& base_norms,
                             uint32_t seen_rows) {
  SearchResult out;
  if (partial_results.empty() || topk == 0) {
    return out;
  }

  uint64_t scanned = 0;
  size_t total_candidates = 0;
  for (size_t ri = 0; ri < partial_results.size(); ++ri) {
    const SearchResult& part = partial_results[ri];
    scanned += part.scanned_candidates;
    size_t route_take = part.topk.size();
    if (exact_rerank_enable) {
      const uint32_t route_cap = ri < rerank_candidates_per_route.size()
                                     ? rerank_candidates_per_route[ri]
                                     : static_cast<uint32_t>(route_take);
      route_take = std::min(route_take, static_cast<size_t>(route_cap));
    }
    total_candidates += route_take;
  }

  std::vector<Candidate> merged;
  merged.reserve(total_candidates);
  for (size_t ri = 0; ri < partial_results.size(); ++ri) {
    const SearchResult& part = partial_results[ri];
    size_t route_take = part.topk.size();
    if (exact_rerank_enable) {
      const uint32_t route_cap = ri < rerank_candidates_per_route.size()
                                     ? rerank_candidates_per_route[ri]
                                     : static_cast<uint32_t>(route_take);
      route_take = std::min(route_take, static_cast<size_t>(route_cap));
    }
    for (size_t i = 0; i < route_take; ++i) {
      const Candidate& cand = part.topk[i];
      if (cand.doc_id >= seen_rows) {
        continue;
      }
      merged.push_back(cand);
    }
  }

  if (partial_results.size() > 1) {
    auto doc_approx_less = [](const Candidate& a, const Candidate& b) {
      if (a.doc_id != b.doc_id) {
        return a.doc_id < b.doc_id;
      }
      if (a.approx_dist != b.approx_dist) {
        return a.approx_dist < b.approx_dist;
      }
      return a.from_new < b.from_new;
    };
    std::sort(merged.begin(), merged.end(), doc_approx_less);
    size_t write = 0;
    for (size_t read = 0; read < merged.size();) {
      if (write != read) {
        merged[write] = std::move(merged[read]);
      }
      const DocId doc_id = merged[write].doc_id;
      ++write;
      do {
        ++read;
      } while (read < merged.size() && merged[read].doc_id == doc_id);
    }
    merged.resize(write);
  }

  if (exact_rerank_enable) {
    const float query_norm = query_whitened.squaredNorm();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && merged.size() > 1)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(merged.size()); ++i) {
      Candidate& cand = merged[static_cast<size_t>(i)];
      if (cand.doc_id >= static_cast<DocId>(base_whitened.rows()) ||
          cand.doc_id >= static_cast<DocId>(base_norms.size())) {
        cand.rerank_dist = std::numeric_limits<float>::infinity();
        continue;
      }
      const Eigen::Index doc_idx = static_cast<Eigen::Index>(cand.doc_id);
      const float dot = base_whitened.row(doc_idx).dot(query_whitened);
      cand.rerank_dist =
          SquaredL2FromNormDot(query_norm, base_norms(doc_idx), dot);
    }
    if (merged.size() > topk) {
      std::nth_element(merged.begin(),
                       merged.begin() + static_cast<std::ptrdiff_t>(topk),
                       merged.end(),
                       RerankCandidateLess);
      merged.resize(topk);
    }
    std::sort(merged.begin(), merged.end(), RerankCandidateLess);
  } else {
    if (merged.size() > topk) {
      std::nth_element(merged.begin(),
                       merged.begin() + static_cast<std::ptrdiff_t>(topk),
                       merged.end(),
                       ApproxCandidateLess);
      merged.resize(topk);
    }
    std::sort(merged.begin(), merged.end(), ApproxCandidateLess);
  }

  out.topk = std::move(merged);
  out.scanned_candidates = scanned;
  return out;
}

Result<EvalMetricsLite> EvaluateStatePrefix(
    const Config& config,
    const MatrixRM& base_whitened,
    const Eigen::VectorXf& base_norms,
    uint32_t seen_rows,
    uint32_t new_begin,
    uint32_t new_end,
    const MatrixRM& queries_whitened,
    const std::shared_ptr<IVFIndex>& main_ivf,
    const VersionSet& main_versions,
    uint32_t main_rows,
    const std::optional<DeltaShard>& frozen_delta,
    const std::optional<DeltaShard>& active_delta,
    const SearchParams& params) {
  if (seen_rows == 0) {
    return Status::InvalidArgument("EvaluateStatePrefix: seen_rows=0");
  }
  if (seen_rows > static_cast<uint32_t>(base_whitened.rows()) ||
      seen_rows > static_cast<uint32_t>(base_norms.size())) {
    return Status::InvalidArgument("EvaluateStatePrefix: invalid base rows");
  }

  if (!main_ivf) {
    return Status::InvalidArgument("EvaluateStatePrefix: null main");
  }

  auto gt_res =
      ExactSearchPrefixBlockwise(queries_whitened,
                                 base_whitened,
                                 seen_rows,
                                 config.topk);
  if (!gt_res.ok()) return gt_res.status();

  std::vector<std::vector<DocId>> gt = std::move(gt_res.value());

  std::vector<SearchRoute> routes;
  routes.push_back(SearchRoute{main_ivf, main_versions, 0, "main", main_rows});

  if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
    routes.push_back(SearchRoute{
        frozen_delta->ivf,
        frozen_delta->versions,
        1,
        "frozen_delta",
        frozen_delta->rows});
  }

  if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
    routes.push_back(SearchRoute{
        active_delta->ivf,
        active_delta->versions,
        1,
        "active_delta",
        active_delta->rows});
  }

  const uint32_t nq = static_cast<uint32_t>(queries_whitened.rows());

  const uint32_t route_count = static_cast<uint32_t>(routes.size());
  const bool enable_exact_rerank = config.exact_rerank_enable;
  
  std::vector<uint32_t> route_nprobes(
      route_count,
      std::max<uint32_t>(1u, params.nprobe));
  
  std::vector<uint32_t> route_topks(
      route_count,
      std::max<uint32_t>(1u, params.topk));
  
  std::vector<uint32_t> route_rerank_candidates(
      route_count,
      std::max<uint32_t>(1u, params.topk));
  
  for (uint32_t ri = 0; ri < route_count; ++ri) {
    if (enable_exact_rerank) {
      route_rerank_candidates[static_cast<size_t>(ri)] =
          ResolveRouteExactRerankCandidates(
              config,
              routes[static_cast<size_t>(ri)]);
  
      route_topks[static_cast<size_t>(ri)] =
          std::max<uint32_t>(
              params.topk,
              route_rerank_candidates[static_cast<size_t>(ri)]);
    }
  }

  std::vector<std::vector<SearchResult>> route_results(
      static_cast<size_t>(nq), std::vector<SearchResult>(static_cast<size_t>(route_count)));
  std::vector<double> query_ms(static_cast<size_t>(nq), 0.0);
  std::vector<double> pq_lut_build_us(static_cast<size_t>(nq), 0.0);
  std::vector<double> pq_adc_scan_us(static_cast<size_t>(nq), 0.0);
  std::vector<double> scanned_counts(static_cast<size_t>(nq), 0.0);
  std::vector<uint64_t> rerank_main_counts(
      config.enable_rerank_source_diag ? static_cast<size_t>(nq) : 0, 0);
  std::vector<uint64_t> rerank_delta_counts(
      config.enable_rerank_source_diag ? static_cast<size_t>(nq) : 0, 0);
  double slowest_route_wall_ms = 0.0;

  for (uint32_t ri = 0; ri < route_count; ++ri) {
    const auto& route = routes[static_cast<size_t>(ri)];
    Timer route_timer;
  
    auto res = route.ivf->SearchBatch(
        queries_whitened,
        route_topks[static_cast<size_t>(ri)],
        route_nprobes[static_cast<size_t>(ri)],
        route.versions,
        route.from_new,
        config.enable_miss_diag);
    const double route_elapsed = route_timer.ElapsedMillis();
    slowest_route_wall_ms = std::max(slowest_route_wall_ms, route_elapsed);
  
    if (!res.ok()) return res.status();
  
    std::vector<SearchResult> batch = std::move(res.value());
  
    if (batch.size() != nq) {
      return Status::Internal("SearchBatch size mismatch");
    }
  
    for (uint32_t qi = 0; qi < nq; ++qi) {
      pq_lut_build_us[static_cast<size_t>(qi)] +=
          batch[static_cast<size_t>(qi)].pq_lut_build_us;
      pq_adc_scan_us[static_cast<size_t>(qi)] +=
          batch[static_cast<size_t>(qi)].pq_adc_scan_us;
      route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)] =
          std::move(batch[static_cast<size_t>(qi)]);
    }
  }

  const double per_query_search_ms =
      nq > 0 ? slowest_route_wall_ms / static_cast<double>(nq) : 0.0;

  std::vector<std::vector<DocId>> pred(static_cast<size_t>(nq));

  Timer merge_wall_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t qi64 = 0; qi64 < static_cast<int64_t>(nq); ++qi64) {
    const uint32_t qi = static_cast<uint32_t>(qi64);
    const Eigen::VectorXf query =
        queries_whitened.row(static_cast<Eigen::Index>(qi)).transpose();
    Timer query_timer;
    SearchResult merged = MergeTopKPrefix(route_results[static_cast<size_t>(qi)],
                                          params.topk,
                                          enable_exact_rerank,
                                          route_rerank_candidates,
                                          query,
                                          base_whitened,
                                          base_norms,
                                          seen_rows);
    const double query_merge_ms = query_timer.ElapsedMillis();
    std::vector<DocId> row;
    row.reserve(merged.topk.size());
    for (const auto& cand : merged.topk) {
      row.push_back(cand.doc_id);
    }
    pred[static_cast<size_t>(qi)] = std::move(row);
    scanned_counts[static_cast<size_t>(qi)] =
        static_cast<double>(merged.scanned_candidates);
    query_ms[static_cast<size_t>(qi)] =
        per_query_search_ms + query_merge_ms;
    if (config.enable_rerank_source_diag) {
      for (const auto& candidate : merged.topk) {
        if (candidate.from_new == 1) {
          rerank_delta_counts[static_cast<size_t>(qi)]++;
        } else {
          rerank_main_counts[static_cast<size_t>(qi)]++;
        }
      }
    }
  }
  const double merge_wall_ms = merge_wall_timer.ElapsedMillis();
  // Qw is transformed once when the whitening version is built/rebuilt, so
  // per-snapshot query evaluation has no additional whitening wall time here.
  const double whitening_wall_ms = 0.0;
  const double query_wall_ms =
      whitening_wall_ms + slowest_route_wall_ms + merge_wall_ms;
  const double scanned_total =
      std::accumulate(scanned_counts.begin(), scanned_counts.end(), 0.0);

  auto recall_res = RecallAtK(gt, pred, config.topk);
  if (!recall_res.ok()) return recall_res.status();
  const RecallAgeMetrics age_metrics =
      ComputeRecallByRecentInsert(gt, pred, new_begin, new_end, seen_rows);
  const DistributionStatsLite query_stats = SummarizeValues(query_ms);
  const DistributionStatsLite scanned_stats = SummarizeValues(scanned_counts);
  const double avg_search_ms = nq > 0
      ? std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / static_cast<double>(nq)
      : 0.0;
  const double avg_pq_lut_build_us = nq > 0
      ? std::accumulate(pq_lut_build_us.begin(), pq_lut_build_us.end(), 0.0) /
            static_cast<double>(nq)
      : 0.0;
  const double avg_pq_adc_scan_us = nq > 0
      ? std::accumulate(pq_adc_scan_us.begin(), pq_adc_scan_us.end(), 0.0) /
            static_cast<double>(nq)
      : 0.0;
  const double avg_rerank_main = nq > 0
      ? static_cast<double>(std::accumulate(rerank_main_counts.begin(), rerank_main_counts.end(),
                                           uint64_t{0})) / static_cast<double>(nq)
      : 0.0;
  const double avg_rerank_delta = nq > 0
      ? static_cast<double>(std::accumulate(rerank_delta_counts.begin(), rerank_delta_counts.end(),
                                           uint64_t{0})) / static_cast<double>(nq)
      : 0.0;
  const uint64_t rerank_main_total = std::accumulate(
      rerank_main_counts.begin(), rerank_main_counts.end(), uint64_t{0});
  const uint64_t rerank_delta_total = std::accumulate(
      rerank_delta_counts.begin(), rerank_delta_counts.end(), uint64_t{0});
  uint64_t gt_total = 0;
  uint64_t gt_probed = 0;
  uint64_t gt_hit_and_probed = 0;
  uint64_t miss_not_probed = 0;
  uint64_t miss_pq_filtered = 0;
  uint64_t pq_rank_loss_count = 0;
  double exact_recall_sum = 0.0;
  double pq_rank_loss_sum = 0.0;
  std::vector<std::pair<double, std::pair<uint32_t, uint32_t>>> worst_order;
  std::vector<std::string> worst_queries;
  if (config.enable_miss_diag) {
    worst_order.reserve(nq);
    for (uint32_t qi = 0; qi < nq; ++qi) {
      const auto& gt_row = gt[static_cast<size_t>(qi)];
      if (gt_row.empty()) continue;
      const auto& pred_row = pred[static_cast<size_t>(qi)];
      std::unordered_set<DocId> pred_set(pred_row.begin(), pred_row.end());
      size_t scanned_reserve = 0;
      for (uint32_t ri = 0; ri < route_count; ++ri) {
        scanned_reserve += route_results[static_cast<size_t>(qi)]
                                       [static_cast<size_t>(ri)]
                                           .scanned_doc_ids.size();
      }
      std::unordered_map<DocId, float> approx_by_doc;
      approx_by_doc.reserve(scanned_reserve * 2 + 1);
      for (uint32_t ri = 0; ri < route_count; ++ri) {
        const auto& result = route_results[static_cast<size_t>(qi)][static_cast<size_t>(ri)];
        const size_t count = std::min(result.scanned_doc_ids.size(),
                                      result.scanned_approx_dists.size());
        for (size_t i = 0; i < count; ++i) {
          const DocId doc = result.scanned_doc_ids[i];
          const float distance = result.scanned_approx_dists[i];
          const auto it = approx_by_doc.find(doc);
          if (it == approx_by_doc.end() || distance < it->second) {
            approx_by_doc[doc] = distance;
          }
        }
      }

      std::vector<std::pair<float, DocId>> approx_ranked;
      std::vector<std::pair<float, DocId>> exact_ranked;
      approx_ranked.reserve(approx_by_doc.size());
      exact_ranked.reserve(approx_by_doc.size());
      const Eigen::VectorXf query =
          queries_whitened.row(static_cast<Eigen::Index>(qi)).transpose();
      const float query_norm = query.squaredNorm();
      for (const auto& entry : approx_by_doc) {
        approx_ranked.emplace_back(entry.second, entry.first);
        const Eigen::Index doc = static_cast<Eigen::Index>(entry.first);
        const float dot = base_whitened.row(doc).dot(query);
        exact_ranked.emplace_back(
            SquaredL2FromNormDot(query_norm, base_norms(doc), dot), entry.first);
      }
      const auto by_distance = [](const auto& left, const auto& right) {
        return left.first < right.first;
      };
      std::sort(approx_ranked.begin(), approx_ranked.end(), by_distance);
      std::sort(exact_ranked.begin(), exact_ranked.end(), by_distance);
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
      std::unordered_set<DocId> exact_topk;
      const size_t exact_count = std::min<size_t>(config.topk, exact_ranked.size());
      exact_topk.reserve(exact_count * 2 + 1);
      for (size_t i = 0; i < exact_count; ++i) exact_topk.insert(exact_ranked[i].second);
      uint32_t exact_hits = 0;
      uint32_t pred_hits = 0;
      uint32_t query_misses = 0;
      for (DocId doc : gt_row) {
        if (exact_topk.find(doc) != exact_topk.end()) ++exact_hits;
        const bool hit = pred_set.find(doc) != pred_set.end();
        if (hit) ++pred_hits;
        gt_total++;
        const auto found = approx_rank.find(doc);
        const bool probed = found != approx_rank.end();
        if (probed) {
          ++gt_probed;
          if (hit) ++gt_hit_and_probed;
        }
        if (!hit) {
          ++query_misses;
          if (!probed) {
            ++miss_not_probed;
          } else {
            ++miss_pq_filtered;
            const auto exact_found = exact_rank.find(doc);
            if (exact_found != exact_rank.end()) {
              const uint32_t loss = found->second > exact_found->second
                  ? found->second - exact_found->second
                  : 0u;
              pq_rank_loss_sum += static_cast<double>(loss);
              ++pq_rank_loss_count;
            }
          }
        }
      }
      exact_recall_sum += static_cast<double>(exact_hits) /
                          static_cast<double>(gt_row.size());
      const double query_recall = static_cast<double>(pred_hits) /
                                  static_cast<double>(gt_row.size());
      if (query_misses > 0) {
        worst_order.push_back({query_recall, {query_misses, qi}});
      }
    }
    std::sort(worst_order.begin(), worst_order.end(), [](const auto& left, const auto& right) {
      if (left.first != right.first) return left.first < right.first;
      return left.second.first > right.second.first;
    });
    const size_t worst_count = std::min<size_t>(10, worst_order.size());
    worst_queries.reserve(worst_count);
    for (size_t i = 0; i < worst_count; ++i) {
      const auto& item = worst_order[i];
      const uint32_t query_id = item.second.second;
      const uint32_t misses = item.second.first;
      const auto& gt_row = gt[static_cast<size_t>(query_id)];
      worst_queries.push_back("qid=" + std::to_string(query_id) +
                              ", recall=" + std::to_string(item.first) +
                              ", misses=" + std::to_string(misses) +
                              ", gt=" + std::to_string(gt_row.size()));
    }
  }

  EvalMetricsLite m;
  m.recall_available = true;
  m.recall = recall_res.value();
  m.recall_new = age_metrics.recall_new;
  m.recall_old = age_metrics.recall_old;
  m.gt_new_ratio = age_metrics.gt_new_ratio;
  m.gt_new_total = age_metrics.gt_new_total;
  m.gt_old_total = age_metrics.gt_old_total;
  m.hit_new_total = age_metrics.hit_new_total;
  m.hit_old_total = age_metrics.hit_old_total;
  m.query_count = nq;
  m.avg_query_ms =
      nq > 0 ? slowest_route_wall_ms / static_cast<double>(nq) : 0.0;
  m.avg_search_ms = avg_search_ms;
  m.end_to_end_overhead_ms = avg_search_ms;
  m.search_p50 = query_stats.p50;
  m.search_p99 = query_stats.p99;
  m.total_p50 = query_stats.p50;
  m.total_p99 = query_stats.p99;
  m.p50_query_ms = m.search_p50;
  m.p99_query_ms = m.search_p99;
  m.avg_pq_lut_build_us = avg_pq_lut_build_us;
  m.avg_pq_adc_scan_us = avg_pq_adc_scan_us;
  m.qps =
      query_wall_ms > 0.0 ? (1000.0 * static_cast<double>(nq) / query_wall_ms) : 0.0;
  m.avg_scanned =
      nq > 0 ? static_cast<double>(scanned_total) / static_cast<double>(nq)
             : 0.0;
  m.scanned_p50 = scanned_stats.p50;
  m.scanned_p99 = scanned_stats.p99;
  m.scanned_max = scanned_stats.max;
  m.scan_ratio =
      seen_rows > 0 ? m.avg_scanned / static_cast<double>(seen_rows) : 0.0;
  m.scanned_per_topk =
      config.topk > 0 ? m.avg_scanned / static_cast<double>(config.topk) : 0.0;
  m.query_eval_ms = query_wall_ms;
  m.main_route_queries = nq;
  for (const auto& route : routes) {
    if (route.name == "active_delta") m.active_delta_route_queries = nq;
    if (route.name == "frozen_delta") m.frozen_delta_route_queries = nq;
  }
  if (config.enable_rerank_source_diag) {
    m.rerank_topk_main_total = rerank_main_total;
    m.rerank_topk_delta_total = rerank_delta_total;
    const uint64_t rerank_total = rerank_main_total + rerank_delta_total;
    if (rerank_total > 0) {
      m.rerank_topk_main_ratio = static_cast<double>(rerank_main_total) /
                                 static_cast<double>(rerank_total);
      m.rerank_topk_delta_ratio = static_cast<double>(rerank_delta_total) /
                                  static_cast<double>(rerank_total);
    }
    m.rerank_topk_main_avg = avg_rerank_main;
    m.rerank_topk_delta_avg = avg_rerank_delta;
  }
  if (config.enable_miss_diag) {
    m.gt_probed_rate = gt_total > 0
        ? static_cast<double>(gt_probed) / static_cast<double>(gt_total)
        : 0.0;
    m.recall_on_probed_gt = gt_probed > 0
        ? static_cast<double>(gt_hit_and_probed) / static_cast<double>(gt_probed)
        : 0.0;
    m.exact_recall_on_probed_candidates = nq > 0
        ? exact_recall_sum / static_cast<double>(nq)
        : 0.0;
    m.avg_pq_rank_loss = pq_rank_loss_count > 0
        ? pq_rank_loss_sum / static_cast<double>(pq_rank_loss_count)
        : 0.0;
    m.miss_not_probed = static_cast<uint32_t>(std::min<uint64_t>(
        miss_not_probed, std::numeric_limits<uint32_t>::max()));
    m.miss_probed_filtered_by_pq = static_cast<uint32_t>(std::min<uint64_t>(
        miss_pq_filtered, std::numeric_limits<uint32_t>::max()));
    m.pq_rank_loss_count = static_cast<uint32_t>(std::min<uint64_t>(
        pq_rank_loss_count, std::numeric_limits<uint32_t>::max()));
    m.worst_queries = std::move(worst_queries);
  }

  return m;
}


}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage:\n"
              << "  " << argv[0]
              << " <config.json> <dataset_dir> [query_or_dir] [workload.json] [summary.json]\n"
              << "\nIf workload.json is omitted, a file ending with _workload.json is resolved under dataset_dir.\n"
              << "If summary.json is omitted, output is written to the same result path as run_eval.\n";
    return 1;
  }

  auto looks_json = [](const std::string& p) -> bool {
    std::filesystem::path path(p);
    return path.extension() == ".json" ||
           p.find("workload") != std::string::npos;
  };

  const std::string config_path = argv[1];
  const std::string dataset_arg = argv[2];

  std::string query_arg;
  std::string workload_path;
  std::string summary_path;

  if (argc == 4) {
    if (looks_json(argv[3])) {
      workload_path = argv[3];
    } else {
      query_arg = argv[3];
    }
  } else if (argc == 5) {
    if (looks_json(argv[3])) {
      workload_path = argv[3];
      summary_path = argv[4];
    } else {
      query_arg = argv[3];
      workload_path = argv[4];
    }
  } else if (argc >= 6) {
    query_arg = argv[3];
    workload_path = argv[4];
    summary_path = argv[5];
  }

  auto cfg_res = LoadConfigFromJson(config_path);
  if (!cfg_res.ok()) {
    std::cerr << cfg_res.status().ToString() << std::endl;
    return 1;
  }

  Config config = cfg_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  if (workload_path.empty()) {
    auto wr = ResolveFileWithSuffix(dataset_arg, "_workload.json");
    if (!wr.ok()) {
      std::cerr << wr.status().ToString() << std::endl;
      return 1;
    }
    workload_path = wr.value();
  }

  auto wl_res = LoadWorkloadJson(workload_path);
  if (!wl_res.ok()) {
    std::cerr << wl_res.status().ToString() << std::endl;
    return 1;
  }

  Workload workload = wl_res.value();

  std::cout << "[WORKLOAD] " << workload.path
            << ", ops=" << workload.operations.size()
            << ", declared_vectors=" << workload.num_vectors
            << std::endl;

  auto base_resolved = ResolveFvecsPath(dataset_arg, "_base.fvecs");
  if (!base_resolved.ok()) {
    std::cerr << base_resolved.status().ToString() << std::endl;
    return 1;
  }

  const std::string base_path = base_resolved.value();
  if (summary_path.empty()) {
    summary_path = ann::eval::BuildDefaultResultPath(config_path, base_path).string();
  }
  std::error_code summary_dir_ec;
  std::filesystem::create_directories(
      std::filesystem::path(summary_path).parent_path(), summary_dir_ec);
  std::filesystem::path memory_trace_path =
      std::filesystem::path(summary_path).parent_path() / "memory_trace.json";
  if (memory_trace_path.empty()) {
    memory_trace_path = "memory_trace.json";
  }

  auto source_res = ann::eval_memory::MakeFvecsBaseSource(base_path);
  if (!source_res.ok()) {
    std::cerr << source_res.status().ToString() << std::endl;
    return 1;
  }

  ann::eval_memory::BaseVectorSource base_source = std::move(source_res.value());
  const uint32_t nx = base_source.rows;

  if (nx == 0) {
    std::cerr << "Empty base" << std::endl;
    return 1;
  }

  std::cout << "[INFO] Using base dataset: " << base_path << std::endl;

  if (workload.num_vectors > 0 && workload.num_vectors != nx) {
    std::cout << "[WARN] workload.num_vectors=" << workload.num_vectors
              << " but base rows=" << nx << std::endl;
  }

  if (config.dim != base_source.dim) {
    std::cout << "[INFO] Override dim " << config.dim
              << " -> " << base_source.dim << std::endl;
    config.dim = base_source.dim;
  }

  std::optional<std::string> query_path;

  if (!query_arg.empty()) {
    auto qr = ResolveFvecsPath(query_arg, "_query.fvecs");
    if (!qr.ok()) {
      std::cerr << qr.status().ToString() << std::endl;
      return 1;
    }
    query_path = qr.value();
  } else {
    namespace fs = std::filesystem;
    fs::path parent = fs::path(base_path).parent_path();
    if (!parent.empty()) {
      auto qr = ResolveFvecsPath(parent.string(), "_query.fvecs");
      if (qr.ok()) query_path = qr.value();
    }
  }

  MatrixRM Q;

  if (query_path) {
    auto q_res = LoadFvecs(*query_path);
    if (!q_res.ok()) {
      std::cerr << q_res.status().ToString() << std::endl;
      return 1;
    }

    Q = q_res.value();

    if (static_cast<uint32_t>(Q.cols()) != config.dim) {
      std::cerr << "Query dim mismatch" << std::endl;
      return 1;
    }

    std::cout << "[INFO] Using query dataset: " << *query_path << std::endl;
  } else {
    uint32_t nq = config.max_queries > 0 ? config.max_queries : 8;
    Q = GenerateRandom(nq, config.dim, config.seed + 1);

    std::cout << "[WARN] Query dataset not found; generated random queries nq="
              << nq << std::endl;
  }

  if (config.max_queries > 0 &&
      static_cast<uint32_t>(Q.rows()) > config.max_queries) {
    Q = Q.topRows(config.max_queries);
  }

  const uint32_t main_bootstrap_rows = ResolveMainRows(config, nx);
  const uint32_t rows_after_main =
      nx > main_bootstrap_rows ? nx - main_bootstrap_rows : 0;
  const uint32_t delta_train_rows =
      ResolveDeltaTrainRows(config, rows_after_main);
  const uint32_t delta_ivf_nlist =
      config.delta_ivf_nlist > 0 ? config.delta_ivf_nlist : config.ivf_nlist;
  const uint32_t merge_trigger_rows =
      config.merge_trigger_rows > 0 ? config.merge_trigger_rows
                                    : delta_train_rows;
  const uint32_t internal_chunk_rows =
      std::max<uint32_t>(1, config.stream_batch_size);

  std::cout << "[INFO] main_bootstrap_rows=" << main_bootstrap_rows
            << ", delta_train_rows=" << delta_train_rows
            << ", delta_ivf_nlist=" << delta_ivf_nlist
            << ", merge_trigger_mode=" << config.merge_trigger_mode
            << ", merge_trigger_rows=" << merge_trigger_rows
            << ", merge_trigger_qe_ratio=" << config.merge_trigger_qe_ratio
            << ", merge_trigger_drift=" << config.merge_trigger_drift
            << ", merge_trigger_delta_main_ratio="
            << config.merge_trigger_delta_main_ratio
            << ", merge_trigger_imbalance_ratio="
            << config.merge_trigger_imbalance_ratio
            << ", enable_global_rebuild=" << std::boolalpha
            << config.enable_global_rebuild
            << ", global_rebuild_max_count="
            << config.global_rebuild_max_count
            << ", global_rebuild_main_imbalance_ratio="
            << config.global_rebuild_main_imbalance_ratio
            << ", global_rebuild_force_main_rows="
            << config.global_rebuild_force_main_rows
            << ", global_rebuild_cooldown_rows="
            << config.global_rebuild_cooldown_rows
            << ", merge_assignment_mode=" << config.merge_assignment_mode
            << ", merge_assignment_top_r=" << config.merge_assignment_top_r
            << ", merge_assignment_gamma=" << config.merge_assignment_gamma
            << ", merge_assignment_hard_cap_ratio="
            << config.merge_assignment_hard_cap_ratio
            << ", merge_assignment_lambda=" << config.merge_assignment_lambda
            << ", merge_score_alpha=" << config.merge_score_alpha
            << ", merge_score_beta=" << config.merge_score_beta
            << ", merge_score_threshold=" << config.merge_score_threshold
            << ", main_exact_rerank_candidates="
            << config.main_exact_rerank_candidates
            << ", active_exact_rerank_candidates="
            << config.active_exact_rerank_candidates
            << ", frozen_exact_rerank_candidates="
            << config.frozen_exact_rerank_candidates
            << ", stream_rows="
            << (nx > main_bootstrap_rows ? nx - main_bootstrap_rows : 0)
            << ", internal_chunk_rows=" << internal_chunk_rows
            << ", streaming_mode=ms_workload_strict"
            << std::endl;

  std::cout << "[MS-EVAL] STRICT workload mode: seen_rows starts at 0; "
            << "workload.json controls every insert/search checkpoint; "
            << "stream_batch_size is internal chunk size only."
            << std::endl;

  IVFParams ivf_params;
  ivf_params.nlist = std::max<uint32_t>(1, config.ivf_nlist);
  ivf_params.dim = config.dim;
  ivf_params.pq.enable = config.pq_enable;
  ivf_params.pq.M = config.pq_m;
  ivf_params.pq.nbits = config.pq_nbits;
  ivf_params.pq.residual = config.pq_residual;

  SearchParams search_params;
  search_params.topk = config.topk;
  search_params.nprobe = config.nprobe;
  search_params.use_whitening = false;

  OnlinePQUpdateOptions online_opts;
  online_opts.enable =
      config.online_pq_enable && config.pq_enable && config.pq_residual;
  online_opts.qe_ratio_threshold = config.online_pq_qe_ratio_threshold;
  online_opts.ema_alpha = config.online_pq_ema_alpha;
  online_opts.nqe_eps = config.online_pq_eps;
  online_opts.warmup_enable = config.online_pq_warmup_enable;
  online_opts.warmup_batches = config.online_pq_warmup_batches;
  online_opts.force_update_interval = config.online_pq_force_update_interval;
  online_opts.partial_top_alpha = config.online_pq_partial_top_alpha;
  online_opts.partial_alpha = config.online_pq_alpha;
  online_opts.partial_top_lambda = config.online_pq_partial_top_lambda;
  online_opts.partial_lambda = config.online_pq_lambda;
  online_opts.reencode_batch_after_update = config.online_pq_reencode_batch;

  auto whitening = CreateWhiteningModel();
  auto main_ivf = CreateIVFIndex();

  VersionId whiten_version = 0;
  VersionSet main_versions{};

  MatrixRM Xw;
  Eigen::VectorXf Xw_norms;
  MatrixRM Qw;

  bool main_built = false;
  uint32_t main_rows_current = 0;
  uint32_t next_delta_shard_id = 1;
  uint32_t seen_rows = 0;
  uint32_t whitened_rows_materialized = 0;
  uint32_t last_snapshot_active_rows = 0;

  std::optional<DeltaShard> active_delta;
  std::optional<DeltaShard> frozen_delta;

  OnlinePQUpdateStats last_pq_stats;
  uint32_t whitening_version_count_estimate = 0;

  ann::eval_memory::MemoryTraceRecorder memory_trace;
  auto collect_memory_components = [&]() {
    std::vector<ann::eval_memory::MemoryComponent> components;
    components.push_back({"base.raw_resident", base_source.ResidentBytes()});
    components.push_back({"base.whitened_matrix", ann::eval_memory::MatrixBytes(Xw)});
    components.push_back({"base.whitened_norms", ann::eval_memory::VectorBytes(Xw_norms)});
    components.push_back({"query.raw", ann::eval_memory::MatrixBytes(Q)});
    components.push_back({"query.whitened", ann::eval_memory::MatrixBytes(Qw)});
    const uint64_t whitening_version_bytes =
        (2ull * static_cast<uint64_t>(config.dim) * static_cast<uint64_t>(config.dim) +
         static_cast<uint64_t>(config.dim)) *
        sizeof(float);
    components.push_back({"whitening.model_versions_estimate",
                          static_cast<uint64_t>(whitening_version_count_estimate) *
                              whitening_version_bytes});
    ann::eval_memory::AddIVFMemoryComponents("main", main_ivf, main_versions, &components);
    if (active_delta.has_value()) {
      ann::eval_memory::AddIVFMemoryComponents(
          "active_delta", active_delta->ivf, active_delta->versions, &components);
    }
    if (frozen_delta.has_value()) {
      ann::eval_memory::AddIVFMemoryComponents(
          "frozen_delta", frozen_delta->ivf, frozen_delta->versions, &components);
    }
    return components;
  };
  auto record_memory = [&](const std::string& stage,
                           uint32_t active_rows,
                           uint64_t transient_estimate_bytes = 0) {
    memory_trace.Record(stage,
                        active_rows,
                        collect_memory_components(),
                        transient_estimate_bytes);
  };
  record_memory("after_dataset_load", 0);

  MergeOptions merge_options;
  merge_options.alpha = config.merge_score_alpha;
  merge_options.beta = config.merge_score_beta;
  merge_options.recluster_threshold = config.merge_score_threshold;
  merge_options.assignment_mode = config.merge_assignment_mode;
  merge_options.assignment_top_r = config.merge_assignment_top_r;
  merge_options.assignment_gamma = config.merge_assignment_gamma;
  merge_options.assignment_hard_cap_ratio =
      config.merge_assignment_hard_cap_ratio;
  merge_options.assignment_lambda = config.merge_assignment_lambda;

  ann::eval::RunEvalMsCounters counters;
  const auto& counter_values = counters.Values();
  const auto& init_ms = counter_values.init_ms;
  const auto& total_update_ms = counter_values.total_update_ms;
  const auto& total_update_whitening_ms = counter_values.total_update_whitening_ms;
  const auto& total_update_insert_ms = counter_values.total_update_insert_ms;
  const auto& total_update_record_build_ms = counter_values.total_update_record_build_ms;
  const auto& total_update_insert_encode_ms = counter_values.total_update_insert_encode_ms;
  const auto& total_update_insert_commit_ms = counter_values.total_update_insert_commit_ms;
  const auto& total_update_onlinepq_maintenance_ms =
      counter_values.total_update_onlinepq_maintenance_ms;
  const auto& total_update_delete_ms = counter_values.total_update_delete_ms;
  const auto& total_update_codebook_update_ms = counter_values.total_update_codebook_update_ms;
  const auto& total_update_reencode_ms = counter_values.total_update_reencode_ms;
  const auto& total_delta_ingest_assignment_us =
      counter_values.total_delta_ingest_assignment_us;
  const auto& total_delta_ingest_assignment_records =
      counter_values.total_delta_ingest_assignment_records;
  const auto& total_merge_ms = counter_values.total_merge_ms;
  const auto& total_merge_compute_ms = counter_values.total_merge_compute_ms;
  const auto& total_global_rebuild_ms = counter_values.total_global_rebuild_ms;
  const auto& pending_update_ms = counter_values.pending_update_ms;
  const auto& pending_merge_compute_ms = counter_values.pending_merge_compute_ms;
  const auto& pending_global_rebuild_ms = counter_values.pending_global_rebuild_ms;
  const auto& merge_count = counter_values.merge_count;
  const auto& global_rebuild_count = counter_values.global_rebuild_count;
  const auto& search_id = counter_values.search_id;
  std::vector<uint32_t> merge_nodes;
  std::vector<uint32_t> global_rebuild_nodes;
  uint32_t last_global_rebuild_rows = 0;
  uint32_t last_global_rebuild_main_rows = 0;

  struct PartitionSummaryLocal {
    uint32_t nlist{0};
    uint32_t non_empty{0};
    uint32_t max_list{0};
    double avg_non_empty{0.0};
    double imbalance{0.0};
  };

  auto partition_summary =
      [&](const std::shared_ptr<IVFIndex>& ivf,
          const VersionSet& versions) -> PartitionSummaryLocal {
    PartitionSummaryLocal s;
    if (!ivf) return s;

    auto sizes = ivf->GetPartitionSizes(versions);
    if (!sizes.ok()) return s;

    s.nlist = static_cast<uint32_t>(sizes.value().size());

    uint64_t total = 0;
    for (uint32_t v : sizes.value()) {
      if (v > 0) {
        s.non_empty++;
        total += v;
      }
      s.max_list = std::max(s.max_list, v);
    }

    if (s.non_empty > 0) {
      s.avg_non_empty =
          static_cast<double>(total) / static_cast<double>(s.non_empty);
      s.imbalance =
          static_cast<double>(s.max_list) / std::max(1.0, s.avg_non_empty);
    }

    return s;
  };

  auto current_imbalance_ratio = [&]() {
    double ratio = partition_summary(main_ivf, main_versions).imbalance;
    if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
      ratio = std::max(
          ratio,
          partition_summary(frozen_delta->ivf, frozen_delta->versions).imbalance);
    }
    if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
      ratio = std::max(
          ratio,
          partition_summary(active_delta->ivf, active_delta->versions).imbalance);
    }
    return ratio;
  };

  auto ensure_whitened_materialized =
      [&](uint32_t required_rows, const std::string& stage) -> Status {
    if (required_rows <= whitened_rows_materialized) {
      return Status::OK();
    }
    if (required_rows > nx) {
      return Status::InvalidArgument(
          "ensure_whitened_materialized: required rows exceed base rows");
    }
    if (config.use_whitening && whiten_version == 0) {
      return Status::InvalidArgument(
          "ensure_whitened_materialized: whitening version missing");
    }

    Timer transform_timer;
    const uint32_t start_rows = whitened_rows_materialized;
    EnsureWhitenedCapacity(&Xw, required_rows, nx, config.dim);
    if (Xw_norms.size() < static_cast<Eigen::Index>(required_rows)) {
      Xw_norms.conservativeResize(static_cast<Eigen::Index>(required_rows));
    }
    uint32_t begin = start_rows;
    while (begin < required_rows) {
      const uint32_t count =
          std::min<uint32_t>(kAddBlockRows, required_rows - begin);
      auto block_res =
          ann::eval_memory::TransformSourceRangeToIndexSpace(base_source,
                                                             begin,
                                                             count,
                                                             config.use_whitening,
                                                             whiten_version,
                                                             whitening,
                                                             config.use_cosine);
      if (!block_res.ok()) {
        return block_res.status();
      }
      Xw.middleRows(static_cast<Eigen::Index>(begin),
                    static_cast<Eigen::Index>(count)) = block_res.value();
      Xw_norms.segment(static_cast<Eigen::Index>(begin),
                       static_cast<Eigen::Index>(count)) =
          Xw.middleRows(static_cast<Eigen::Index>(begin),
                        static_cast<Eigen::Index>(count))
              .rowwise()
              .squaredNorm();
      begin += count;
    }
    whitened_rows_materialized = required_rows;
    const double transform_ms = transform_timer.ElapsedMillis();
    counters.AddWhiteningUpdate(transform_ms);
    record_memory(stage,
                  seen_rows,
                  2ull * ann::eval_memory::MatrixActiveBytes(
                             required_rows - start_rows, config.dim));
    return Status::OK();
  };

  auto ensure_active_delta = [&](uint32_t train_begin) -> Status {
    if (!main_built) {
      return Status::InvalidArgument("ensure_active_delta: main is not built");
    }

    if (active_delta.has_value()) return Status::OK();
    if (train_begin >= nx) return Status::OK();

    const uint32_t requested_train =
        std::max<uint32_t>(1, delta_train_rows);
    const uint32_t train_rows =
        std::min<uint32_t>(requested_train, nx - train_begin);

    Status materialize_status =
        ensure_whitened_materialized(train_begin + train_rows,
                                     "materialize_delta_train_window");
    if (!materialize_status.ok()) {
      return materialize_status;
    }

    MatrixRM delta_train = Xw.middleRows(train_begin, train_rows);

    IVFParams dp = ivf_params;
    dp.nlist = std::max<uint32_t>(1, delta_ivf_nlist);
    dp.kmeans_iterations = kDeltaKMeansIterationsDefault;

    auto d = BuildDeltaShard(delta_train, dp, whiten_version,
                             next_delta_shard_id++);
    if (!d.ok()) return d.status();

    active_delta = d.value();

    std::cout << "[MERGE] activate_delta shard=" << active_delta->shard_id
              << ", warmup_rows=" << train_rows
              << ", train_begin=" << train_begin
              << std::endl;
    record_memory("activate_delta", seen_rows);

    return Status::OK();
  };

  auto commit_frozen = [&](const std::string& trigger_reason) -> Status {
    if (!frozen_delta.has_value() || frozen_delta->rows == 0) {
      return Status::OK();
    }

    Timer mt;

    MatrixVectorAccessor vector_accessor(Xw, Xw_norms);
    auto res = merge_frozen_delta_into_main(main_ivf,
                                            main_versions,
                                            frozen_delta->ivf,
                                            frozen_delta->versions,
                                            vector_accessor,
                                            merge_options);
    if (!res.ok()) return res.status();

    const double elapsed = mt.ElapsedMillis();

    counters.AddMerge(elapsed, res.value().merge_compute_ms);
    merge_nodes.push_back(seen_rows);

    main_rows_current += res.value().frozen_records;

    const auto main_summary = partition_summary(main_ivf, main_versions);

    std::cout << "[MERGE] commit done: frozen_rows="
              << res.value().frozen_records
              << ", patched_partitions=" << res.value().patch_partitions
              << ", append_parts=" << res.value().append_partitions
              << ", recluster_parts=" << res.value().recluster_partitions
              << ", moved_delta_ratio=" << res.value().moved_delta_ratio
              << ", avg_assignment_dist_ratio="
              << res.value().avg_assignment_dist_ratio
              << ", max_assignment_dist_ratio="
              << res.value().max_assignment_dist_ratio
              << ", imbalance_before=" << res.value().imbalance_before
              << ", imbalance_after=" << res.value().imbalance_after
              << ", main_imbalance_after_real=" << main_summary.imbalance
              << ", trigger_reason=" << trigger_reason
              << ", merge_compute_ms=" << res.value().merge_compute_ms
              << ", codebook_rebuild_ms=" << res.value().codebook_rebuild_ms
              << ", merge_ms=" << elapsed
              << std::endl;
    ann::eval::PrintMergeProfile(res.value().profiling);

    frozen_delta.reset();
    TrimAllocatorRetainedMemory("merge_commit_done");
    record_memory("merge_commit_done", seen_rows);

    return Status::OK();
  };

  auto maybe_freeze_active = [&]() -> Status {
    if (!main_built ||
        !active_delta.has_value() ||
        active_delta->rows == 0) {
      return Status::OK();
    }

    const auto active_summary =
        partition_summary(active_delta->ivf, active_delta->versions);

    const double delta_main_ratio =
        main_rows_current > 0
            ? static_cast<double>(active_delta->rows) /
                  static_cast<double>(main_rows_current)
            : 0.0;

    const bool rows_trigger =
        merge_trigger_rows > 0 &&
        active_delta->rows >= merge_trigger_rows;

    const bool ratio_trigger =
        config.merge_trigger_delta_main_ratio > 0.0 &&
        delta_main_ratio >= config.merge_trigger_delta_main_ratio;

    const bool imbalance_trigger =
        config.merge_trigger_imbalance_ratio > 0.0 &&
        active_summary.imbalance >= config.merge_trigger_imbalance_ratio;

    const bool qe_trigger =
        config.merge_trigger_qe_ratio > 0.0 &&
        last_pq_stats.qe_ratio >= config.merge_trigger_qe_ratio;

    const bool drift_trigger =
        config.merge_trigger_drift > 0.0 &&
        last_pq_stats.codebook_drift_l2 >= config.merge_trigger_drift;

    const std::string mode = config.merge_trigger_mode;

    bool should = false;

    if (mode == "rows") {
      should = rows_trigger;
    } else if (mode == "delta_main_ratio") {
      should = ratio_trigger;
    } else if (mode == "imbalance") {
      should = imbalance_trigger;
    } else if (mode == "qe_ratio") {
      should = qe_trigger;
    } else if (mode == "drift") {
      should = drift_trigger;
    } else if (mode == "state") {
      should = ratio_trigger ||
               imbalance_trigger ||
               qe_trigger ||
               drift_trigger;
    } else {
      should = rows_trigger ||
               ratio_trigger ||
               imbalance_trigger ||
               qe_trigger ||
               drift_trigger;
    }

    if (!should) return Status::OK();

    std::string reason = "unknown";
    if (rows_trigger) reason = "rows";
    if (ratio_trigger) reason = "delta_main_ratio";
    if (imbalance_trigger) reason = "structure_imbalance";
    if (qe_trigger) reason = "qe_ratio";
    if (drift_trigger) reason = "codebook_drift";

    if (frozen_delta.has_value()) {
      Status c = commit_frozen(reason);
      if (!c.ok()) return c;
    }

    std::cout << "[MERGE] freeze_delta rows=" << active_delta->rows
              << ", reason=" << reason
              << ", rows_trigger=" << (rows_trigger ? "true" : "false")
              << ", structure_trigger="
              << (imbalance_trigger ? "true" : "false")
              << ", qe_ratio=" << last_pq_stats.qe_ratio
              << ", drift=" << last_pq_stats.codebook_drift_l2
              << ", delta_main_ratio=" << delta_main_ratio
              << ", imbalance_ratio=" << active_summary.imbalance
              << ", start background-style merge window training from row="
              << seen_rows
              << std::endl;

    frozen_delta = active_delta;
    active_delta.reset();
    record_memory("freeze_delta", seen_rows);

    return commit_frozen(reason);
  };

  auto maybe_global_rebuild = [&]() -> Status {
    if (!main_built ||
        !config.enable_global_rebuild ||
        config.global_rebuild_max_count == 0) {
      return Status::OK();
    }

    if (global_rebuild_count >= config.global_rebuild_max_count) {
      return Status::OK();
    }

    const auto main_summary = partition_summary(main_ivf, main_versions);

    const uint32_t main_rows_since_last =
        main_rows_current >= last_global_rebuild_main_rows
            ? main_rows_current - last_global_rebuild_main_rows
            : 0;

    bool rows_trigger = false;

    if (config.global_rebuild_force_main_rows > 0) {
      rows_trigger =
          main_rows_since_last >= config.global_rebuild_force_main_rows;
    }

    const bool imbalance_trigger =
        config.global_rebuild_main_imbalance_ratio > 0.0 &&
        main_summary.imbalance >=
            config.global_rebuild_main_imbalance_ratio;

    bool should = rows_trigger || imbalance_trigger;

    std::string reason =
        should ? (imbalance_trigger ? "main_structure_imbalance"
                                    : "main_rows")
               : "none";

    if (should &&
        config.global_rebuild_cooldown_rows > 0 &&
        seen_rows > last_global_rebuild_rows &&
        seen_rows - last_global_rebuild_rows <
            config.global_rebuild_cooldown_rows) {
      should = false;
      reason = "cooldown";
    }

    std::cout << "[GLOBAL REBUILD] check: base_rows=" << seen_rows
              << ", main_rows_current=" << main_rows_current
              << ", main_rows_since_last=" << main_rows_since_last
              << ", imbalance=" << main_summary.imbalance
              << ", imbalance_threshold="
              << config.global_rebuild_main_imbalance_ratio
              << ", rows_trigger=" << (rows_trigger ? "true" : "false")
              << ", imbalance_trigger="
              << (imbalance_trigger ? "true" : "false")
              << ", should_trigger=" << (should ? "true" : "false")
              << ", reason=" << reason
              << std::endl;

    if (!should) return Status::OK();

    Timer total_timer;

    const uint32_t old_main_rows = main_rows_current;
    const uint32_t rebuild_seen = seen_rows;

    uint32_t rebuild_main_rows = rebuild_seen;
    if (delta_train_rows > 0 && rebuild_seen > delta_train_rows) {
      rebuild_main_rows = rebuild_seen - delta_train_rows;
    }
    rebuild_main_rows = std::max<uint32_t>(1, rebuild_main_rows);

    const uint32_t active_seed_begin = rebuild_main_rows;
    const uint32_t active_seed_rows =
        rebuild_seen > active_seed_begin
            ? rebuild_seen - active_seed_begin
            : 0;

    const VersionId old_whiten_version = whiten_version;

    VersionId new_whiten_version = old_whiten_version;
    double whitening_ms = 0.0;
    if (config.use_whitening) {
      Timer wt;
      auto new_wv =
          config.use_cosine
              ? ann::eval_memory::FitWhiteningFromSourcePrefix(
                    base_source, rebuild_seen, kAddBlockRows, whitening)
              : ann::eval_memory::FitWhiteningFromWhitenedPrefix(
                    Xw, rebuild_seen, old_whiten_version, kAddBlockRows, whitening);
      if (!new_wv.ok()) return new_wv.status();
      new_whiten_version = new_wv.value();
      whitening_ms = wt.ElapsedMillis();
      ++whitening_version_count_estimate;
    }
    record_memory("global_rebuild_after_fit",
                  rebuild_seen,
                  !config.use_whitening
                      ? 0
                      : config.use_cosine
                      ? ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim)
                      : 2ull * ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim));

    Timer transform_timer;
    MatrixRM Xw_new;
    Status retarget_status;
    if (!config.use_whitening || config.use_cosine) {
      retarget_status = ann::eval_memory::TransformSourcePrefixToIndexSpace(
          base_source,
          rebuild_seen,
          kAddBlockRows,
          config.use_whitening,
          new_whiten_version,
          whitening,
          config.use_cosine,
          &Xw_new);
    } else {
      retarget_status = ann::eval_memory::RetargetWhitenedPrefix(Xw,
                                                                 rebuild_seen,
                                                                 kAddBlockRows,
                                                                 old_whiten_version,
                                                                 new_whiten_version,
                                                                 whitening,
                                                                 config.use_cosine,
                                                                 &Xw_new);
    }
    if (!retarget_status.ok()) return retarget_status;

    auto new_q = ann::eval_memory::TransformBatchToIndexSpace(
        Q, config.use_whitening, new_whiten_version, whitening, config.use_cosine);
    if (!new_q.ok()) return new_q.status();

    MatrixRM Qw_new = new_q.value();


    const double whitening_transform_ms = transform_timer.ElapsedMillis();
    record_memory("global_rebuild_after_base_retarget",
                  rebuild_seen,
                  2ull * ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim));

    auto new_main = CreateIVFIndex();

    std::vector<DocId> ids(rebuild_main_rows);
    std::iota(ids.begin(), ids.end(), 0);

    Timer build_timer;
    MatrixRM rebuild_main_train = Xw_new.topRows(rebuild_main_rows);
    std::vector<int> routing_assignments;

    auto ver =
        new_main->Build(rebuild_main_train, ids, ivf_params, 0, &routing_assignments);
    if (!ver.ok()) return ver.status();

    VersionSet new_main_versions{new_wv.value(), ver.value()};

    const double main_build_ms = build_timer.ElapsedMillis();

    Timer add_timer;
    Status add = AddRangeToIndex(new_main,
                                 Xw_new,
                                 0,
                                 rebuild_main_rows,
                                 config.dim,
                                 new_main_versions,
                                 routing_assignments.size() == rebuild_main_rows
                                     ? &routing_assignments
                                     : nullptr);
    if (!add.ok()) return add;

    const double main_add_ms = add_timer.ElapsedMillis();

    std::optional<DeltaShard> new_delta;
    double delta_seed_ms = 0.0;

    if (active_seed_rows > 0) {
      Timer delta_timer;

      IVFParams dp = ivf_params;
      dp.nlist = std::max<uint32_t>(1, delta_ivf_nlist);
      dp.kmeans_iterations = kDeltaKMeansIterationsDefault;

      MatrixRM rebuild_delta_train =
          Xw_new.middleRows(active_seed_begin, active_seed_rows);

      auto d = BuildDeltaShard(rebuild_delta_train,
                               dp,
                               new_wv.value(),
                               next_delta_shard_id++);
      if (!d.ok()) return d.status();

      new_delta = d.value();

      add = AddRangeToIndex(new_delta->ivf,
                            Xw_new,
                            active_seed_begin,
                            rebuild_seen,
                            config.dim,
                            new_delta->versions);
      if (!add.ok()) return add;

      new_delta->rows = active_seed_rows;
      delta_seed_ms = delta_timer.ElapsedMillis();
    }

    whiten_version = new_whiten_version;
    Xw = std::move(Xw_new);
    Xw_norms = Xw.rowwise().squaredNorm();
    whitened_rows_materialized = rebuild_seen;
    Qw = std::move(Qw_new);
    main_ivf = std::move(new_main);
    main_versions = new_main_versions;
    main_rows_current = rebuild_main_rows;
    active_delta = std::move(new_delta);
    frozen_delta.reset();
    last_pq_stats = OnlinePQUpdateStats{};
    TrimAllocatorRetainedMemory("global_rebuild_done");
    record_memory("global_rebuild_done", rebuild_seen);

    const double total_ms = total_timer.ElapsedMillis();

    counters.AddGlobalRebuild(total_ms);
    global_rebuild_nodes.push_back(rebuild_seen);
    last_global_rebuild_rows = rebuild_seen;
    last_global_rebuild_main_rows = main_rows_current;

    std::cout << "[GLOBAL REBUILD] done: base_rows=" << rebuild_seen
              << ", reason=" << reason
              << ", count=" << global_rebuild_count
              << "/" << config.global_rebuild_max_count
              << ", main_rows_since_last=" << main_rows_since_last
              << ", main_rows_threshold="
              << config.global_rebuild_force_main_rows
              << ", imbalance=" << main_summary.imbalance
              << ", imbalance_threshold="
              << config.global_rebuild_main_imbalance_ratio
              << ", old_main_rows=" << old_main_rows
              << ", new_main_rows=" << rebuild_main_rows
              << ", active_seed_rows=" << active_seed_rows
              << ", whitening_ms=" << whitening_ms
              << ", whitening_transform_ms="
              << whitening_transform_ms
              << ", main_build_ms=" << main_build_ms
              << ", main_add_ms=" << main_add_ms
              << ", delta_seed_ms=" << delta_seed_ms
              << ", total_ms=" << total_ms
              << std::endl;

    return Status::OK();
  };

  auto bootstrap_main_if_ready = [&]() -> Status {
    if (main_built || seen_rows < main_bootstrap_rows) {
      return Status::OK();
    }

    Timer total_timer;

    std::cout << "[BOOTSTRAP] start main construction at active_rows="
              << seen_rows
              << ", main_rows=" << main_bootstrap_rows
              << std::endl;

    double whitening_ms = 0.0;
    if (config.use_whitening) {
      Timer wt;
      auto wfit = ann::eval_memory::FitWhiteningFromSourcePrefix(
          base_source, main_bootstrap_rows, kAddBlockRows, whitening);
      if (!wfit.ok()) return wfit.status();
      whiten_version = wfit.value();
      whitening_version_count_estimate = 1;
      whitening_ms = wt.ElapsedMillis();
    } else {
      whiten_version = 0;
      whitening_version_count_estimate = 0;
    }
    record_memory("bootstrap_after_whitening_fit",
                  seen_rows,
                  config.use_whitening
                      ? ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim)
                      : 0);

    Timer transform_timer;

    Status xb_status = ann::eval_memory::TransformSourcePrefixToIndexSpace(
        base_source,
        seen_rows,
        kAddBlockRows,
        config.use_whitening,
        whiten_version,
        whitening,
        config.use_cosine,
        &Xw);
    if (!xb_status.ok()) return xb_status;

    auto qb = ann::eval_memory::TransformBatchToIndexSpace(
        Q, config.use_whitening, whiten_version, whitening, config.use_cosine);
    if (!qb.ok()) return qb.status();

    Qw = qb.value();

    Xw_norms = Xw.rowwise().squaredNorm();
    whitened_rows_materialized = seen_rows;

    const double transform_ms = transform_timer.ElapsedMillis();
    record_memory("bootstrap_after_base_transform",
                  seen_rows,
                  (config.use_whitening ? 2ull : 1ull) *
                      ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim));

    std::vector<DocId> main_ids(main_bootstrap_rows);
    std::iota(main_ids.begin(), main_ids.end(), 0);

    Timer build_timer;

    auto main_version = main_ivf->Build(
        Xw.topRows(main_bootstrap_rows), main_ids, ivf_params, 0);
    if (!main_version.ok()) return main_version.status();

    main_versions = VersionSet{whiten_version, main_version.value()};

    const double main_build_ms = build_timer.ElapsedMillis();

    auto init_build_profile_res = main_ivf->GetBuildProfiling(main_versions);
    if (init_build_profile_res.ok()) {
      const IVFBuildProfiling& build_profile = init_build_profile_res.value();
      std::cout << "[INITIAL_PQ_BUILD_PROFILE] build_pq_routing_assignment_us="
                << build_profile.build_pq_routing_assignment_us
                << ", build_pq_routing_assignment_rows="
                << build_profile.build_pq_routing_assignment_rows
                << ", build_pq_subspace_materialize_us="
                << build_profile.build_pq_subspace_materialize_us
                << ", build_pq_subspace_materialized_rows="
                << build_profile.build_pq_subspace_materialized_rows
                << ", build_pq_subspace_materialized_bytes="
                << build_profile.build_pq_subspace_materialized_bytes
                << ", build_pq_max_live_subspaces="
                << build_profile.build_pq_max_live_subspaces
                << ", build_pq_kmeans_us=" << build_profile.build_pq_kmeans_us
                << ", build_pq_training_total_us="
                << build_profile.build_pq_training_total_us
                << ", build_pq_full_residual_bytes="
                << build_profile.build_pq_full_residual_bytes
                << ", build_pq_training_concurrency="
                << build_profile.build_pq_training_concurrency << std::endl;
    }

    Timer add_timer;

    Status st = AddRangeToIndex(main_ivf,
                                Xw,
                                0,
                                main_bootstrap_rows,
                                config.dim,
                                main_versions);
    if (!st.ok()) return st;

    const double main_add_ms = add_timer.ElapsedMillis();

    main_built = true;
    main_rows_current = main_bootstrap_rows;
    last_global_rebuild_rows = seen_rows;
    last_global_rebuild_main_rows = main_rows_current;
    last_snapshot_active_rows = main_bootstrap_rows;

    double delta_seed_ms = 0.0;

    const uint32_t overflow_rows =
        seen_rows > main_bootstrap_rows
            ? seen_rows - main_bootstrap_rows
            : 0;

    if (overflow_rows > 0) {
      Timer delta_timer;

      Status ens = ensure_active_delta(main_bootstrap_rows);
      if (!ens.ok()) return ens;

      if (active_delta.has_value()) {
        st = AddRangeToIndex(active_delta->ivf,
                             Xw,
                             main_bootstrap_rows,
                             seen_rows,
                             config.dim,
                             active_delta->versions);
        if (!st.ok()) return st;

        active_delta->rows += overflow_rows;
      }

      delta_seed_ms = delta_timer.ElapsedMillis();
    }

    counters.AddInitializationMs(total_timer.ElapsedMillis());
    record_memory("bootstrap_done", seen_rows);

    std::cout << "[BOOTSTRAP] done active_rows=" << seen_rows
              << ", main_rows=" << main_rows_current
              << ", overflow_delta_rows=" << overflow_rows
              << ", whitening_ms=" << whitening_ms
              << ", whitening_transform_ms=" << transform_ms
              << ", main_build_ms=" << main_build_ms
              << ", main_add_ms=" << main_add_ms
              << ", delta_seed_ms=" << delta_seed_ms
              << ", total_ms=" << total_timer.ElapsedMillis()
              << std::endl;

    return maybe_global_rebuild();
  };

  auto insert_range = [&](uint32_t begin, uint32_t end) -> Status {
    if (begin > end || end > nx) {
      return Status::InvalidArgument("insert_range invalid range");
    }

    if (!main_built) {
      seen_rows = end;

      Status bs = bootstrap_main_if_ready();
      if (!bs.ok()) return bs;

      return Status::OK();
    }

    uint32_t cur = begin;

    while (cur < end) {
      const uint32_t chunk_end =
          std::min<uint32_t>(end, cur + internal_chunk_rows);

      Status ens = ensure_active_delta(cur);
      if (!ens.ok()) return ens;

      if (!active_delta.has_value()) {
        return Status::Internal("active_delta missing after ensure");
      }

      Status materialize_status =
          ensure_whitened_materialized(chunk_end, "insert_after_transform");
      if (!materialize_status.ok()) return materialize_status;

      Timer t;

      auto add = AddRangeToIndexWithOnlinePQ(active_delta->ivf,
                                             Xw,
                                             cur,
                                             chunk_end,
                                             config.dim,
                                             active_delta->versions,
                                             online_opts);
      if (!add.ok()) return add.status();

      const double add_ms = t.ElapsedMillis();

      counters.AddInsertElapsed(add_ms);
      last_pq_stats = add.value();
      counters.AddInsertStats(last_pq_stats);
      counters.AddPendingInsertElapsed(add_ms);
      counters.AddDeltaIngestAssignment(last_pq_stats.insert_assignment_us,
                                        last_pq_stats.processed_vectors);
      std::cout << "[STREAM_PROFILE] begin=" << cur
                << ", end=" << chunk_end
                << ", delta_ingest_assignment_us="
                << last_pq_stats.insert_assignment_us
                << ", records=" << last_pq_stats.processed_vectors
                << std::endl;

      active_delta->rows += (chunk_end - cur);
      cur = chunk_end;
      seen_rows = cur;
      record_memory("insert_after_delta_add", seen_rows);

      Status fr = maybe_freeze_active();
      if (!fr.ok()) return fr;
    }

    Status gr = maybe_global_rebuild();
    if (!gr.ok()) return gr;

    return Status::OK();
  };

  std::vector<SnapshotLite> snapshots;
  uint32_t last_insert_begin = 0;
  uint32_t last_insert_end = 0;

  for (const auto& op : workload.operations) {
    if (op.operation == "insert") {
      if (op.end <= seen_rows) {
        last_insert_begin = seen_rows;
        last_insert_end = seen_rows;
        std::cout << "[INSERT] op_id=" << op.op_id
                  << " skipped duplicate/past range=["
                  << op.start << "," << op.end << ")"
                  << ", seen_rows=" << seen_rows
                  << std::endl;
        continue;
      }

      if (op.start > seen_rows) {
        std::cerr << "[ERROR] workload is not prefix-contiguous at op_id="
                  << op.op_id
                  << ": op.start=" << op.start
                  << " > seen_rows=" << seen_rows
                  << std::endl;
        return 1;
      }

      const uint32_t begin = std::max<uint32_t>(seen_rows, op.start);

      Status ins = insert_range(begin, op.end);
      if (!ins.ok()) {
        std::cerr << ins.ToString() << std::endl;
        return 1;
      }
      last_insert_begin = begin;
      last_insert_end = op.end;

      std::cout << (main_built ? "[INSERT]" : "[INSERT-WARMUP]")
                << " op_id=" << op.op_id
                << ", round=" << op.round
                << ", cluster=" << op.cluster
                << ", range=[" << begin << "," << op.end << ")"
                << ", seen_rows=" << seen_rows
                << ", main_built=" << (main_built ? "true" : "false")
                << std::endl;
    } else if (op.operation == "search") {
      counters.NextSearchId();

      SnapshotLite snap;
      snap.search_id = search_id;
      snap.op_id = op.op_id;
      snap.active_rows = seen_rows;
      snap.main_rows = main_rows_current;
      snap.active_delta_rows =
          active_delta.has_value() ? active_delta->rows : 0;
      snap.frozen_delta_rows =
          frozen_delta.has_value() ? frozen_delta->rows : 0;
      snap.round = op.round;
      snap.cluster = op.cluster;
      snap.cumulative_update_ms = total_update_ms;
      snap.cumulative_merge_ms = total_merge_compute_ms;
      snap.cumulative_global_rebuild_ms = total_global_rebuild_ms;

      if (!main_built || seen_rows == 0) {
        snap.skipped_warmup = true;
        record_memory("search_skipped_warmup", seen_rows);
        snapshots.push_back(snap);

        std::cout << "[SEARCH] op_id=" << op.op_id
                  << " skipped warmup, seen_rows=" << seen_rows
                  << ", main_bootstrap_rows=" << main_bootstrap_rows
                  << std::endl;
        continue;
      }

      auto ev = EvaluateStatePrefix(config,
                                    Xw,
                                    Xw_norms,
                                    seen_rows,
                                    last_insert_begin,
                                    last_insert_end,
                                    Qw,
                                    main_ivf,
                                    main_versions,
                                    main_rows_current,
                                    frozen_delta,
                                    active_delta,
                                    search_params);
      if (!ev.ok()) {
        std::cerr << ev.status().ToString() << std::endl;
        return 1;
      }

      snap.recall = ev.value().recall;
      snap.recall_new = ev.value().recall_new;
      snap.recall_old = ev.value().recall_old;
      snap.gt_new_ratio = ev.value().gt_new_ratio;
      snap.gt_new_total = ev.value().gt_new_total;
      snap.gt_old_total = ev.value().gt_old_total;
      snap.hit_new_total = ev.value().hit_new_total;
      snap.hit_old_total = ev.value().hit_old_total;
      snap.avg_query_ms = ev.value().avg_query_ms;
      snap.avg_search_ms = ev.value().avg_search_ms;
      snap.end_to_end_overhead_ms = ev.value().end_to_end_overhead_ms;
      snap.avg_pq_lut_build_us = ev.value().avg_pq_lut_build_us;
      snap.avg_pq_adc_scan_us = ev.value().avg_pq_adc_scan_us;
      snap.search_p50 = ev.value().search_p50;
      snap.search_p99 = ev.value().search_p99;
      snap.total_p50 = ev.value().total_p50;
      snap.total_p99 = ev.value().total_p99;
      snap.p50_query_ms = ev.value().p50_query_ms;
      snap.p99_query_ms = ev.value().p99_query_ms;
      snap.qps = ev.value().qps;
      snap.avg_scanned = ev.value().avg_scanned;
      snap.scanned_p50 = ev.value().scanned_p50;
      snap.scanned_p99 = ev.value().scanned_p99;
      snap.scanned_max = ev.value().scanned_max;
      snap.scan_ratio = ev.value().scan_ratio;
      snap.scanned_per_topk = ev.value().scanned_per_topk;
      snap.query_eval_ms = ev.value().query_eval_ms;
      snap.query_count = ev.value().query_count;
      snap.main_route_queries = ev.value().main_route_queries;
      snap.active_delta_route_queries = ev.value().active_delta_route_queries;
      snap.frozen_delta_route_queries = ev.value().frozen_delta_route_queries;
      snap.gt_probed_rate = ev.value().gt_probed_rate;
      snap.recall_on_probed_gt = ev.value().recall_on_probed_gt;
      snap.exact_recall_on_probed_candidates =
          ev.value().exact_recall_on_probed_candidates;
      snap.avg_pq_rank_loss = ev.value().avg_pq_rank_loss;
      snap.miss_not_probed = ev.value().miss_not_probed;
      snap.miss_probed_filtered_by_pq = ev.value().miss_probed_filtered_by_pq;
      snap.pq_rank_loss_count = ev.value().pq_rank_loss_count;
      snap.rerank_topk_main_total = ev.value().rerank_topk_main_total;
      snap.rerank_topk_delta_total = ev.value().rerank_topk_delta_total;
      snap.rerank_topk_main_ratio = ev.value().rerank_topk_main_ratio;
      snap.rerank_topk_delta_ratio = ev.value().rerank_topk_delta_ratio;
      snap.rerank_topk_main_avg = ev.value().rerank_topk_main_avg;
      snap.rerank_topk_delta_avg = ev.value().rerank_topk_delta_avg;
      snap.worst_queries = ev.value().worst_queries;
      snap.imbalance_ratio = current_imbalance_ratio();

      ann::eval::PrintMsEval(search_id, seen_rows, config.topk, snap);
      ann::eval::PrintMsDiagnostics(config, snap);

      const uint32_t snapshot_rows =
          seen_rows >= last_snapshot_active_rows
              ? seen_rows - last_snapshot_active_rows
              : 0;

      const double snapshot_maintenance_ms =
          pending_update_ms +
          pending_merge_compute_ms +
          pending_global_rebuild_ms;

      const double throughput =
          snapshot_rows > 0 && snapshot_maintenance_ms > 0.0
              ? 1000.0 * static_cast<double>(snapshot_rows) /
                    snapshot_maintenance_ms
              : 0.0;
      snap.snapshot_rows = static_cast<double>(snapshot_rows);
      snap.snapshot_maintenance_ms = snapshot_maintenance_ms;
      snap.maintenance_ms_per_vector =
          snapshot_rows > 0
              ? snapshot_maintenance_ms / static_cast<double>(snapshot_rows)
              : 0.0;
      snap.update_throughput_vecps = throughput;

      const uint32_t streamed_rows =
          seen_rows > main_bootstrap_rows
              ? seen_rows - main_bootstrap_rows
              : 0;

      const double cumulative_maintenance_ms =
          total_update_ms + total_merge_compute_ms + total_global_rebuild_ms;

      const double amortized =
          streamed_rows > 0 && cumulative_maintenance_ms > 0.0
              ? 1000.0 * static_cast<double>(streamed_rows) /
                    cumulative_maintenance_ms
              : 0.0;
      snap.amortized_update_throughput_vecps = amortized;

      record_memory("search_snapshot", seen_rows);
      snapshots.push_back(snap);

      ann::eval::PrintMsSnapshot(config.topk,
                                 seen_rows,
                                 snapshot_rows,
                                 throughput,
                                 snapshot_maintenance_ms,
                                 amortized,
                                 snap);

      last_snapshot_active_rows = seen_rows;
      counters.ResetPending();
    }
  }

  if (main_built &&
      (snapshots.empty() || snapshots.back().active_rows != seen_rows)) {
    auto ev = EvaluateStatePrefix(config,
                                  Xw,
                                  Xw_norms,
                                  seen_rows,
                                  last_insert_begin,
                                  last_insert_end,
                                  Qw,
                                  main_ivf,
                                  main_versions,
                                  main_rows_current,
                                  frozen_delta,
                                  active_delta,
                                  search_params);
    if (ev.ok()) {
      SnapshotLite snap;
      snap.search_id = search_id + 1;
      snap.op_id = 0;
      snap.active_rows = seen_rows;
      snap.main_rows = main_rows_current;
      snap.active_delta_rows =
          active_delta.has_value() ? active_delta->rows : 0;
      snap.frozen_delta_rows =
          frozen_delta.has_value() ? frozen_delta->rows : 0;
      snap.recall = ev.value().recall;
      snap.recall_new = ev.value().recall_new;
      snap.recall_old = ev.value().recall_old;
      snap.gt_new_ratio = ev.value().gt_new_ratio;
      snap.gt_new_total = ev.value().gt_new_total;
      snap.gt_old_total = ev.value().gt_old_total;
      snap.hit_new_total = ev.value().hit_new_total;
      snap.hit_old_total = ev.value().hit_old_total;
      snap.avg_query_ms = ev.value().avg_query_ms;
      snap.avg_search_ms = ev.value().avg_search_ms;
      snap.end_to_end_overhead_ms = ev.value().end_to_end_overhead_ms;
      snap.avg_pq_lut_build_us = ev.value().avg_pq_lut_build_us;
      snap.avg_pq_adc_scan_us = ev.value().avg_pq_adc_scan_us;
      snap.search_p50 = ev.value().search_p50;
      snap.search_p99 = ev.value().search_p99;
      snap.total_p50 = ev.value().total_p50;
      snap.total_p99 = ev.value().total_p99;
      snap.p50_query_ms = ev.value().p50_query_ms;
      snap.p99_query_ms = ev.value().p99_query_ms;
      snap.qps = ev.value().qps;
      snap.avg_scanned = ev.value().avg_scanned;
      snap.scanned_p50 = ev.value().scanned_p50;
      snap.scanned_p99 = ev.value().scanned_p99;
      snap.scanned_max = ev.value().scanned_max;
      snap.scan_ratio = ev.value().scan_ratio;
      snap.scanned_per_topk = ev.value().scanned_per_topk;
      snap.gt_probed_rate = ev.value().gt_probed_rate;
      snap.recall_on_probed_gt = ev.value().recall_on_probed_gt;
      snap.exact_recall_on_probed_candidates =
          ev.value().exact_recall_on_probed_candidates;
      snap.avg_pq_rank_loss = ev.value().avg_pq_rank_loss;
      snap.miss_not_probed = ev.value().miss_not_probed;
      snap.miss_probed_filtered_by_pq = ev.value().miss_probed_filtered_by_pq;
      snap.pq_rank_loss_count = ev.value().pq_rank_loss_count;
      snap.rerank_topk_main_total = ev.value().rerank_topk_main_total;
      snap.rerank_topk_delta_total = ev.value().rerank_topk_delta_total;
      snap.rerank_topk_main_ratio = ev.value().rerank_topk_main_ratio;
      snap.rerank_topk_delta_ratio = ev.value().rerank_topk_delta_ratio;
      snap.rerank_topk_main_avg = ev.value().rerank_topk_main_avg;
      snap.rerank_topk_delta_avg = ev.value().rerank_topk_delta_avg;
      snap.worst_queries = ev.value().worst_queries;
      snap.imbalance_ratio = current_imbalance_ratio();
      snap.cumulative_update_ms = total_update_ms;
      snap.cumulative_merge_ms = total_merge_compute_ms;
      snap.cumulative_global_rebuild_ms = total_global_rebuild_ms;
      snap.query_eval_ms = ev.value().query_eval_ms;
      snap.query_count = ev.value().query_count;
      snap.main_route_queries = ev.value().main_route_queries;
      snap.active_delta_route_queries = ev.value().active_delta_route_queries;
      snap.frozen_delta_route_queries = ev.value().frozen_delta_route_queries;
      const uint32_t snapshot_rows =
          seen_rows >= last_snapshot_active_rows
              ? seen_rows - last_snapshot_active_rows
              : 0;
      const double snapshot_maintenance_ms =
          pending_update_ms + pending_merge_compute_ms + pending_global_rebuild_ms;
      snap.snapshot_rows = static_cast<double>(snapshot_rows);
      snap.snapshot_maintenance_ms = snapshot_maintenance_ms;
      snap.maintenance_ms_per_vector =
          snapshot_rows > 0
              ? snapshot_maintenance_ms / static_cast<double>(snapshot_rows)
              : 0.0;
      snap.update_throughput_vecps =
          snapshot_rows > 0 && snapshot_maintenance_ms > 0.0
              ? 1000.0 * static_cast<double>(snapshot_rows) / snapshot_maintenance_ms
              : 0.0;
      const uint32_t streamed_rows =
          seen_rows > main_bootstrap_rows ? seen_rows - main_bootstrap_rows : 0;
      const double cumulative_maintenance_ms =
          total_update_ms + total_merge_compute_ms + total_global_rebuild_ms;
      snap.amortized_update_throughput_vecps =
          streamed_rows > 0 && cumulative_maintenance_ms > 0.0
              ? 1000.0 * static_cast<double>(streamed_rows) / cumulative_maintenance_ms
              : 0.0;
      record_memory("final_snapshot", seen_rows);
      snapshots.push_back(snap);
    }
  }

  record_memory("final", seen_rows);
  const std::vector<ann::eval_memory::MemoryComponent> memory_metadata = {
      {"base_rows", base_source.rows},
      {"base_dim", base_source.dim},
      {"base_raw_full_dataset_bytes",
       ann::eval_memory::MatrixActiveBytes(base_source.rows, base_source.dim)},
      {"query_rows", static_cast<uint64_t>(Q.rows())},
      {"query_dim", static_cast<uint64_t>(Q.cols())},
      {"memory_block_rows", kAddBlockRows},
      {"raw_base_is_resident", base_source.UsesResidentMatrix() ? 1ull : 0ull}};
  Status memory_write_status = ann::eval::WriteMemoryTraceJson(
      memory_trace, memory_trace_path.string(), memory_metadata);
  if (!memory_write_status.ok()) {
    std::cerr << memory_write_status.ToString() << std::endl;
    return 1;
  }

  std::vector<double> summary_recalls;
  std::vector<double> summary_qps;
  std::vector<double> summary_latencies;
  std::vector<double> summary_scanned;
  std::vector<double> summary_update_throughput;
  ann::eval::run_eval_ms::SummaryOutput output_summary;
  uint64_t maintenance_rows = 0;
  for (const auto& snapshot : snapshots) {
    if (snapshot.skipped_warmup) continue;
    output_summary.summary_gt_new_total += snapshot.gt_new_total;
    output_summary.summary_gt_old_total += snapshot.gt_old_total;
    output_summary.summary_hit_new_total += snapshot.hit_new_total;
    output_summary.summary_hit_old_total += snapshot.hit_old_total;
    summary_recalls.push_back(snapshot.recall);
    summary_qps.push_back(snapshot.qps);
    summary_latencies.push_back(snapshot.avg_query_ms);
    summary_scanned.push_back(snapshot.avg_scanned);
    output_summary.final_imbalance_ratio = snapshot.imbalance_ratio;
    output_summary.final_scan_ratio = snapshot.scan_ratio;
    output_summary.final_scanned_per_topk = snapshot.scanned_per_topk;
    maintenance_rows += static_cast<uint64_t>(snapshot.snapshot_rows);
    if (snapshot.update_throughput_vecps > 0.0) {
      summary_update_throughput.push_back(snapshot.update_throughput_vecps);
    }
  }
  output_summary.recall_stats = SummarizeValues(summary_recalls);
  output_summary.qps_stats = SummarizeValues(summary_qps);
  output_summary.latency_stats = SummarizeValues(summary_latencies);
  output_summary.scanned_stats = SummarizeValues(summary_scanned);
  output_summary.update_throughput_stats = SummarizeValues(summary_update_throughput);
  output_summary.recall_available = !summary_recalls.empty();
  output_summary.final_recall = summary_recalls.empty() ? 0.0 : summary_recalls.back();
  output_summary.total_maintenance_ms =
      total_update_ms + total_merge_compute_ms + total_global_rebuild_ms;
  output_summary.build_rebuild_ms = init_ms + total_global_rebuild_ms;
  const uint32_t initial_rows = snapshots.empty() ? 0u : snapshots.front().active_rows;
  const uint32_t streamed_rows = seen_rows > initial_rows ? seen_rows - initial_rows : 0;
  output_summary.amortized_update_throughput =
      streamed_rows > 0 && output_summary.total_maintenance_ms > 0.0
          ? 1000.0 * static_cast<double>(streamed_rows) / output_summary.total_maintenance_ms
          : 0.0;
  output_summary.maintenance_ms_per_vector =
      maintenance_rows > 0
          ? output_summary.total_maintenance_ms / static_cast<double>(maintenance_rows)
          : 0.0;
  for (auto it = snapshots.rbegin(); it != snapshots.rend(); ++it) {
    if (!it->skipped_warmup) {
      output_summary.final_snapshot = &*it;
      break;
    }
  }
  const uint64_t summary_gt_total = output_summary.summary_gt_new_total +
                                    output_summary.summary_gt_old_total;
  output_summary.summary_recall_new = output_summary.summary_gt_new_total > 0
      ? static_cast<double>(output_summary.summary_hit_new_total) /
            static_cast<double>(output_summary.summary_gt_new_total)
      : 0.0;
  output_summary.summary_recall_old = output_summary.summary_gt_old_total > 0
      ? static_cast<double>(output_summary.summary_hit_old_total) /
            static_cast<double>(output_summary.summary_gt_old_total)
      : 0.0;
  output_summary.summary_gt_new_ratio = summary_gt_total > 0
      ? static_cast<double>(output_summary.summary_gt_new_total) /
            static_cast<double>(summary_gt_total)
      : 0.0;
  const uint32_t updated_rows = seen_rows > initial_rows ? seen_rows - initial_rows : 0;
  output_summary.update_per_vector_ms = updated_rows > 0
      ? output_summary.total_maintenance_ms / static_cast<double>(updated_rows)
      : 0.0;
  output_summary.update_throughput_vecps = output_summary.total_maintenance_ms > 0.0
      ? 1000.0 * static_cast<double>(updated_rows) / output_summary.total_maintenance_ms
      : 0.0;

  ann::eval::WriteSummaryJson(summary_path,
                   config,
                   workload,
                   snapshots,
                   init_ms,
                   total_update_ms,
                   total_update_whitening_ms,
                   total_update_insert_ms,
                   total_update_record_build_ms,
                   total_update_insert_encode_ms,
                   total_update_insert_commit_ms,
                   total_update_onlinepq_maintenance_ms,
                   total_update_delete_ms,
                   total_update_codebook_update_ms,
                   total_update_reencode_ms,
                   total_delta_ingest_assignment_us,
                   total_delta_ingest_assignment_records,
                   total_merge_ms,
                   total_merge_compute_ms,
                   total_global_rebuild_ms,
                   merge_count,
                   global_rebuild_count,
                   merge_nodes,
                   global_rebuild_nodes,
                   seen_rows,
                   full_run_timer.ElapsedMillis(),
                   output_summary);

  std::cout << "[DONE] seen_rows=" << seen_rows
            << ", main_built=" << (main_built ? "true" : "false")
            << ", snapshots=" << snapshots.size()
            << ", final_recall="
            << (snapshots.empty() ? 0.0 : snapshots.back().recall)
            << ", avg_query_ms="
            << (snapshots.empty() ? 0.0 : snapshots.back().avg_query_ms)
            << ", query_qps=" << (snapshots.empty() ? 0.0 : snapshots.back().qps)
            << ", search_p50_ms="
            << (snapshots.empty() ? 0.0 : snapshots.back().search_p50)
            << ", search_p99_ms="
            << (snapshots.empty() ? 0.0 : snapshots.back().search_p99)
            << ", total_p50_ms="
            << (snapshots.empty() ? 0.0 : snapshots.back().total_p50)
            << ", total_p99_ms="
            << (snapshots.empty() ? 0.0 : snapshots.back().total_p99)
            << ", scanned_avg="
            << (snapshots.empty() ? 0.0 : snapshots.back().avg_scanned)
            << ", scanned_p50="
            << (snapshots.empty() ? 0.0 : snapshots.back().scanned_p50)
            << ", scanned_p99="
            << (snapshots.empty() ? 0.0 : snapshots.back().scanned_p99)
            << ", scanned_max="
            << (snapshots.empty() ? 0.0 : snapshots.back().scanned_max)
            << ", update_ms=" << total_update_ms
            << ", merge_ms=" << total_merge_ms
            << ", delta_ingest_assignment_us="
            << total_delta_ingest_assignment_us
            << ", delta_ingest_assignment_records="
            << total_delta_ingest_assignment_records
            << ", merge_count=" << merge_count
            << ", global_rebuild_ms=" << total_global_rebuild_ms
            << ", summary=" << summary_path
            << ", memory_trace=" << memory_trace_path
            << std::endl;

  return 0;
}
