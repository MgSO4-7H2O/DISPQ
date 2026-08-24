#include <algorithm>
#include <array>
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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

#include <Eigen/Dense>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/config.h"
#include "common/dataset.h"
#include "common/timer.h"
#include "common/types.h"
#include "common/vector_accessor.h"
#include "eval_memory.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "index/merge.h"
#include "search/hybrid_search.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

constexpr uint32_t kDeltaKMeansIterationsDefault = 10;
constexpr uint32_t kAddBlockRows = 65536;
constexpr uint32_t kWorstQueryDiagCount = 10;
constexpr uint32_t kSlowQueryDebugCount = 5;
constexpr size_t kGroundTruthBlockTargetBytes = static_cast<size_t>(64) << 20;

struct DocRange {
  DocId begin{0};
  DocId end{0};  // Exclusive.
};

class DocRangeQueue {
 public:
  void Append(DocId doc_id) { AppendRange(doc_id, doc_id + 1); }

  void AppendRange(DocId begin, DocId end) {
    if (begin == end) {
      return;
    }
    if (!ranges_.empty() && ranges_.back().end == begin) {
      ranges_.back().end = end;
    } else {
      ranges_.push_back({begin, end});
    }
    size_ += static_cast<size_t>(end - begin);
  }

  void PopFront(size_t count) {
    size_ -= count;
    while (count > 0) {
      DocRange& range = ranges_.front();
      const size_t range_size = static_cast<size_t>(range.end - range.begin);
      if (count < range_size) {
        range.begin += static_cast<DocId>(count);
        return;
      }
      count -= range_size;
      ranges_.pop_front();
    }
  }

  std::vector<DocId> Materialize() const { return MaterializeFront(size_); }

  std::vector<DocId> MaterializeFront(size_t count) const {
    std::vector<DocId> doc_ids;
    doc_ids.reserve(count);
    for (const DocRange& range : ranges_) {
      const size_t range_size = static_cast<size_t>(range.end - range.begin);
      const size_t take = std::min(count - doc_ids.size(), range_size);
      for (size_t i = 0; i < take; ++i) {
        doc_ids.push_back(range.begin + static_cast<DocId>(i));
      }
      if (doc_ids.size() == count) {
        break;
      }
    }
    return doc_ids;
  }

  DocId front() const { return ranges_.front().begin; }
  DocId back() const { return ranges_.back().end - 1; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  void clear() {
    ranges_.clear();
    size_ = 0;
  }

  uint64_t EstimatedMemoryBytes() const {
    return static_cast<uint64_t>(sizeof(*this)) +
           static_cast<uint64_t>(ranges_.size()) * sizeof(DocRange);
  }

 private:
  std::deque<DocRange> ranges_;
  size_t size_{0};
};

void TrimAllocatorRetainedMemory(const char* stage) {
#ifdef __GLIBC__
  const int released = malloc_trim(0);
  std::cout << "[MALLOC_TRIM] stage=" << stage
            << ", released=" << released << std::endl;
#else
  (void)stage;
#endif
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

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
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


class VecsRandomAccessReader {
 public:
  Result<void> Open(const std::string& path) {
    path_ = path;
    const std::string extension = std::filesystem::path(path_).extension().string();
    if (extension == ".fvecs") {
      value_bytes_ = sizeof(float);
    } else if (extension == ".bvecs") {
      value_bytes_ = sizeof(uint8_t);
    } else {
      return Status::InvalidArgument(
          "VecsRandomAccessReader: expected .fvecs or .bvecs: " + path_);
    }
    std::ifstream ifs(path_, std::ios::binary);
    if (!ifs) {
      return Status::IOError("VecsRandomAccessReader: cannot open " + path_);
    }
    int32_t dim_i32 = 0;
    ifs.read(reinterpret_cast<char*>(&dim_i32), sizeof(int32_t));
    if (!ifs || dim_i32 <= 0) {
      return Status::InvalidArgument("VecsRandomAccessReader: invalid vector header");
    }
    dim_ = static_cast<uint32_t>(dim_i32);
    record_bytes_ = sizeof(int32_t) + static_cast<uint64_t>(dim_) * value_bytes_;
    std::error_code ec;
    const uint64_t file_bytes = static_cast<uint64_t>(std::filesystem::file_size(path_, ec));
    if (ec || file_bytes < record_bytes_ || file_bytes % record_bytes_ != 0) {
      return Status::InvalidArgument("VecsRandomAccessReader: invalid vector file size");
    }
    count_ = static_cast<uint32_t>(file_bytes / record_bytes_);
    return Result<void>::Ok();
  }

  uint32_t dim() const { return dim_; }
  uint32_t count() const { return count_; }

  Result<MatrixRM> ReadRange(uint32_t begin, uint32_t rows) const {
    if (begin > count_ || rows > count_ - begin) {
      return Status::InvalidArgument("VecsRandomAccessReader::ReadRange: range out of bounds");
    }
    MatrixRM out(static_cast<Eigen::Index>(rows), dim_);
    if (rows == 0) {
      return out;
    }
    std::ifstream ifs(path_, std::ios::binary);
    if (!ifs) {
      return Status::IOError("VecsRandomAccessReader::ReadRange: cannot open " + path_);
    }
    ifs.seekg(static_cast<std::streamoff>(static_cast<uint64_t>(begin) * record_bytes_),
              std::ios::beg);
    std::vector<uint8_t> bytes(value_bytes_ == sizeof(uint8_t) ? dim_ : 0u);
    for (uint32_t r = 0; r < rows; ++r) {
      int32_t dim_i32 = 0;
      ifs.read(reinterpret_cast<char*>(&dim_i32), sizeof(int32_t));
      if (!ifs || dim_i32 != static_cast<int32_t>(dim_)) {
        return Status::InvalidArgument("VecsRandomAccessReader::ReadRange: dim mismatch");
      }
      if (value_bytes_ == sizeof(float)) {
        ifs.read(reinterpret_cast<char*>(out.row(static_cast<Eigen::Index>(r)).data()),
                 static_cast<std::streamsize>(dim_) *
                     static_cast<std::streamsize>(sizeof(float)));
      } else {
        ifs.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
        for (uint32_t c = 0; c < dim_; ++c) {
          out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
              static_cast<float>(bytes[c]);
        }
      }
      if (!ifs) {
        return Status::IOError("VecsRandomAccessReader::ReadRange: short read");
      }
    }
    return out;
  }

  Result<MatrixRM> ReadDocIds(const std::vector<DocId>& doc_ids) const {
    MatrixRM out(static_cast<Eigen::Index>(doc_ids.size()), dim_);
    if (doc_ids.empty()) {
      return out;
    }
    std::vector<std::pair<DocId, size_t>> order;
    order.reserve(doc_ids.size());
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      if (doc_ids[i] >= count_) {
        return Status::InvalidArgument("VecsRandomAccessReader::ReadDocIds: doc_id out of bounds");
      }
      order.emplace_back(doc_ids[i], i);
    }
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    size_t pos = 0;
    while (pos < order.size()) {
      const DocId begin = order[pos].first;
      size_t next = pos + 1;
      while (next < order.size() && order[next].first == order[next - 1].first + 1) {
        ++next;
      }
      auto block_res = ReadRange(begin, static_cast<uint32_t>(next - pos));
      if (!block_res.ok()) {
        return block_res.status();
      }
      const MatrixRM& block = block_res.value();
      for (size_t i = pos; i < next; ++i) {
        out.row(static_cast<Eigen::Index>(order[i].second)) =
            block.row(static_cast<Eigen::Index>(i - pos));
      }
      pos = next;
    }
    return out;
  }

 private:
  std::string path_;
  uint32_t dim_{0};
  uint32_t count_{0};
  uint64_t value_bytes_{0};
  uint64_t record_bytes_{0};
};

Result<std::string> ResolveVecsPath(const std::string& path_or_dir,
                                    const std::string& role_suffix) {
  auto fvecs = ResolveFvecsPath(path_or_dir, role_suffix + ".fvecs");
  if (fvecs.ok()) {
    return fvecs.value();
  }
  auto bvecs = ResolveFvecsPath(path_or_dir, role_suffix + ".bvecs");
  if (bvecs.ok()) {
    return bvecs.value();
  }
  return Status::NotFound("No " + role_suffix + ".fvecs or " + role_suffix +
                          ".bvecs file found under " + path_or_dir);
}

void ReleaseMatrix(MatrixRM* matrix) {
  if (matrix == nullptr) {
    return;
  }
  MatrixRM empty;
  matrix->swap(empty);
}

struct MeanCovStats {
  Eigen::VectorXf mean;
  MatrixRM covariance;
  uint64_t count{0};
  double read_ms{0.0};
};

class MeanCovAccumulator {
 public:
  explicit MeanCovAccumulator(uint32_t dim)
      : dim_(dim),
        sum_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dim))),
        cross_(Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(dim),
                                     static_cast<Eigen::Index>(dim))) {}

  Status Add(Eigen::Ref<const MatrixRM> rows) {
    if (rows.rows() == 0) {
      return Status::OK();
    }
    if (static_cast<uint32_t>(rows.cols()) != dim_) {
      return Status::InvalidArgument("MeanCovAccumulator::Add: dimension mismatch");
    }
    const Eigen::VectorXf block_sum = rows.colwise().sum();
    sum_ += block_sum.cast<double>();
    const MatrixRM block_cross = rows.transpose() * rows;
    cross_ += block_cross.cast<double>();
    count_ += static_cast<uint64_t>(rows.rows());
    return Status::OK();
  }

  Result<MeanCovStats> Finish() const {
    if (count_ == 0) {
      return Status::InvalidArgument("MeanCovAccumulator::Finish: no samples");
    }
    const Eigen::VectorXd mean_d = sum_ / static_cast<double>(count_);
    Eigen::MatrixXd cov_d = cross_ / static_cast<double>(count_);
    cov_d.noalias() -= mean_d * mean_d.transpose();
    MeanCovStats stats;
    stats.mean = mean_d.cast<float>();
    stats.covariance = cov_d.cast<float>();
    stats.count = count_;
    return stats;
  }

 private:
  uint32_t dim_{0};
  uint64_t count_{0};
  Eigen::VectorXd sum_;
  Eigen::MatrixXd cross_;
};

Result<MeanCovStats> ComputeMeanCovForRange(const VecsRandomAccessReader& reader,
                                            uint32_t begin,
                                            uint32_t rows) {
  if (begin > reader.count() || rows > reader.count() - begin) {
    return Status::InvalidArgument("ComputeMeanCovForRange: range out of bounds");
  }
  MeanCovAccumulator accumulator(reader.dim());
  double read_ms = 0.0;
  for (uint32_t offset = 0; offset < rows;) {
    const uint32_t chunk = std::min<uint32_t>(kAddBlockRows, rows - offset);
    Timer read_timer;
    auto block_res = reader.ReadRange(begin + offset, chunk);
    if (!block_res.ok()) {
      return block_res.status();
    }
    read_ms += read_timer.ElapsedMillis();
    Status add = accumulator.Add(block_res.value());
    if (!add.ok()) {
      return add;
    }
    offset += chunk;
  }
  auto stats_res = accumulator.Finish();
  if (!stats_res.ok()) {
    return stats_res.status();
  }
  MeanCovStats stats = std::move(stats_res.value());
  stats.read_ms = read_ms;
  return stats;
}

Result<MeanCovStats> ComputeMeanCovForDocIds(const VecsRandomAccessReader& reader,
                                             const std::vector<DocId>& doc_ids) {
  if (doc_ids.empty()) {
    return Status::InvalidArgument("ComputeMeanCovForDocIds: empty doc_ids");
  }
  MeanCovAccumulator accumulator(reader.dim());
  double read_ms = 0.0;
  size_t pos = 0;
  while (pos < doc_ids.size()) {
    const DocId begin = doc_ids[pos];
    if (begin >= reader.count()) {
      return Status::InvalidArgument("ComputeMeanCovForDocIds: doc_id out of bounds");
    }
    size_t next = pos + 1;
    while (next < doc_ids.size() && doc_ids[next] == doc_ids[next - 1] + 1 &&
           next - pos < kAddBlockRows) {
      ++next;
    }
    const uint32_t rows = static_cast<uint32_t>(next - pos);
    Timer read_timer;
    auto block_res = reader.ReadRange(begin, rows);
    if (!block_res.ok()) {
      return block_res.status();
    }
    read_ms += read_timer.ElapsedMillis();
    Status add = accumulator.Add(block_res.value());
    if (!add.ok()) {
      return add;
    }
    pos = next;
  }
  auto stats_res = accumulator.Finish();
  if (!stats_res.ok()) {
    return stats_res.status();
  }
  MeanCovStats stats = std::move(stats_res.value());
  stats.read_ms = read_ms;
  return stats;
}

Result<MatrixRM> TransformRangeToWhitened(const VecsRandomAccessReader& reader,
                                          uint32_t begin,
                                          uint32_t rows,
                                          const std::shared_ptr<WhiteningModel>& whitening,
                                          VersionId whiten_version,
                                          bool normalize_rows,
                                          double* read_ms,
                                          double* transform_ms) {
  if (!whitening) {
    return Status::InvalidArgument("TransformRangeToWhitened: null whitening model");
  }
  if (begin > reader.count() || rows > reader.count() - begin) {
    return Status::InvalidArgument("TransformRangeToWhitened: range out of bounds");
  }
  MatrixRM out(static_cast<Eigen::Index>(rows), reader.dim());
  for (uint32_t offset = 0; offset < rows;) {
    const uint32_t chunk = std::min<uint32_t>(kAddBlockRows, rows - offset);
    Timer read_timer;
    auto raw_res = reader.ReadRange(begin + offset, chunk);
    if (!raw_res.ok()) {
      return raw_res.status();
    }
    if (read_ms != nullptr) {
      *read_ms += read_timer.ElapsedMillis();
    }
    Timer transform_timer;
    auto whiten_res = whitening->TransformBatch(raw_res.value(), whiten_version);
    if (!whiten_res.ok()) {
      return whiten_res.status();
    }
    MatrixRM block = std::move(whiten_res.value());
    if (normalize_rows) {
      NormalizeRowsL2(&block);
    }
    if (transform_ms != nullptr) {
      *transform_ms += transform_timer.ElapsedMillis();
    }
    out.block(static_cast<Eigen::Index>(offset), 0, block.rows(), block.cols()) = block;
    offset += chunk;
  }
  return out;
}

Result<MatrixRM> TransformDocIdsToWhitened(const VecsRandomAccessReader& reader,
                                           const std::vector<DocId>& doc_ids,
                                           const std::shared_ptr<WhiteningModel>& whitening,
                                           VersionId whiten_version,
                                           bool normalize_rows,
                                           double* read_ms,
                                           double* transform_ms) {
  if (!whitening) {
    return Status::InvalidArgument("TransformDocIdsToWhitened: null whitening model");
  }
  MatrixRM out(static_cast<Eigen::Index>(doc_ids.size()), reader.dim());
  if (doc_ids.empty()) {
    return out;
  }
  size_t pos = 0;
  while (pos < doc_ids.size()) {
    const DocId begin = doc_ids[pos];
    if (begin >= reader.count()) {
      return Status::InvalidArgument("TransformDocIdsToWhitened: doc_id out of bounds");
    }
    size_t next = pos + 1;
    while (next < doc_ids.size() && doc_ids[next] == doc_ids[next - 1] + 1 &&
           next - pos < kAddBlockRows) {
      ++next;
    }
    const uint32_t rows = static_cast<uint32_t>(next - pos);
    Timer read_timer;
    auto raw_res = reader.ReadRange(begin, rows);
    if (!raw_res.ok()) {
      return raw_res.status();
    }
    if (read_ms != nullptr) {
      *read_ms += read_timer.ElapsedMillis();
    }
    Timer transform_timer;
    auto whiten_res = whitening->TransformBatch(raw_res.value(), whiten_version);
    if (!whiten_res.ok()) {
      return whiten_res.status();
    }
    MatrixRM block = std::move(whiten_res.value());
    if (normalize_rows) {
      NormalizeRowsL2(&block);
    }
    if (transform_ms != nullptr) {
      *transform_ms += transform_timer.ElapsedMillis();
    }
    out.block(static_cast<Eigen::Index>(pos), 0, block.rows(), block.cols()) = block;
    pos = next;
  }
  return out;
}

class WhitenedVectorCache final : public VectorAccessor {
 public:
  struct ReclaimStats {
    uint64_t reclaimed_vector_bytes{0};
    uint64_t reclaim_calls{0};
    uint64_t dead_prefix_rows{0};
    uint64_t reclaim_failures{0};
  };

  WhitenedVectorCache() = default;
  WhitenedVectorCache(const WhitenedVectorCache&) = delete;
  WhitenedVectorCache& operator=(const WhitenedVectorCache&) = delete;
  WhitenedVectorCache(WhitenedVectorCache&&) noexcept = default;
  WhitenedVectorCache& operator=(WhitenedVectorCache&&) noexcept = default;

  uint32_t dim() const override { return dim_; }
  size_t size() const { return locations_.size(); }

  ReclaimStats GetReclaimStats() const {
    ReclaimStats stats;
    stats.reclaimed_vector_bytes = reclaimed_vector_bytes_;
    stats.reclaim_calls = reclaim_calls_;
    stats.reclaim_failures = reclaim_failures_;
    for (const Block& block : blocks_) {
      stats.dead_prefix_rows += static_cast<uint64_t>(block.dead_prefix_rows);
    }
    return stats;
  }

  void Clear(uint32_t dim) {
    dim_ = dim;
    blocks_.clear();
    free_blocks_.clear();
    locations_.Clear();
    reclaimed_vector_bytes_ = 0;
    reclaim_calls_ = 0;
    reclaim_failures_ = 0;
  }

  void Reserve(size_t rows) { locations_.Reserve(rows); }

  Status AppendBatch(const std::vector<DocId>& doc_ids, const MatrixRM& vectors) {
    std::vector<DocId> ids_copy = doc_ids;
    MatrixRM vectors_copy = vectors;
    return AppendBatchOwned(std::move(ids_copy), std::move(vectors_copy));
  }

  Status AppendBatchOwned(std::vector<DocId> doc_ids, MatrixRM vectors) {
    if (doc_ids.size() != static_cast<size_t>(vectors.rows())) {
      return Status::InvalidArgument("WhitenedVectorCache::AppendBatch: row count mismatch");
    }
    if (vectors.rows() == 0) {
      return Status::OK();
    }
    if (dim_ == 0) {
      dim_ = static_cast<uint32_t>(vectors.cols());
    }
    if (static_cast<uint32_t>(vectors.cols()) != dim_) {
      return Status::InvalidArgument("WhitenedVectorCache::AppendBatch: dimension mismatch");
    }
    std::unordered_set<DocId> batch_ids;
    batch_ids.reserve(doc_ids.size() * 2 + 1);
    for (DocId doc_id : doc_ids) {
      if (!batch_ids.insert(doc_id).second || locations_.Get(doc_id, nullptr)) {
        return Status::AlreadyExists("WhitenedVectorCache::AppendBatch: duplicate doc_id");
      }
    }
    const uint32_t row_count = static_cast<uint32_t>(doc_ids.size());
    Block block;
    block.ids = std::move(doc_ids);
    block.data = std::move(vectors);
    block.norms = block.data.rowwise().squaredNorm();
    block.live = row_count;
    uint32_t block_id = static_cast<uint32_t>(blocks_.size());
    if (!free_blocks_.empty()) {
      block_id = free_blocks_.back();
      free_blocks_.pop_back();
    }
    for (uint32_t i = 0; i < row_count; ++i) {
      locations_.Set(block.ids[static_cast<size_t>(i)], Location{block_id, i});
    }
    if (block_id == blocks_.size()) {
      blocks_.push_back(std::move(block));
    } else {
      blocks_[block_id] = std::move(block);
    }
    return Status::OK();
  }

  std::vector<ann::eval_memory::MemoryComponent> EstimateMemoryComponents(
      const std::string& prefix) const {
    uint64_t block_container = static_cast<uint64_t>(blocks_.capacity()) *
                               static_cast<uint64_t>(sizeof(Block));
    uint64_t ids_bytes = 0;
    uint64_t data_bytes = 0;
    uint64_t norms_bytes = 0;
    for (const Block& block : blocks_) {
      ids_bytes += ann::eval_memory::StdVectorBytes(block.ids);
      data_bytes += ann::eval_memory::MatrixBytes(block.data);
      norms_bytes += ann::eval_memory::VectorBytes(block.norms);
    }
    const uint64_t free_bytes = ann::eval_memory::StdVectorBytes(free_blocks_);
    const uint64_t location_bytes = locations_.MemoryBytes();
    std::vector<ann::eval_memory::MemoryComponent> components;
    components.push_back({prefix + ".blocks", block_container});
    components.push_back({prefix + ".ids", ids_bytes});
    components.push_back({prefix + ".vectors", data_bytes});
    components.push_back({prefix + ".norms", norms_bytes});
    components.push_back({prefix + ".free_blocks", free_bytes});
    components.push_back({prefix + ".locations", location_bytes});
    components.push_back({prefix + ".total",
                          block_container + ids_bytes + data_bytes + norms_bytes +
                              free_bytes + location_bytes});
    return components;
  }

  Status RemoveDocIds(const std::vector<DocId>& doc_ids) {
    std::vector<uint32_t> touched_prefix_blocks;
    touched_prefix_blocks.reserve(std::min(doc_ids.size(), blocks_.size()));
    auto reclaim_touched_prefixes = [&]() {
      for (uint32_t block_id : touched_prefix_blocks) {
        Block& block = blocks_[block_id];
        if (block.live != 0) {
          AdvanceAndReclaimDeadPrefix(&block);
        }
      }
    };
    for (DocId doc_id : doc_ids) {
      Location loc;
      if (!locations_.Get(doc_id, &loc)) {
        reclaim_touched_prefixes();
        return Status::NotFound("WhitenedVectorCache::RemoveDocIds: missing doc_id");
      }
      if (loc.block >= blocks_.size() || loc.row >= blocks_[loc.block].ids.size()) {
        reclaim_touched_prefixes();
        return Status::Internal("WhitenedVectorCache::RemoveDocIds: corrupt location");
      }
      Block& block = blocks_[loc.block];
      if (block.ids[loc.row] != doc_id) {
        reclaim_touched_prefixes();
        return Status::Internal("WhitenedVectorCache::RemoveDocIds: doc_id/location mismatch");
      }
      if (loc.row == block.dead_prefix_rows) {
        touched_prefix_blocks.push_back(loc.block);
      }
      block.ids[loc.row] = kDeletedDoc;
      if (block.live > 0) {
        block.live--;
      }
      if (block.live == 0) {
        block.ids.clear();
        block.data.resize(0, 0);
        block.norms.resize(0);
        block.dead_prefix_rows = 0;
        free_blocks_.push_back(loc.block);
      }
      locations_.Erase(doc_id);
    }
    reclaim_touched_prefixes();
    return Status::OK();
  }

  Result<Eigen::VectorXf> GetVector(DocId doc_id) const override {
    Location loc;
    if (!locations_.Get(doc_id, &loc)) {
      return Status::NotFound("WhitenedVectorCache::GetVector: missing doc_id");
    }
    if (loc.block >= blocks_.size()) {
      return Status::Internal("WhitenedVectorCache::GetVector: corrupt block index");
    }
    const Block& block = blocks_[loc.block];
    if (loc.row >= block.ids.size() || block.ids[loc.row] != doc_id || block.data.rows() == 0) {
      return Status::Internal("WhitenedVectorCache::GetVector: corrupt row index");
    }
    return Eigen::VectorXf(block.data.row(static_cast<Eigen::Index>(loc.row)).transpose());
  }

  Result<float> GetNorm(DocId doc_id) const override {
    Location loc;
    if (!locations_.Get(doc_id, &loc)) {
      return Status::NotFound("WhitenedVectorCache::GetNorm: missing doc_id");
    }
    if (loc.block >= blocks_.size()) {
      return Status::Internal("WhitenedVectorCache::GetNorm: corrupt block index");
    }
    const Block& block = blocks_[loc.block];
    if (loc.row >= block.ids.size() || block.ids[loc.row] != doc_id ||
        loc.row >= static_cast<uint32_t>(block.norms.size())) {
      return Status::Internal("WhitenedVectorCache::GetNorm: corrupt row index");
    }
    return block.norms(static_cast<Eigen::Index>(loc.row));
  }

  Status Materialize(const std::vector<DocId>& doc_ids, MatrixRM* out) const override {
    if (out == nullptr) {
      return Status::InvalidArgument("WhitenedVectorCache::Materialize: null output");
    }
    MatrixRM rows(static_cast<Eigen::Index>(doc_ids.size()), dim_);
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      Location loc;
      if (!locations_.Get(doc_ids[i], &loc)) {
        return Status::NotFound("WhitenedVectorCache::Materialize: missing doc_id");
      }
      const Block& block = blocks_[loc.block];
      if (loc.row >= block.ids.size() || block.ids[loc.row] != doc_ids[i] || block.data.rows() == 0) {
        return Status::Internal("WhitenedVectorCache::Materialize: corrupt location");
      }
      rows.row(static_cast<Eigen::Index>(i)) = block.data.row(static_cast<Eigen::Index>(loc.row));
    }
    *out = std::move(rows);
    return Status::OK();
  }

 private:
  static constexpr DocId kDeletedDoc = std::numeric_limits<DocId>::max();
  static constexpr uint32_t kInvalidBlock = std::numeric_limits<uint32_t>::max();
  struct Location {
    uint32_t block{kInvalidBlock};
    uint32_t row{0};
  };
  class PagedLocationMap {
   public:
    bool Get(DocId doc_id, Location* out) const {
      const auto it = pages_.find(doc_id >> kPageBits);
      if (it == pages_.end()) {
        return false;
      }
      const Location& location = it->second->values[doc_id & kPageMask];
      if (location.block == kInvalidBlock) {
        return false;
      }
      if (out != nullptr) {
        *out = location;
      }
      return true;
    }

    void Set(DocId doc_id, Location location) {
      auto [it, inserted] = pages_.try_emplace(doc_id >> kPageBits);
      if (inserted) {
        it->second = std::make_unique<Page>();
      }
      Page& page = *it->second;
      Location& current = page.values[doc_id & kPageMask];
      if (current.block == kInvalidBlock) {
        ++page.live;
        ++size_;
      }
      current = location;
    }

    void Erase(DocId doc_id) {
      const auto it = pages_.find(doc_id >> kPageBits);
      if (it == pages_.end()) {
        return;
      }
      Page& page = *it->second;
      Location& location = page.values[doc_id & kPageMask];
      if (location.block == kInvalidBlock) {
        return;
      }
      location = Location{};
      --page.live;
      --size_;
      if (page.live == 0) {
        pages_.erase(it);
      }
    }

    void Clear() {
      pages_.clear();
      size_ = 0;
    }

    void Reserve(size_t docs) { pages_.reserve((docs + kPageSize - 1) / kPageSize); }

    size_t size() const { return size_; }

    uint64_t MemoryBytes() const {
      return static_cast<uint64_t>(sizeof(*this)) +
             static_cast<uint64_t>(pages_.bucket_count()) *
                 static_cast<uint64_t>(sizeof(void*)) +
             static_cast<uint64_t>(pages_.size()) *
                 static_cast<uint64_t>(sizeof(Page) + sizeof(uint32_t) +
                                       sizeof(std::unique_ptr<Page>) + sizeof(void*));
    }

   private:
    static constexpr uint32_t kPageBits = 8;
    static constexpr uint32_t kPageSize = 1u << kPageBits;
    static constexpr uint32_t kPageMask = kPageSize - 1;

    struct Page {
      std::array<Location, kPageSize> values{};
      uint32_t live{0};
    };

    std::unordered_map<uint32_t, std::unique_ptr<Page>> pages_;
    size_t size_{0};
  };
  struct Block {
    std::vector<DocId> ids;
    MatrixRM data;
    Eigen::VectorXf norms;
    uint32_t live{0};
    size_t dead_prefix_rows{0};
  };

  void AdvanceAndReclaimDeadPrefix(Block* block) {
    const size_t old_dead_prefix_rows = block->dead_prefix_rows;
    while (block->dead_prefix_rows < block->ids.size() &&
           block->ids[block->dead_prefix_rows] == kDeletedDoc) {
      ++block->dead_prefix_rows;
    }
    if (block->dead_prefix_rows == old_dead_prefix_rows) {
      return;
    }

#ifdef __linux__
    static const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
      ++reclaim_failures_;
      return;
    }
    const uintptr_t page_size = static_cast<uintptr_t>(page_size_value);
    const uintptr_t base = reinterpret_cast<uintptr_t>(block->data.data());
    const uintptr_t row_bytes = static_cast<uintptr_t>(block->data.cols()) * sizeof(float);
    // Exclude allocator metadata, the leading partial page, and the page containing live bytes.
    const uintptr_t first_full_page = ((base + page_size - 1) / page_size) * page_size;
    const uintptr_t old_dead_end = base + old_dead_prefix_rows * row_bytes;
    const uintptr_t new_dead_end = base + block->dead_prefix_rows * row_bytes;
    const uintptr_t reclaim_begin =
        std::max(first_full_page, old_dead_end - old_dead_end % page_size);
    const uintptr_t reclaim_end =
        std::max(first_full_page, new_dead_end - new_dead_end % page_size);
    if (reclaim_end > reclaim_begin) {
      ++reclaim_calls_;
      const size_t reclaim_bytes = static_cast<size_t>(reclaim_end - reclaim_begin);
      if (madvise(reinterpret_cast<void*>(reclaim_begin), reclaim_bytes, MADV_DONTNEED) == 0) {
        reclaimed_vector_bytes_ += static_cast<uint64_t>(reclaim_bytes);
      } else {
        ++reclaim_failures_;
      }
    }
#endif
  }

  uint32_t dim_{0};
  std::vector<Block> blocks_;
  std::vector<uint32_t> free_blocks_;
  PagedLocationMap locations_;
  uint64_t reclaimed_vector_bytes_{0};
  uint64_t reclaim_calls_{0};
  uint64_t reclaim_failures_{0};
};

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

struct DeltaActivationMetrics {
  double materialize_ms{0.0};
  double whitening_ms{0.0};
  double cache_ms{0.0};
  double build_ms{0.0};
  double insert_ms{0.0};
  double wall_ms{0.0};
};

Result<DeltaShard> BuildDeltaShard(Eigen::Ref<const MatrixRM> train_data,
  const IVFParams& ivf_params,
  VersionId whiten_version,
  uint32_t shard_id);

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
  double update_wall_total_ms{0.0};
  double update_wall_per_vector_ms{0.0};
  double update_wall_throughput_vecps{0.0};
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
  double update_wall_ms{0.0};
  double update_materialize_ms{0.0};
  double update_whitening_ms{0.0};
  double update_insert_ms{0.0};
  double update_record_build_ms{0.0};
  double update_insert_encode_ms{0.0};
  double update_insert_entry_ms{0.0};
  double update_insert_commit_ms{0.0};
  double update_onlinepq_maintenance_ms{0.0};
  double update_delete_ms{0.0};
  double update_onlinepq_stats_ms{0.0};
  double update_codebook_update_ms{0.0};
  double update_reencode_ms{0.0};
  double update_throughput_vecps{0.0};
  double update_wall_throughput_vecps{0.0};
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
  double update_wall_ms{0.0};
  double update_materialize_ms{0.0};
  double update_whitening_ms{0.0};
  double update_insert_ms{0.0};
  double update_record_build_ms{0.0};
  double update_insert_encode_ms{0.0};
  double update_insert_entry_ms{0.0};
  double update_insert_commit_ms{0.0};
  double update_onlinepq_maintenance_ms{0.0};
  double update_delete_ms{0.0};
  double update_onlinepq_stats_ms{0.0};
  double update_codebook_update_ms{0.0};
  double update_reencode_ms{0.0};
  double update_replace_ms{0.0};
  double query_eval_ms{0.0};
  double merge_compute_ms{0.0};
  double merge_wall_ms{0.0};
  double global_rebuild_ms{0.0};
  double global_rebuild_wall_ms{0.0};
  double snapshot_total_ms{0.0};
  double snapshot_wall_total_ms{0.0};
  double update_throughput_vecps{0.0};
  double update_wall_throughput_vecps{0.0};
  double amortized_update_throughput_vecps{0.0};
  double amortized_update_wall_throughput_vecps{0.0};
  uint64_t rss_bytes{0};
  uint64_t peak_rss_bytes{0};
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
  uint32_t evicted_main_rows{0};
  uint32_t main_rows_before{0};
  uint32_t main_rows_after{0};
  DocId live_main_begin_doc{0};
  DocId live_main_end_doc{0};
  double main_materialize_ms{0.0};
  double main_build_ms{0.0};
  double main_add_ms{0.0};
  double pre_replacement_ms{0.0};
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
  MergeProfiling profiling;
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
  double materialize_ms{0.0};
  double whitening_ms{0.0};
  double whitening_transform_ms{0.0};
  double query_transform_ms{0.0};
  double cache_ms{0.0};
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

Status AddDocIdsToIndex(const std::shared_ptr<IVFIndex>& ivf,
                        const MatrixRM& x_whitened,
                        const std::vector<DocId>& doc_ids,
                        uint32_t dim,
                        const VersionSet& versions) {
  if (!ivf) {
    return Status::InvalidArgument("AddDocIdsToIndex: null ivf");
  }
  AlignedVector<VectorRecord> records;
  records.reserve(doc_ids.size());
  for (DocId doc_id : doc_ids) {
    if (static_cast<Eigen::Index>(doc_id) >= x_whitened.rows()) {
      return Status::InvalidArgument("AddDocIdsToIndex: doc_id out of range");
    }
    VectorRecord rec;
    rec.doc_id = doc_id;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<Eigen::Index>(doc_id)).transpose();
    records.push_back(std::move(rec));
  }
  return ivf->Add(records);
}


Status AddMatrixRowsToIndex(const std::shared_ptr<IVFIndex>& ivf,
                            const MatrixRM& x_whitened,
                            const std::vector<DocId>& doc_ids,
                            uint32_t dim,
                            const VersionSet& versions) {
  if (!ivf) {
    return Status::InvalidArgument("AddMatrixRowsToIndex: null ivf");
  }
  if (doc_ids.size() != static_cast<size_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddMatrixRowsToIndex: row/doc_id mismatch");
  }
  for (size_t begin = 0; begin < doc_ids.size();) {
    const size_t end = std::min(doc_ids.size(), begin + kAddBlockRows);
    AlignedVector<VectorRecord> records(end - begin);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ii = static_cast<int64_t>(begin); ii < static_cast<int64_t>(end); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      VectorRecord rec;
      rec.doc_id = doc_ids[i];
      rec.dim = dim;
      rec.versions = versions;
      rec.ivf_id = 0;
      rec.x = x_whitened.row(static_cast<Eigen::Index>(i)).transpose();
      records[i - begin] = std::move(rec);
    }
    Status add = ivf->AddBatch(records, end == doc_ids.size());
    if (!add.ok()) {
      return add;
    }
    begin = end;
  }
  return Status::OK();
}

Result<OnlinePQUpdateStats> AddMatrixRowsWithOnlinePQ(
    const std::shared_ptr<IVFIndex>& ivf,
    const MatrixRM& x_whitened,
    const std::vector<DocId>& doc_ids,
    uint32_t dim,
    const VersionSet& versions,
    const OnlinePQUpdateOptions& options) {
  if (!ivf) {
    return Status::InvalidArgument("AddMatrixRowsWithOnlinePQ: null ivf");
  }
  if (doc_ids.size() != static_cast<size_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddMatrixRowsWithOnlinePQ: row/doc_id mismatch");
  }
  OnlinePQUpdateStats empty_stats;
  if (doc_ids.empty()) {
    return empty_stats;
  }
  AlignedVector<VectorRecord> records(doc_ids.size());
  Timer record_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(doc_ids.size()); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    VectorRecord rec;
    rec.doc_id = doc_ids[i];
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<Eigen::Index>(i)).transpose();
    records[i] = std::move(rec);
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

Result<OnlinePQUpdateStats> AddMatrixRowsWithOnlinePQSlidingWindow(
    const std::shared_ptr<IVFIndex>& ivf,
    const MatrixRM& x_whitened,
    const std::vector<DocId>& doc_ids,
    uint32_t dim,
    const VersionSet& versions,
    const std::vector<DocId>& delete_doc_ids,
    const VectorAccessor& cache,
    const OnlinePQUpdateOptions& options) {
  if (!ivf) {
    return Status::InvalidArgument("AddMatrixRowsWithOnlinePQSlidingWindow: null ivf");
  }
  if (doc_ids.size() != static_cast<size_t>(x_whitened.rows())) {
    return Status::InvalidArgument("AddMatrixRowsWithOnlinePQSlidingWindow: row/doc_id mismatch");
  }
  AlignedVector<VectorRecord> records(doc_ids.size());
  Timer record_timer;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(doc_ids.size()); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    VectorRecord rec;
    rec.doc_id = doc_ids[i];
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = x_whitened.row(static_cast<Eigen::Index>(i)).transpose();
    records[i] = std::move(rec);
  }
  const double record_build_ms = record_timer.ElapsedMillis();
  AlignedVector<VectorRecord> delete_records;
  delete_records.reserve(delete_doc_ids.size());
  for (DocId doc_id : delete_doc_ids) {
    auto vec_res = cache.GetVector(doc_id);
    if (!vec_res.ok()) {
      return vec_res.status();
    }
    VectorRecord rec;
    rec.doc_id = doc_id;
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = std::move(vec_res.value());
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

Result<DeltaShard> RebuildDeltaShardFromDocIdsAccessor(const VectorAccessor& cache,
                                                       const std::vector<DocId>& doc_ids,
                                                       const IVFParams& ivf_params,
                                                       VersionId whitening_version,
                                                       uint32_t shard_id,
                                                       uint32_t dim) {
  if (doc_ids.empty()) {
    return Status::InvalidArgument("RebuildDeltaShardFromDocIdsAccessor: empty doc_ids");
  }
  MatrixRM train;
  Status materialize = cache.Materialize(doc_ids, &train);
  if (!materialize.ok()) {
    return materialize;
  }
  auto shard_res = BuildDeltaShard(train, ivf_params, whitening_version, shard_id);
  if (!shard_res.ok()) {
    return shard_res.status();
  }
  DeltaShard shard = shard_res.value();
  Status add = AddMatrixRowsToIndex(shard.ivf, train, doc_ids, dim, shard.versions);
  if (!add.ok()) {
    return add;
  }
  shard.rows = static_cast<uint32_t>(doc_ids.size());
  return shard;
}

Status MaterializeDocIdRows(const MatrixRM& source,
                            const std::vector<DocId>& doc_ids,
                            MatrixRM* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("MaterializeDocIdRows: null output");
  }
  MatrixRM rows(static_cast<Eigen::Index>(doc_ids.size()), source.cols());
  for (size_t i = 0; i < doc_ids.size(); ++i) {
    const DocId doc_id = doc_ids[i];
    if (static_cast<Eigen::Index>(doc_id) >= source.rows()) {
      return Status::InvalidArgument("MaterializeDocIdRows: doc_id out of range");
    }
    rows.row(static_cast<Eigen::Index>(i)) =
        source.row(static_cast<Eigen::Index>(doc_id));
  }
  *out = std::move(rows);
  return Status::OK();
}

struct IndexRebuildResult {
  std::shared_ptr<IVFIndex> ivf;
  VersionSet versions{};
  uint32_t rows{0};
  double materialize_ms{0.0};
  double build_ms{0.0};
  double add_ms{0.0};
  double total_ms{0.0};
};

Result<IndexRebuildResult> RebuildMainFromDocIds(
    const MatrixRM& x_whitened,
    const std::vector<DocId>& doc_ids,
    const IVFParams& ivf_params,
    VersionId whitening_version,
    uint32_t dim,
    const MatrixRM* fixed_routing_centroids) {
  if (doc_ids.empty()) {
    return Status::InvalidArgument("RebuildMainFromDocIds: empty doc_ids");
  }
  Timer total_timer;
  Timer materialize_timer;
  MatrixRM train;
  Status materialize = MaterializeDocIdRows(x_whitened, doc_ids, &train);
  if (!materialize.ok()) {
    return materialize;
  }
  const double materialize_ms = materialize_timer.ElapsedMillis();

  IVFParams params = ivf_params;
  if (fixed_routing_centroids != nullptr) {
    params.use_fixed_routing_centroids = true;
    params.fixed_routing_centroids = *fixed_routing_centroids;
  }

  auto ivf = CreateIVFIndex();
  Timer build_timer;
  auto version_res = ivf->Build(train, doc_ids, params, 0);
  if (!version_res.ok()) {
    return version_res.status();
  }
  const double build_ms = build_timer.ElapsedMillis();
  const VersionSet versions{whitening_version, version_res.value()};

  Timer add_timer;
  Status add = AddDocIdsToIndex(ivf, x_whitened, doc_ids, dim, versions);
  if (!add.ok()) {
    return add;
  }
  const double add_ms = add_timer.ElapsedMillis();

  IndexRebuildResult out;
  out.ivf = std::move(ivf);
  out.versions = versions;
  out.rows = static_cast<uint32_t>(doc_ids.size());
  out.materialize_ms = materialize_ms;
  out.build_ms = build_ms;
  out.add_ms = add_ms;
  out.total_ms = total_timer.ElapsedMillis();
  return out;
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
  return ivf->AddWithOnlinePQSlidingWindowRecords(records, delete_records, options);
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

Result<DeltaShard> RebuildDeltaShardFromDocIds(const MatrixRM& x_whitened,
                                               const std::vector<DocId>& doc_ids,
                                               const IVFParams& ivf_params,
                                               VersionId whitening_version,
                                               uint32_t shard_id,
                                               uint32_t dim) {
  if (doc_ids.empty()) {
    return Status::InvalidArgument("RebuildDeltaShardFromDocIds: empty doc_ids");
  }
  MatrixRM train;
  Status materialize = MaterializeDocIdRows(x_whitened, doc_ids, &train);
  if (!materialize.ok()) {
    return materialize;
  }
  auto shard_res = BuildDeltaShard(train, ivf_params, whitening_version, shard_id);
  if (!shard_res.ok()) {
    return shard_res.status();
  }
  DeltaShard shard = shard_res.value();
  Status add = AddDocIdsToIndex(shard.ivf, x_whitened, doc_ids, dim, shard.versions);
  if (!add.ok()) {
    return add;
  }
  shard.rows = static_cast<uint32_t>(doc_ids.size());
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


Result<std::vector<std::vector<DocId>>> ExactSearchDocIdsBlockwiseAccessor(
    Eigen::Ref<const MatrixRM> queries,
    const VectorAccessor& database,
    const std::vector<DocId>& doc_ids,
    uint32_t topk) {
  if (queries.cols() == 0 || database.dim() == 0) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwiseAccessor: empty matrices");
  }
  if (static_cast<uint32_t>(queries.cols()) != database.dim()) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwiseAccessor: dimension mismatch");
  }
  if (queries.rows() == 0) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwiseAccessor: no queries");
  }
  if (doc_ids.empty()) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwiseAccessor: no searchable docs");
  }
  if (topk == 0) {
    return Status::InvalidArgument("ExactSearchDocIdsBlockwiseAccessor: topk must be positive");
  }

  const size_t limit = std::min<size_t>(topk, doc_ids.size());
  std::vector<std::vector<ExactDocCandidate>> heaps(static_cast<size_t>(queries.rows()));
  for (auto& heap : heaps) {
    heap.reserve(limit);
  }
  const Eigen::VectorXf query_norms = queries.rowwise().squaredNorm();
  const size_t block_rows = ResolveGroundTruthBlockRows(queries.rows(), queries.cols(), doc_ids.size());

  for (size_t block_begin = 0; block_begin < doc_ids.size(); block_begin += block_rows) {
    const size_t block_count = std::min(block_rows, doc_ids.size() - block_begin);
    std::vector<DocId> block_doc_ids(doc_ids.begin() + static_cast<std::ptrdiff_t>(block_begin),
                                     doc_ids.begin() + static_cast<std::ptrdiff_t>(block_begin + block_count));
    MatrixRM block;
    Status materialize = database.Materialize(block_doc_ids, &block);
    if (!materialize.ok()) {
      return materialize;
    }
    Eigen::VectorXf block_norms(static_cast<Eigen::Index>(block_count));
    for (size_t bi = 0; bi < block_count; ++bi) {
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
                     2.0f * dots(static_cast<Eigen::Index>(qi), static_cast<Eigen::Index>(bi));
        if (dist < 0.0f) {
          dist = 0.0f;
        }
        ExactDocCandidate cand{dist, block_doc_ids[bi]};
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
                               const MatrixRM& base_whitened,
                               const Eigen::VectorXf& base_norms) {
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
      if (cand.doc_id >= static_cast<DocId>(base_whitened.rows()) ||
          cand.doc_id >= static_cast<DocId>(base_norms.size())) {
        return Status::InvalidArgument("MergeTopK: doc id out of range for exact rerank");
      }
    }
    const float query_norm = query_whitened.squaredNorm();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && merged.size() > 1)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(merged.size()); ++i) {
      auto& cand = merged[static_cast<size_t>(i)];
      const Eigen::Index doc_idx = static_cast<Eigen::Index>(cand.doc_id);
      const float dot = base_whitened.row(doc_idx).dot(query_whitened);
      cand.rerank_dist = SquaredL2FromNormDot(query_norm, base_norms(doc_idx), dot);
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


Result<SearchResult> MergeTopKAccessor(const std::vector<SearchResult>& partial_results,
                                       uint32_t topk,
                                       bool exact_rerank_enable,
                                       const std::vector<uint32_t>& rerank_candidates_per_route,
                                       Eigen::Ref<const Eigen::VectorXf> query_whitened,
                                       const VectorAccessor& base_vectors) {
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
    const float query_norm = query_whitened.squaredNorm();
    for (auto& cand : merged) {
      auto vec_res = base_vectors.GetVector(cand.doc_id);
      if (!vec_res.ok()) {
        return vec_res.status();
      }
      auto norm_res = base_vectors.GetNorm(cand.doc_id);
      if (!norm_res.ok()) {
        return norm_res.status();
      }
      cand.rerank_dist =
          SquaredL2FromNormDot(query_norm, norm_res.value(), query_whitened.dot(vec_res.value()));
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
                                  const Eigen::VectorXf& base_norms,
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
  if (seen_rows > static_cast<uint32_t>(base_norms.size())) {
    return Status::InvalidArgument("EvaluateState: invalid base_norms");
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
      if (!searchable_seen.insert(doc_id).second) {
        return Status::AlreadyExists("EvaluateState: doc_id appears in multiple live routes");
      }
      searchable_doc_ids.push_back(doc_id);
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

  Timer exact_gt_timer;
  auto gt_res = ExactSearchDocIdsBlockwise(
      queries_whitened, base_whitened, searchable_doc_ids, config.topk);
  if (!gt_res.ok()) {
    return gt_res.status();
  }
  std::cout << "[EXACT_GT] searchable_docs=" << searchable_doc_ids.size()
            << ", queries=" << queries_whitened.rows()
            << ", topk=" << config.topk
            << ", elapsed_ms=" << exact_gt_timer.ElapsedMillis() << std::endl;
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
                                base_whitened,
                                base_norms);
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
      const float qnorm = qv.squaredNorm();
      for (const auto& kv : approx_by_doc) {
        approx_ranked.push_back({kv.second, kv.first});
        const Eigen::Index doc_idx = static_cast<Eigen::Index>(kv.first);
        const float dot = base_whitened.row(doc_idx).dot(qv);
        exact_ranked.push_back(
            {SquaredL2FromNormDot(qnorm, base_norms(doc_idx), dot), kv.first});
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


Result<EvalMetrics> EvaluateStateStreaming(const Config& config,
                                  const VectorAccessor& base_vectors,
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
  if (seen_rows == 0) {
    return Status::InvalidArgument("EvaluateStateStreaming: invalid seen_rows");
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
      if (!searchable_seen.insert(doc_id).second) {
        return Status::AlreadyExists("EvaluateState: doc_id appears in multiple live routes");
      }
      searchable_doc_ids.push_back(doc_id);
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

  Timer exact_gt_timer;
  auto gt_res = ExactSearchDocIdsBlockwiseAccessor(
      queries_whitened, base_vectors, searchable_doc_ids, config.topk);
  if (!gt_res.ok()) {
    return gt_res.status();
  }
  std::cout << "[EXACT_GT] searchable_docs=" << searchable_doc_ids.size()
            << ", queries=" << queries_whitened.rows()
            << ", topk=" << config.topk
            << ", elapsed_ms=" << exact_gt_timer.ElapsedMillis() << std::endl;
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
    auto merged_res = MergeTopKAccessor(route_results[static_cast<size_t>(qi)],
                                        params.topk,
                                        enable_exact_rerank,
                                        route_rerank_candidates,
                                        q,
                                        base_vectors);
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
      const float qnorm = qv.squaredNorm();
      for (const auto& kv : approx_by_doc) {
        approx_ranked.push_back({kv.second, kv.first});
        auto vec_res = base_vectors.GetVector(kv.first);
        if (!vec_res.ok()) {
          return vec_res.status();
        }
        auto norm_res = base_vectors.GetNorm(kv.first);
        if (!norm_res.ok()) {
          return norm_res.status();
        }
        exact_ranked.push_back(
            {SquaredL2FromNormDot(qnorm, norm_res.value(), qv.dot(vec_res.value())), kv.first});
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
    auto resolved = ResolveVecsPath(*dataset_spec, "_base");
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
    auto resolved_query = ResolveVecsPath(*query_spec, "_query");
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
      auto resolved_query = ResolveVecsPath(*dataset_spec, "_query");
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
        auto resolved_query = ResolveVecsPath(parent.string(), "_query");
        if (resolved_query.ok()) {
          query_dataset_path = resolved_query.value();
          std::cout << "[INFO] Using query dataset: " << *query_dataset_path << std::endl;
        }
      }
    }
  }

  uint32_t nx = 0;
  VecsRandomAccessReader base_reader;
  if (!base_dataset_path) {
    std::cerr << "run_eval_large_streaming requires a base .fvecs or .bvecs dataset path."
              << std::endl;
    return 1;
  }
  auto reader_open = base_reader.Open(*base_dataset_path);
  if (!reader_open.ok()) {
    std::cerr << reader_open.status().ToString() << std::endl;
    return 1;
  }
  nx = base_reader.count();
  if (nx == 0) {
    std::cerr << "Base dataset contains no vectors." << std::endl;
    return 1;
  }
  if (config.dim != base_reader.dim()) {
    std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << base_reader.dim() << std::endl;
    config.dim = base_reader.dim();
  }

  uint32_t nq = 0;
  MatrixRM Q;
  if (query_dataset_path) {
    VecsRandomAccessReader query_reader;
    auto query_open = query_reader.Open(*query_dataset_path);
    if (!query_open.ok()) {
      std::cerr << query_open.status().ToString() << std::endl;
      return 1;
    }
    auto load_res = query_reader.ReadRange(0, query_reader.count());
    if (!load_res.ok()) {
      std::cerr << load_res.status().ToString() << std::endl;
      return 1;
    }
    Q = std::move(load_res.value());
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
  if (config.max_main == 0) {
    std::cerr << "run_eval_large_streaming requires config.max_main > 0" << std::endl;
    return 1;
  }
  const uint32_t main_max_rows = std::min<uint32_t>(config.max_main, nx);
  if (main_rows_initial > main_max_rows) {
    std::cerr << "main_index_rows/resolved main_rows (" << main_rows_initial
              << ") must be <= max_main (" << main_max_rows << ")" << std::endl;
    return 1;
  }
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
            << ", max_main=" << main_max_rows
            << ", stream_rows=" << total_stream_rows
            << ", stream_batch_size=" << insert_step
            << ", streaming_mode=" << config.streaming_mode << std::endl;

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
  const std::filesystem::path memory_trace_path = results_dir / "memory_trace.json";
  std::cout << "[INFO] Memory trace will be written to " << memory_trace_path << std::endl;
  const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

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
  MatrixRM Q_whitened;
  WhitenedVectorCache vector_cache;
  vector_cache.Reserve(static_cast<size_t>(main_max_rows) + static_cast<size_t>(delta_train_rows) * 2 + 1024);
  uint32_t main_rows_current = main_rows_initial;
  uint32_t next_insert_idx = stream_start_idx;
  DocRangeQueue main_window_doc_ids;
  std::optional<DeltaShard> active_delta;
  std::optional<DeltaShard> frozen_delta;
  DocRangeQueue sliding_window_doc_ids;
  ann::eval_memory::MemoryTraceRecorder memory_trace;

  auto collect_memory_components = [&]() {
    std::vector<ann::eval_memory::MemoryComponent> components;
    components.push_back({"base.raw_resident", 0});
    components.push_back({"query.raw", ann::eval_memory::MatrixBytes(Q)});
    components.push_back({"query.whitened", ann::eval_memory::MatrixBytes(Q_whitened)});
    components.push_back(
        {"main.window_doc_ids", main_window_doc_ids.EstimatedMemoryBytes()});
    components.push_back(
        {"delta.sliding_window_doc_ids", sliding_window_doc_ids.EstimatedMemoryBytes()});
    std::vector<ann::eval_memory::MemoryComponent> cache_components =
        vector_cache.EstimateMemoryComponents("base.whitened_cache");
    components.insert(components.end(), cache_components.begin(), cache_components.end());
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
                           uint64_t active_rows,
                           uint64_t transient_bytes = 0) {
    memory_trace.Record(stage, active_rows, collect_memory_components(), transient_bytes);
    const WhitenedVectorCache::ReclaimStats reclaim_stats = vector_cache.GetReclaimStats();
    std::cout << "[CACHE_RECLAIM_PROFILE] stage=" << stage
              << " cache_reclaimed_vector_bytes=" << reclaim_stats.reclaimed_vector_bytes
              << " cache_reclaim_calls=" << reclaim_stats.reclaim_calls
              << " cache_dead_prefix_rows=" << reclaim_stats.dead_prefix_rows
              << " cache_reclaim_failures=" << reclaim_stats.reclaim_failures << std::endl;
  };
  record_memory("after_dataset_open", 0);

  double init_materialize_ms = 0.0;
  double init_whitening_ms = 0.0;
  double init_whitening_transform_ms = 0.0;
  double init_main_build_ms = 0.0;
  double init_main_add_ms = 0.0;
  double init_delta_materialize_ms = 0.0;
  double init_delta_build_ms = 0.0;
  double init_delta_add_ms = 0.0;
  double init_delta_wall_ms = 0.0;
  Timer init_total_timer;
  auto init_stats_res = ComputeMeanCovForRange(base_reader, 0, main_rows_initial);
  if (!init_stats_res.ok()) {
    std::cerr << init_stats_res.status().ToString() << std::endl;
    return 1;
  }
  MeanCovStats init_stats = std::move(init_stats_res.value());
  init_materialize_ms += init_stats.read_ms;
  Timer init_fit_timer;
  auto whiten_version_res = whitening->FitFromMeanCov(
      init_stats.mean, init_stats.covariance, init_stats.count);
  if (!whiten_version_res.ok()) {
    std::cerr << whiten_version_res.status().ToString() << std::endl;
    return 1;
  }
  whiten_version = whiten_version_res.value();
  init_whitening_ms = init_fit_timer.ElapsedMillis();
  record_memory("after_initial_whitening_fit",
                main_rows_initial,
                ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim));
  ReleaseMatrix(&init_stats.covariance);
  init_stats.mean.resize(0);

  double init_main_transform_read_ms = 0.0;
  double init_main_transform_ms = 0.0;
  auto main_whiten_res = TransformRangeToWhitened(base_reader,
                                                  0,
                                                  main_rows_initial,
                                                  whitening,
                                                  whiten_version,
                                                  config.use_cosine,
                                                  &init_main_transform_read_ms,
                                                  &init_main_transform_ms);
  if (!main_whiten_res.ok()) {
    std::cerr << main_whiten_res.status().ToString() << std::endl;
    return 1;
  }
  MatrixRM main_train = std::move(main_whiten_res.value());
  init_materialize_ms += init_main_transform_read_ms;
  init_whitening_transform_ms += init_main_transform_ms;
  record_memory("after_initial_base_transform",
                main_rows_initial,
                ann::eval_memory::MatrixBytes(main_train) +
                    2ull * ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim));
  auto qb_res = whitening->TransformBatch(Q, whiten_version);
  if (!qb_res.ok()) {
    std::cerr << qb_res.status().ToString() << std::endl;
    return 1;
  }
  Q_whitened = qb_res.value();
  if (config.use_cosine) {
    NormalizeRowsL2(&Q_whitened);
    std::cout << "[INFO] Cosine mode enabled: normalized whitened live base/query rows"
              << std::endl;
  }

  std::vector<DocId> main_ids(static_cast<size_t>(main_rows_initial));
  std::iota(main_ids.begin(), main_ids.end(), 0);
  Timer init_main_build_timer;
  auto main_version_res = main_ivf->Build(main_train, main_ids, ivf_params, 0);
  if (!main_version_res.ok()) {
    std::cerr << main_version_res.status().ToString() << std::endl;
    return 1;
  }
  init_main_build_ms = init_main_build_timer.ElapsedMillis();
  main_versions = VersionSet{whiten_version, main_version_res.value()};
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
  Timer init_main_add_timer;
  Status add_main = AddMatrixRowsToIndex(main_ivf, main_train, main_ids, config.dim, main_versions);
  if (!add_main.ok()) {
    std::cerr << add_main.ToString() << std::endl;
    return 1;
  }
  init_main_add_ms = init_main_add_timer.ElapsedMillis();
  Status cache_main = vector_cache.AppendBatchOwned(std::move(main_ids), std::move(main_train));
  if (!cache_main.ok()) {
    std::cerr << cache_main.ToString() << std::endl;
    return 1;
  }
  record_memory("after_initial_main_index", main_rows_initial);
  const double init_total_wall_ms = init_total_timer.ElapsedMillis();
  main_window_doc_ids.AppendRange(0, main_rows_initial);
  uint64_t evicted_main_rows_total = 0;
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
    Timer init_delta_wall_timer;
    double delta_transform_read_ms = 0.0;
    double delta_transform_ms = 0.0;
    auto delta_whiten_res = TransformRangeToWhitened(base_reader,
                                                     main_rows_initial,
                                                     delta_train_rows,
                                                     whitening,
                                                     whiten_version,
                                                     config.use_cosine,
                                                     &delta_transform_read_ms,
                                                     &delta_transform_ms);
    if (!delta_whiten_res.ok()) {
      std::cerr << delta_whiten_res.status().ToString() << std::endl;
      return 1;
    }
    MatrixRM delta_train = std::move(delta_whiten_res.value());
    init_delta_materialize_ms = delta_transform_read_ms;
    init_whitening_transform_ms += delta_transform_ms;
    std::vector<DocId> delta_seed_ids;
    delta_seed_ids.reserve(delta_train_rows);
    for (uint32_t i = main_rows_initial; i < stream_start_idx; ++i) {
      delta_seed_ids.push_back(i);
    }
    IVFParams delta_params = ivf_params;
    delta_params.nlist = std::max(1u, delta_ivf_nlist);
    delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;
    Timer init_delta_build_timer;
    auto active_res = BuildDeltaShard(delta_train, delta_params, whiten_version, 1);
    if (!active_res.ok()) {
      std::cerr << active_res.status().ToString() << std::endl;
      return 1;
    }
    init_delta_build_ms = init_delta_build_timer.ElapsedMillis();
    active_delta = active_res.value();
    Timer init_delta_add_timer;
    const Status add_delta_seed = AddMatrixRowsToIndex(active_delta->ivf,
                                                       delta_train,
                                                       delta_seed_ids,
                                                       config.dim,
                                                       active_delta->versions);
    if (!add_delta_seed.ok()) {
      std::cerr << add_delta_seed.ToString() << std::endl;
      return 1;
    }
    init_delta_add_ms = init_delta_add_timer.ElapsedMillis();
    active_delta->rows = delta_train_rows;
    Status cache_delta =
        vector_cache.AppendBatchOwned(std::move(delta_seed_ids), std::move(delta_train));
    if (!cache_delta.ok()) {
      std::cerr << cache_delta.ToString() << std::endl;
      return 1;
    }
    init_delta_wall_ms = init_delta_wall_timer.ElapsedMillis();
    record_memory("after_initial_delta_seed", stream_start_idx);
  }

  auto ActivatePendingDeltaFromSubsequentWindow =
      [&](uint32_t end_row) -> Result<DeltaActivationMetrics> {
    DeltaActivationMetrics metrics;
    if (!pending_active_train || active_delta.has_value()) {
      return metrics;
    }
    if (delta_train_rows == 0) {
      return Status::InvalidArgument(
          "ActivatePendingDeltaFromSubsequentWindow: delta_train_rows is 0");
    }
    if (end_row <= pending_active_train_begin) {
      return metrics;
    }
    const uint32_t available = end_row - pending_active_train_begin;
    if (available < delta_train_rows && end_row < nx) {
      return metrics;
    }
    const uint32_t train_rows = std::min<uint32_t>(delta_train_rows, available);
    if (train_rows == 0) {
      return metrics;
    }
    Timer activation_wall_timer;
    const uint32_t train_begin = pending_active_train_begin;
    double transform_read_ms = 0.0;
    double transform_ms = 0.0;
    auto delta_whiten_res = TransformRangeToWhitened(base_reader,
                                                     train_begin,
                                                     train_rows,
                                                     whitening,
                                                     whiten_version,
                                                     config.use_cosine,
                                                     &transform_read_ms,
                                                     &transform_ms);
    if (!delta_whiten_res.ok()) {
      return delta_whiten_res.status();
    }
    MatrixRM delta_train = std::move(delta_whiten_res.value());
    metrics.materialize_ms = transform_read_ms;
    metrics.whitening_ms = transform_ms;
    std::vector<DocId> train_doc_ids;
    train_doc_ids.reserve(train_rows);
    for (uint32_t i = train_begin; i < train_begin + train_rows; ++i) {
      train_doc_ids.push_back(i);
    }
    IVFParams delta_params = ivf_params;
    delta_params.nlist = std::max(1u, delta_ivf_nlist);
    delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;
    Timer build_timer;
    auto active_res =
        BuildDeltaShard(delta_train, delta_params, whiten_version, next_delta_shard_id++);
    if (!active_res.ok()) {
      return active_res.status();
    }
    metrics.build_ms = build_timer.ElapsedMillis();
    DeltaShard shard = active_res.value();
    Timer insert_timer;
    const Status add_delta_seed = AddMatrixRowsToIndex(shard.ivf,
                                                       delta_train,
                                                       train_doc_ids,
                                                       config.dim,
                                                       shard.versions);
    if (!add_delta_seed.ok()) {
      return add_delta_seed;
    }
    metrics.insert_ms = insert_timer.ElapsedMillis();
    shard.rows = train_rows;
    Timer cache_timer;
    Status cache_delta =
        vector_cache.AppendBatchOwned(std::move(train_doc_ids), std::move(delta_train));
    if (!cache_delta.ok()) {
      return cache_delta;
    }
    metrics.cache_ms = cache_timer.ElapsedMillis();
    active_delta = std::move(shard);
    pending_active_train = false;
    pending_active_train_begin = 0;
    sliding_window_doc_ids.clear();
    sliding_window_doc_ids.AppendRange(train_begin, train_begin + train_rows);
    record_memory("activate_delta", train_begin + train_rows);
    std::cout << "[MERGE] activate_delta shard=" << active_delta->shard_id
              << ", warmup_rows=" << train_rows << std::endl;
    metrics.wall_ms = activation_wall_timer.ElapsedMillis();
    return metrics;
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
    record_memory("freeze_delta", end_row);
    return Status::OK();
  };
  const double init_rebuild_ms = init_whitening_ms + init_whitening_transform_ms +
                                 init_main_build_ms + init_main_add_ms;
  std::cout << "[INIT BUILD] prebuilt=false"
            << ", materialize_ms=" << init_materialize_ms
            << ", whitening_ms=" << init_whitening_ms
            << ", whitening_transform_ms=" << init_whitening_transform_ms
            << ", main_build_ms=" << init_main_build_ms
            << ", main_add_ms=" << init_main_add_ms
            << ", aligned_ms=" << init_rebuild_ms
            << ", wall_ms=" << init_total_wall_ms
            << ", delta_materialize_ms=" << init_delta_materialize_ms
            << ", delta_build_ms=" << init_delta_build_ms
            << ", delta_add_ms=" << init_delta_add_ms
            << ", delta_wall_ms=" << init_delta_wall_ms << std::endl;
  double rebuild_ms_total = init_rebuild_ms;
  double total_update_ms = 0.0;
  double total_update_wall_ms = 0.0;
  double total_update_materialize_ms = 0.0;
  double total_update_whitening_ms = 0.0;
  double total_update_insert_ms = 0.0;
  double total_update_record_build_ms = 0.0;
  double total_update_insert_encode_ms = 0.0;
  double total_update_insert_entry_ms = 0.0;
  double total_update_insert_commit_ms = 0.0;
  double total_update_onlinepq_maintenance_ms = 0.0;
  double total_update_delete_ms = 0.0;
  double total_update_onlinepq_stats_ms = 0.0;
  double total_update_codebook_update_ms = 0.0;
  double total_update_reencode_ms = 0.0;
  double total_update_replace_ms = 0.0;
  double total_activation_materialize_ms = 0.0;
  double total_activation_whitening_ms = 0.0;
  double total_activation_cache_ms = 0.0;
  double total_activation_build_ms = 0.0;
  double total_activation_insert_ms = 0.0;
  double total_activation_wall_ms = 0.0;
  double pending_update_ms = 0.0;
  double pending_update_wall_ms = 0.0;
  double pending_update_materialize_ms = 0.0;
  double pending_update_whitening_ms = 0.0;
  double pending_update_insert_ms = 0.0;
  double pending_update_record_build_ms = 0.0;
  double pending_update_insert_encode_ms = 0.0;
  double pending_update_insert_entry_ms = 0.0;
  double pending_update_insert_commit_ms = 0.0;
  double pending_update_onlinepq_maintenance_ms = 0.0;
  double pending_update_delete_ms = 0.0;
  double pending_update_onlinepq_stats_ms = 0.0;
  double pending_update_codebook_update_ms = 0.0;
  double pending_update_reencode_ms = 0.0;
  double pending_update_replace_ms = 0.0;
  double total_merge_compute_ms = 0.0;
  double total_merge_wall_ms = 0.0;
  double pending_merge_compute_ms = 0.0;
  double pending_merge_wall_ms = 0.0;
  double total_global_rebuild_ms = 0.0;
  double total_global_rebuild_wall_ms = 0.0;
  double pending_global_rebuild_ms = 0.0;
  double pending_global_rebuild_wall_ms = 0.0;

  auto account_delta_activation = [&](const DeltaActivationMetrics& activation) {
    if (activation.wall_ms <= 0.0) {
      return;
    }
    total_update_ms += activation.insert_ms;
    total_update_insert_ms += activation.insert_ms;
    total_update_wall_ms += activation.wall_ms;
    pending_update_ms += activation.insert_ms;
    pending_update_insert_ms += activation.insert_ms;
    pending_update_wall_ms += activation.wall_ms;
    total_activation_materialize_ms += activation.materialize_ms;
    total_activation_whitening_ms += activation.whitening_ms;
    total_activation_cache_ms += activation.cache_ms;
    total_activation_build_ms += activation.build_ms;
    total_activation_insert_ms += activation.insert_ms;
    total_activation_wall_ms += activation.wall_ms;
  };

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
    sliding_window_doc_ids.AppendRange(main_rows_initial, stream_start_idx);
  }

  OnlinePQRollup online_pq_rollup;
  OnlinePQUpdateStats last_online_pq_stats;
  std::vector<GlobalRebuildEventRecord> global_rebuild_events;
  uint32_t global_rebuild_count = 0;
  uint32_t last_global_rebuild_rows = 0;
  uint64_t progress_ts = 0;
  std::ofstream progress_out;
  auto write_progress_jsonl = [&](const std::string& line) {
    if (!progress_out) {
      return;
    }
    progress_out << line << '\n';
    progress_out.flush();
  };
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
    if (seen_rows >= last_global_rebuild_rows) {
      decision.main_rows_since_last_rebuild = seen_rows - last_global_rebuild_rows;
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

    std::vector<DocId> main_ids = main_window_doc_ids.Materialize();
    if (main_ids.empty()) {
      return Status::InvalidArgument("RunGlobalRebuild: empty main window");
    }

    std::vector<DocId> active_doc_ids;
    std::vector<DocId> frozen_doc_ids;
    if (active_delta.has_value() && active_delta->rows > 0 && active_delta->ivf) {
      auto ids_res = active_delta->ivf->SnapshotDocIds(active_delta->versions);
      if (!ids_res.ok()) {
        return ids_res.status();
      }
      active_doc_ids = std::move(ids_res.value());
      std::sort(active_doc_ids.begin(), active_doc_ids.end());
    }
    if (frozen_delta.has_value() && frozen_delta->rows > 0 && frozen_delta->ivf) {
      auto ids_res = frozen_delta->ivf->SnapshotDocIds(frozen_delta->versions);
      if (!ids_res.ok()) {
        return ids_res.status();
      }
      frozen_doc_ids = std::move(ids_res.value());
      std::sort(frozen_doc_ids.begin(), frozen_doc_ids.end());
    }

    std::vector<DocId> whitening_fit_ids = main_ids;
    whitening_fit_ids.insert(whitening_fit_ids.end(), active_doc_ids.begin(), active_doc_ids.end());
    whitening_fit_ids.insert(whitening_fit_ids.end(), frozen_doc_ids.begin(), frozen_doc_ids.end());
    std::sort(whitening_fit_ids.begin(), whitening_fit_ids.end());
    whitening_fit_ids.erase(std::unique(whitening_fit_ids.begin(), whitening_fit_ids.end()),
                            whitening_fit_ids.end());

    auto fit_stats_res = ComputeMeanCovForDocIds(base_reader, whitening_fit_ids);
    if (!fit_stats_res.ok()) {
      return fit_stats_res.status();
    }
    MeanCovStats fit_stats = std::move(fit_stats_res.value());
    const double fit_materialize_ms = fit_stats.read_ms;

    Timer fit_timer;
    auto new_whiten_res =
        whitening->FitFromMeanCov(fit_stats.mean, fit_stats.covariance, fit_stats.count);
    if (!new_whiten_res.ok()) {
      return new_whiten_res.status();
    }
    const VersionId new_whiten_version = new_whiten_res.value();
    const double whitening_ms = fit_timer.ElapsedMillis();
    ReleaseMatrix(&fit_stats.covariance);
    fit_stats.mean.resize(0);
    record_memory("global_rebuild_after_fit",
                  seen_rows,
                  ann::eval_memory::MatrixActiveBytes(kAddBlockRows, config.dim));

    Timer query_transform_timer;
    auto qb_res = whitening->TransformBatch(Q, new_whiten_version);
    if (!qb_res.ok()) {
      return qb_res.status();
    }
    MatrixRM new_q_whitened = qb_res.value();
    if (config.use_cosine) {
      NormalizeRowsL2(&new_q_whitened);
    }
    const double query_transform_ms = query_transform_timer.ElapsedMillis();

    WhitenedVectorCache rebuilt_cache;
    rebuilt_cache.Reserve(whitening_fit_ids.size() + 1024);
    rebuilt_cache.Clear(config.dim);
    double transform_materialize_ms = 0.0;
    double whitening_transform_ms = 0.0;
    double cache_ms = 0.0;
    auto transform_doc_ids = [&](const std::vector<DocId>& doc_ids) -> Result<MatrixRM> {
      return TransformDocIdsToWhitened(base_reader,
                                       doc_ids,
                                       whitening,
                                       new_whiten_version,
                                       config.use_cosine,
                                       &transform_materialize_ms,
                                       &whitening_transform_ms);
    };

    Timer main_materialize_timer;
    auto main_train_res = transform_doc_ids(main_ids);
    if (!main_train_res.ok()) {
      return main_train_res.status();
    }
    MatrixRM main_train = std::move(main_train_res.value());
    const double main_materialize_ms = main_materialize_timer.ElapsedMillis();
    const size_t main_id_count = main_ids.size();

    auto rebuilt_main_ivf = CreateIVFIndex();
    Timer main_build_timer;
    auto main_version_res = rebuilt_main_ivf->Build(main_train, main_ids, ivf_params, 0);
    if (!main_version_res.ok()) {
      return main_version_res.status();
    }
    const double main_build_ms = main_build_timer.ElapsedMillis();
    const VersionSet rebuilt_main_versions{new_whiten_version, main_version_res.value()};
    Timer main_add_timer;
    Status main_add = AddMatrixRowsToIndex(
        rebuilt_main_ivf, main_train, main_ids, config.dim, rebuilt_main_versions);
    if (!main_add.ok()) {
      return main_add;
    }
    const double main_add_ms = main_add_timer.ElapsedMillis();
    Timer main_cache_timer;
    Status main_cache =
        rebuilt_cache.AppendBatchOwned(std::move(main_ids), std::move(main_train));
    if (!main_cache.ok()) {
      return main_cache;
    }
    cache_ms += main_cache_timer.ElapsedMillis();
    IndexRebuildResult main_rebuild;
    main_rebuild.ivf = std::move(rebuilt_main_ivf);
    main_rebuild.versions = rebuilt_main_versions;
    main_rebuild.rows = static_cast<uint32_t>(main_id_count);
    main_rebuild.materialize_ms = main_materialize_ms;
    main_rebuild.build_ms = main_build_ms;
    main_rebuild.add_ms = main_add_ms;
    main_rebuild.total_ms = main_materialize_ms + main_build_ms + main_add_ms;

    IVFParams delta_params = ivf_params;
    delta_params.nlist = std::max(1u, delta_ivf_nlist);
    delta_params.kmeans_iterations = kDeltaKMeansIterationsDefault;

    std::optional<DeltaShard> rebuilt_active_delta;
    std::optional<DeltaShard> rebuilt_frozen_delta;
    DocRangeQueue rebuilt_sliding_window_doc_ids;
    bool rebuilt_pending_active_train = pending_active_train && active_doc_ids.empty();
    uint32_t rebuilt_pending_active_train_begin = pending_active_train_begin;
    double delta_seed_ms = 0.0;
    if (!active_doc_ids.empty()) {
      Timer delta_timer;
      auto active_train_res = transform_doc_ids(active_doc_ids);
      if (!active_train_res.ok()) {
        return active_train_res.status();
      }
      MatrixRM active_train = std::move(active_train_res.value());
      const uint32_t active_rows = static_cast<uint32_t>(active_doc_ids.size());
      auto active_res = BuildDeltaShard(active_train,
                                        delta_params,
                                        new_whiten_version,
                                        next_delta_shard_id++);
      if (!active_res.ok()) {
        return active_res.status();
      }
      rebuilt_active_delta = active_res.value();
      Status active_add = AddMatrixRowsToIndex(rebuilt_active_delta->ivf,
                                               active_train,
                                               active_doc_ids,
                                               config.dim,
                                               rebuilt_active_delta->versions);
      if (!active_add.ok()) {
        return active_add;
      }
      rebuilt_active_delta->rows = active_rows;
      for (DocId doc_id : active_doc_ids) {
        rebuilt_sliding_window_doc_ids.Append(doc_id);
      }
      Timer cache_timer;
      Status cache_status =
          rebuilt_cache.AppendBatchOwned(std::move(active_doc_ids), std::move(active_train));
      if (!cache_status.ok()) {
        return cache_status;
      }
      cache_ms += cache_timer.ElapsedMillis();
      rebuilt_pending_active_train = false;
      rebuilt_pending_active_train_begin = 0;
      delta_seed_ms += delta_timer.ElapsedMillis();
    }
    if (!frozen_doc_ids.empty()) {
      Timer delta_timer;
      auto frozen_train_res = transform_doc_ids(frozen_doc_ids);
      if (!frozen_train_res.ok()) {
        return frozen_train_res.status();
      }
      MatrixRM frozen_train = std::move(frozen_train_res.value());
      const uint32_t frozen_rows = static_cast<uint32_t>(frozen_doc_ids.size());
      auto frozen_res = BuildDeltaShard(frozen_train,
                                        delta_params,
                                        new_whiten_version,
                                        next_delta_shard_id++);
      if (!frozen_res.ok()) {
        return frozen_res.status();
      }
      rebuilt_frozen_delta = frozen_res.value();
      Status frozen_add = AddMatrixRowsToIndex(rebuilt_frozen_delta->ivf,
                                               frozen_train,
                                               frozen_doc_ids,
                                               config.dim,
                                               rebuilt_frozen_delta->versions);
      if (!frozen_add.ok()) {
        return frozen_add;
      }
      rebuilt_frozen_delta->rows = frozen_rows;
      Timer cache_timer;
      Status cache_status =
          rebuilt_cache.AppendBatchOwned(std::move(frozen_doc_ids), std::move(frozen_train));
      if (!cache_status.ok()) {
        return cache_status;
      }
      cache_ms += cache_timer.ElapsedMillis();
      delta_seed_ms += delta_timer.ElapsedMillis();
    }

    Q_whitened = std::move(new_q_whitened);
    vector_cache = std::move(rebuilt_cache);
    whiten_version = new_whiten_version;
    main_ivf = std::move(main_rebuild.ivf);
    main_versions = main_rebuild.versions;
    main_rows_current = main_rebuild.rows;
    active_delta = std::move(rebuilt_active_delta);
    frozen_delta = std::move(rebuilt_frozen_delta);
    pending_active_train = rebuilt_pending_active_train;
    pending_active_train_begin = rebuilt_pending_active_train_begin;
    sliding_window_doc_ids = std::move(rebuilt_sliding_window_doc_ids);
    last_online_pq_stats = OnlinePQUpdateStats{};
    TrimAllocatorRetainedMemory("global_rebuild_done");
    record_memory("global_rebuild_done", seen_rows);

    const double wall_total_ms = total_timer.ElapsedMillis();
    const double total_ms =
        whitening_ms + whitening_transform_ms + main_build_ms + main_add_ms;
    rebuild_ms_total += total_ms;
    total_global_rebuild_ms += total_ms;
    pending_global_rebuild_ms += total_ms;
    total_global_rebuild_wall_ms += wall_total_ms;
    pending_global_rebuild_wall_ms += wall_total_ms;
    global_rebuild_count++;
    last_global_rebuild_rows = seen_rows;

    GlobalRebuildEventRecord event;
    event.base_rows = seen_rows;
    event.old_main_rows = old_main_rows;
    event.new_main_rows = main_rows_current;
    event.active_seed_rows = active_delta.has_value() ? active_delta->rows : 0;
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
    event.materialize_ms = fit_materialize_ms + transform_materialize_ms;
    event.whitening_ms = whitening_ms;
    event.whitening_transform_ms = whitening_transform_ms;
    event.query_transform_ms = query_transform_ms;
    event.cache_ms = cache_ms;
    event.main_build_ms = main_rebuild.build_ms;
    event.main_add_ms = main_rebuild.add_ms;
    event.delta_seed_ms = delta_seed_ms;
    event.total_ms = total_ms;
    event.wall_total_ms = wall_total_ms;
    event.reason = trigger.reason;
    global_rebuild_events.push_back(event);
    {
      std::ostringstream line;
      line << R"({"type":"global_rebuild")"
           << R"(,"ts":)" << progress_ts
           << R"(,"base_rows":)" << event.base_rows
           << R"(,"reason":")" << JsonEscape(event.reason) << R"(")"
           << R"(,"old_main_rows":)" << event.old_main_rows
           << R"(,"new_main_rows":)" << event.new_main_rows
           << R"(,"active_seed_rows":)" << event.active_seed_rows
           << R"(,"rebuild_count":)" << event.rebuild_count
           << R"(,"max_count":)" << event.max_count
           << R"(,"trigger_main_rows":)" << (event.trigger_main_rows ? "true" : "false")
           << R"(,"trigger_imbalance":)" << (event.trigger_imbalance ? "true" : "false")
           << R"(,"main_imbalance_ratio":)" << event.main_imbalance_ratio
           << R"(,"materialize_ms":)" << event.materialize_ms
           << R"(,"whitening_ms":)" << event.whitening_ms
           << R"(,"whitening_transform_ms":)" << event.whitening_transform_ms
           << R"(,"query_transform_ms":)" << event.query_transform_ms
           << R"(,"cache_ms":)" << event.cache_ms
           << R"(,"main_build_ms":)" << event.main_build_ms
           << R"(,"main_add_ms":)" << event.main_add_ms
           << R"(,"delta_seed_ms":)" << event.delta_seed_ms
           << R"(,"total_ms":)" << event.total_ms
           << R"(,"wall_total_ms":)" << event.wall_total_ms
           << R"(,"rss_bytes":)" << ReadProcStatusBytes("VmRSS:")
           << R"(,"peak_rss_bytes":)" << ReadProcStatusBytes("VmHWM:")
           << "}";
      write_progress_jsonl(line.str());
    }

    std::cout << "[GLOBAL REBUILD] done: base_rows=" << seen_rows
              << ", reason=" << trigger.reason
              << ", count=" << global_rebuild_count << "/" << config.global_rebuild_max_count
              << ", inserted_since_last=" << trigger.main_rows_since_last_rebuild
              << ", imbalance=" << trigger.main_imbalance_ratio
              << ", old_main_rows=" << old_main_rows
              << ", new_main_rows=" << main_rows_current
              << ", active_rows=" << (active_delta.has_value() ? active_delta->rows : 0)
              << ", frozen_rows=" << (frozen_delta.has_value() ? frozen_delta->rows : 0)
              << ", whitening_fit_materialize_ms=" << fit_materialize_ms
              << ", whitening_ms=" << whitening_ms
              << ", whitening_transform_ms=" << whitening_transform_ms
              << ", main_materialize_ms=" << main_rebuild.materialize_ms
              << ", main_build_ms=" << main_rebuild.build_ms
              << ", main_add_ms=" << main_rebuild.add_ms
              << ", delta_seed_ms=" << delta_seed_ms
              << ", total_ms=" << total_ms << std::endl;
    return Status::OK();
  };

  progress_ts = static_cast<uint64_t>(ts);
  const std::filesystem::path progress_path = results_dir / "online_eval.progress.jsonl";
  progress_out.open(progress_path, std::ios::app);
  if (!progress_out) {
    std::cerr << "Failed to open progress jsonl at " << progress_path << std::endl;
  } else {
    std::cout << "Streaming progress jsonl to " << progress_path << std::endl;
  }

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
    auto res = EvaluateStateStreaming(config,
                                      vector_cache,
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
    std::ostringstream line;
    line << R"({"type":"eval")"
         << R"(,"ts":)" << progress_ts
         << R"(,"eval_id":)" << eval_id
         << R"(,"stage":")" << JsonEscape(stage) << R"(")"
         << R"(,"base_rows":)" << active_rows
         << R"(,"new_begin":)" << new_begin
         << R"(,"new_end":)" << new_end
         << R"(,"main_rows":)" << main_rows_current
         << R"(,"active_delta_rows":)"
         << (active_delta.has_value() ? active_delta->rows : 0)
         << R"(,"frozen_delta_rows":)"
         << (frozen_delta.has_value() ? frozen_delta->rows : 0)
         << R"(,"recall_at_)" << config.topk << R"(":)" << m.recall
         << R"(,"recall_new":)" << m.recall_new
         << R"(,"recall_old":)" << m.recall_old
         << R"(,"gt_new_ratio":)" << m.gt_new_ratio
         << R"(,"latency_ms":)" << m.avg_query_ms
         << R"(,"avg_search_ms":)" << m.avg_search_ms
         << R"(,"query_qps":)" << m.query_qps
         << R"(,"scanned_avg":)" << m.scanned_avg
         << R"(,"query_eval_ms":)" << m.query_eval_ms
         << R"(,"rss_bytes":)" << ReadProcStatusBytes("VmRSS:")
         << R"(,"peak_rss_bytes":)" << ReadProcStatusBytes("VmHWM:")
         << "}";
    write_progress_jsonl(line.str());
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
    snap.update_wall_ms = pending_update_wall_ms;
    snap.update_materialize_ms = pending_update_materialize_ms;
    snap.update_whitening_ms = pending_update_whitening_ms;
    snap.update_insert_ms = pending_update_insert_ms;
    snap.update_record_build_ms = pending_update_record_build_ms;
    snap.update_insert_encode_ms = pending_update_insert_encode_ms;
    snap.update_insert_entry_ms = pending_update_insert_entry_ms;
    snap.update_insert_commit_ms = pending_update_insert_commit_ms;
    snap.update_onlinepq_maintenance_ms = pending_update_onlinepq_maintenance_ms;
    snap.update_delete_ms = pending_update_delete_ms;
    snap.update_onlinepq_stats_ms = pending_update_onlinepq_stats_ms;
    snap.update_codebook_update_ms = pending_update_codebook_update_ms;
    snap.update_reencode_ms = pending_update_reencode_ms;
    snap.update_replace_ms = pending_update_replace_ms;
    snap.query_eval_ms = metrics.query_eval_ms;
    snap.merge_compute_ms = pending_merge_compute_ms;
    snap.merge_wall_ms = pending_merge_wall_ms;
    snap.global_rebuild_ms = pending_global_rebuild_ms;
    snap.global_rebuild_wall_ms = pending_global_rebuild_wall_ms;
    snap.snapshot_total_ms =
        snap.update_ms + snap.update_replace_ms + snap.merge_compute_ms + snap.global_rebuild_ms;
    snap.snapshot_wall_total_ms =
        snap.update_wall_ms + snap.merge_wall_ms + snap.global_rebuild_wall_ms;
    snap.update_throughput_vecps =
        (snap.snapshot_rows > 0 && snap.snapshot_total_ms > 0.0)
            ? (static_cast<double>(snap.snapshot_rows) / (snap.snapshot_total_ms / 1000.0))
            : 0.0;
    snap.update_wall_throughput_vecps =
        (snap.snapshot_rows > 0 && snap.snapshot_wall_total_ms > 0.0)
            ? (static_cast<double>(snap.snapshot_rows) / (snap.snapshot_wall_total_ms / 1000.0))
            : 0.0;
    const uint32_t streamed_rows =
        active_rows >= stream_start_idx ? active_rows - stream_start_idx : 0;
    const double cumulative_maintenance_ms =
        total_update_ms + total_update_replace_ms + total_merge_compute_ms +
        total_global_rebuild_ms;
    snap.amortized_update_throughput_vecps =
        (streamed_rows > 0 && cumulative_maintenance_ms > 0.0)
            ? (1000.0 * static_cast<double>(streamed_rows) / cumulative_maintenance_ms)
            : 0.0;
    const double cumulative_wall_ms =
        total_update_wall_ms + total_merge_wall_ms + total_global_rebuild_wall_ms;
    snap.amortized_update_wall_throughput_vecps =
        (streamed_rows > 0 && cumulative_wall_ms > 0.0)
            ? (1000.0 * static_cast<double>(streamed_rows) / cumulative_wall_ms)
            : 0.0;
    snap.rss_bytes = ReadProcStatusBytes("VmRSS:");
    snap.peak_rss_bytes = ReadProcStatusBytes("VmHWM:");
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
    std::ostringstream line;
    line << R"({"type":"snapshot")"
         << R"(,"ts":)" << progress_ts
         << R"(,"base_rows":)" << snap.base_rows
         << R"(,"snapshot_rows":)" << snap.snapshot_rows
         << R"(,"main_rows":)" << snap.main_rows
         << R"(,"active_delta_rows":)" << snap.active_delta_docs
         << R"(,"frozen_delta_rows":)" << snap.frozen_delta_docs
         << R"(,"recall_at_)" << config.topk << R"(":)" << snap.recall
         << R"(,"recall_new":)" << snap.recall_new
         << R"(,"recall_old":)" << snap.recall_old
         << R"(,"latency_ms":)" << snap.latency_ms
         << R"(,"query_qps":)" << snap.query_qps
         << R"(,"update_ms":)" << snap.update_ms
         << R"(,"update_wall_ms":)" << snap.update_wall_ms
         << R"(,"update_materialize_ms":)" << snap.update_materialize_ms
         << R"(,"update_whitening_ms":)" << snap.update_whitening_ms
         << R"(,"update_insert_ms":)" << snap.update_insert_ms
         << R"(,"update_replace_ms":)" << snap.update_replace_ms
         << R"(,"merge_compute_ms":)" << snap.merge_compute_ms
         << R"(,"merge_wall_ms":)" << snap.merge_wall_ms
         << R"(,"global_rebuild_ms":)" << snap.global_rebuild_ms
         << R"(,"global_rebuild_wall_ms":)" << snap.global_rebuild_wall_ms
         << R"(,"snapshot_total_ms":)" << snap.snapshot_total_ms
         << R"(,"snapshot_wall_total_ms":)" << snap.snapshot_wall_total_ms
         << R"(,"update_throughput_vecps":)" << snap.update_throughput_vecps
         << R"(,"update_wall_throughput_vecps":)" << snap.update_wall_throughput_vecps
         << R"(,"rss_bytes":)" << snap.rss_bytes
         << R"(,"peak_rss_bytes":)" << snap.peak_rss_bytes
         << "}";
    write_progress_jsonl(line.str());
    snapshots.push_back(std::move(snap));
    record_memory("snapshot", active_rows);
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
      account_delta_activation(activate_status.value());
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

      last_online_pq_stats = OnlinePQUpdateStats{};
      double step_materialize_ms = 0.0;
      double step_whitening_ms = 0.0;
      Timer update_wall_timer;
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
          delete_doc_ids = sliding_window_doc_ids.MaterializeFront(need_delete);
        }

        std::vector<DocId> insert_doc_ids;
        insert_doc_ids.reserve(chunk);
        for (uint32_t i = begin; i < end; ++i) {
          insert_doc_ids.push_back(i);
        }
        double transform_read_ms = 0.0;
        double transform_ms = 0.0;
        auto whiten_batch_res = TransformRangeToWhitened(base_reader,
                                                         begin,
                                                         chunk,
                                                         whitening,
                                                         whiten_version,
                                                         config.use_cosine,
                                                         &transform_read_ms,
                                                         &transform_ms);
        if (!whiten_batch_res.ok()) {
          std::cerr << whiten_batch_res.status().ToString() << std::endl;
          return 1;
        }
        MatrixRM insert_whitened = std::move(whiten_batch_res.value());
        step_materialize_ms = transform_read_ms;
        step_whitening_ms = transform_ms;

        Result<OnlinePQUpdateStats> add_res = OnlinePQUpdateStats{};
        if (use_sliding_window) {
          add_res = AddMatrixRowsWithOnlinePQSlidingWindow(active_delta->ivf,
                                                           insert_whitened,
                                                           insert_doc_ids,
                                                           config.dim,
                                                           active_delta->versions,
                                                           delete_doc_ids,
                                                           vector_cache,
                                                           online_pq_options);
        } else {
          add_res = AddMatrixRowsWithOnlinePQ(active_delta->ivf,
                                              insert_whitened,
                                              insert_doc_ids,
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
        Status cache_insert =
            vector_cache.AppendBatchOwned(std::move(insert_doc_ids), std::move(insert_whitened));
        if (!cache_insert.ok()) {
          std::cerr << cache_insert.ToString() << std::endl;
          return 1;
        }
        if (use_sliding_window) {
          const uint32_t deleted_rows = static_cast<uint32_t>(delete_doc_ids.size());
          if (!delete_doc_ids.empty()) {
            Status cache_remove = vector_cache.RemoveDocIds(delete_doc_ids);
            if (!cache_remove.ok()) {
              std::cerr << cache_remove.ToString() << std::endl;
              return 1;
            }
          }
          sliding_window_doc_ids.PopFront(deleted_rows);
          sliding_window_doc_ids.AppendRange(begin, end);
          active_delta->rows = active_delta->rows + chunk - deleted_rows;
        } else {
          active_delta->rows += chunk;
        }
      }
      const double step_insert_ms = last_online_pq_stats.insert_ms;
      const double step_update_ms = step_whitening_ms + step_insert_ms;
      const double step_update_wall_ms = update_wall_timer.ElapsedMillis();

      next_insert_idx = end;
      last_insert_begin = begin;
      last_insert_end = end;
      inserted_rows = next_insert_idx - stream_start_idx;
      total_update_ms += step_update_ms;
      total_update_wall_ms += step_update_wall_ms;
      total_update_materialize_ms += step_materialize_ms;
      total_update_whitening_ms += step_whitening_ms;
      total_update_insert_ms += step_insert_ms;
      total_update_record_build_ms += last_online_pq_stats.record_build_ms;
      total_update_insert_encode_ms += last_online_pq_stats.insert_encode_ms;
      total_update_insert_entry_ms += last_online_pq_stats.insert_entry_ms;
      total_update_insert_commit_ms += last_online_pq_stats.insert_commit_ms;
      total_update_onlinepq_maintenance_ms += last_online_pq_stats.maintenance_ms;
      total_update_delete_ms += last_online_pq_stats.delete_ms;
      total_update_onlinepq_stats_ms += last_online_pq_stats.onlinepq_stats_ms;
      total_update_codebook_update_ms += last_online_pq_stats.codebook_update_ms;
      total_update_reencode_ms += last_online_pq_stats.reencode_ms;
      pending_update_ms += step_update_ms;
      pending_update_wall_ms += step_update_wall_ms;
      pending_update_materialize_ms += step_materialize_ms;
      pending_update_whitening_ms += step_whitening_ms;
      pending_update_insert_ms += step_insert_ms;
      pending_update_record_build_ms += last_online_pq_stats.record_build_ms;
      pending_update_insert_encode_ms += last_online_pq_stats.insert_encode_ms;
      pending_update_insert_entry_ms += last_online_pq_stats.insert_entry_ms;
      pending_update_insert_commit_ms += last_online_pq_stats.insert_commit_ms;
      pending_update_onlinepq_maintenance_ms += last_online_pq_stats.maintenance_ms;
      pending_update_delete_ms += last_online_pq_stats.delete_ms;
      pending_update_onlinepq_stats_ms += last_online_pq_stats.onlinepq_stats_ms;
      pending_update_codebook_update_ms += last_online_pq_stats.codebook_update_ms;
      pending_update_reencode_ms += last_online_pq_stats.reencode_ms;

      activate_status = ActivatePendingDeltaFromSubsequentWindow(next_insert_idx);
      if (!activate_status.ok()) {
        std::cerr << activate_status.status().ToString() << std::endl;
        return 1;
      }
      account_delta_activation(activate_status.value());

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
        minibatch.update_wall_ms = step_update_wall_ms;
        minibatch.update_materialize_ms = step_materialize_ms;
        minibatch.update_whitening_ms = step_whitening_ms;
        minibatch.update_insert_ms = step_insert_ms;
        minibatch.update_record_build_ms = last_online_pq_stats.record_build_ms;
        minibatch.update_insert_encode_ms = last_online_pq_stats.insert_encode_ms;
        minibatch.update_insert_entry_ms = last_online_pq_stats.insert_entry_ms;
        minibatch.update_insert_commit_ms = last_online_pq_stats.insert_commit_ms;
        minibatch.update_onlinepq_maintenance_ms = last_online_pq_stats.maintenance_ms;
        minibatch.update_delete_ms = last_online_pq_stats.delete_ms;
        minibatch.update_onlinepq_stats_ms = last_online_pq_stats.onlinepq_stats_ms;
        minibatch.update_codebook_update_ms = last_online_pq_stats.codebook_update_ms;
        minibatch.update_reencode_ms = last_online_pq_stats.reencode_ms;
        minibatch.update_throughput_vecps =
            (chunk > 0 && step_update_ms > 0.0)
                ? (static_cast<double>(chunk) / (step_update_ms / 1000.0))
                : 0.0;
        minibatch.update_wall_throughput_vecps =
            (chunk > 0 && step_update_wall_ms > 0.0)
                ? (static_cast<double>(chunk) / (step_update_wall_ms / 1000.0))
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
        const uint32_t main_rows_before = main_rows_current;

        auto frozen_doc_ids_res = frozen_delta->ivf->SnapshotDocIds(frozen_delta->versions);
        if (!frozen_doc_ids_res.ok()) {
          std::cerr << frozen_doc_ids_res.status().ToString() << std::endl;
          return 1;
        }
        std::vector<DocId> frozen_doc_ids = std::move(frozen_doc_ids_res.value());
        std::sort(frozen_doc_ids.begin(), frozen_doc_ids.end());
        frozen_doc_ids.erase(std::unique(frozen_doc_ids.begin(), frozen_doc_ids.end()),
                             frozen_doc_ids.end());
        const uint32_t frozen_rows = static_cast<uint32_t>(frozen_doc_ids.size());
        if (frozen_rows > main_max_rows) {
          std::cerr << "[MERGE-LARGE-MODIFIED] frozen_rows exceeds max_main; "
                    << "cannot preserve bounded main with original merge: frozen_rows="
                    << frozen_rows << ", max_main=" << main_max_rows << std::endl;
          return 1;
        }

        uint32_t evicted_rows = 0;
        if (main_rows_before + frozen_rows > main_max_rows) {
          evicted_rows = main_rows_before + frozen_rows - main_max_rows;
        }
        if (evicted_rows > main_window_doc_ids.size()) {
          std::cerr << "[MERGE-LARGE-MODIFIED] eviction exceeds live main window: evict="
                    << evicted_rows << ", live_main_window=" << main_window_doc_ids.size()
                    << std::endl;
          return 1;
        }

        std::vector<DocId> evict_doc_ids =
            main_window_doc_ids.MaterializeFront(evicted_rows);

        Timer pre_replace_timer;
        if (!evict_doc_ids.empty()) {
          Status remove_status = main_ivf->RemoveDocIds(main_versions, evict_doc_ids);
          if (!remove_status.ok()) {
            std::cerr << remove_status.ToString() << std::endl;
            return 1;
          }
          Status cache_remove = vector_cache.RemoveDocIds(evict_doc_ids);
          if (!cache_remove.ok()) {
            std::cerr << cache_remove.ToString() << std::endl;
            return 1;
          }
          main_window_doc_ids.PopFront(evicted_rows);
          evicted_main_rows_total += evicted_rows;
          main_rows_current -= evicted_rows;
        }
        const double pre_replacement_ms = pre_replace_timer.ElapsedMillis();

        auto merge_res = merge_frozen_delta_into_main(main_ivf,
                                                      main_versions,
                                                      frozen_delta->ivf,
                                                      frozen_delta->versions,
                                                      vector_cache,
                                                      merge_options);
        if (!merge_res.ok()) {
          std::cerr << merge_res.status().ToString() << std::endl;
          return 1;
        }

        for (DocId doc_id : frozen_doc_ids) {
          main_window_doc_ids.Append(doc_id);
        }

        const double merge_commit_ms = merge_commit_timer.ElapsedMillis();
        total_update_replace_ms += pre_replacement_ms;
        pending_update_replace_ms += pre_replacement_ms;
        total_merge_compute_ms += merge_res.value().merge_compute_ms;
        pending_merge_compute_ms += merge_res.value().merge_compute_ms;
        total_merge_wall_ms += merge_commit_ms;
        pending_merge_wall_ms += merge_commit_ms;
        rebuild_ms_total += merge_commit_ms;
        main_rows_current = main_rows_current + merge_res.value().frozen_records;
        if (main_rows_current > main_max_rows ||
            main_window_doc_ids.size() != static_cast<size_t>(main_rows_current)) {
          std::cerr << "[MERGE-LARGE-MODIFIED] live main accounting mismatch: main_rows_current="
                    << main_rows_current << ", window_size=" << main_window_doc_ids.size()
                    << ", max_main=" << main_max_rows << std::endl;
          return 1;
        }

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
        merge_event.evicted_main_rows = evicted_rows;
        merge_event.main_rows_before = main_rows_before;
        merge_event.main_rows_after = main_rows_current;
        merge_event.live_main_begin_doc = main_window_doc_ids.empty() ? 0 : main_window_doc_ids.front();
        merge_event.live_main_end_doc = main_window_doc_ids.empty() ? 0 : main_window_doc_ids.back() + 1;
        merge_event.main_materialize_ms = 0.0;
        merge_event.main_build_ms = 0.0;
        merge_event.main_add_ms = 0.0;
        merge_event.pre_replacement_ms = pre_replacement_ms;
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
        merge_event.profiling = merge_res.value().profiling;
        merge_events.push_back(merge_event);
        {
          std::ostringstream line;
          line << R"({"type":"merge")"
               << R"(,"ts":)" << progress_ts
               << R"(,"base_rows":)" << merge_event.base_rows
               << R"(,"frozen_rows":)" << merge_event.frozen_rows
               << R"(,"evicted_main_rows":)" << merge_event.evicted_main_rows
               << R"(,"main_rows_before":)" << merge_event.main_rows_before
               << R"(,"main_rows_after":)" << merge_event.main_rows_after
               << R"(,"live_main_begin_doc":)" << merge_event.live_main_begin_doc
               << R"(,"live_main_end_doc":)" << merge_event.live_main_end_doc
               << R"(,"max_main":)" << main_max_rows
               << R"(,"pre_replacement_ms":)" << merge_event.pre_replacement_ms
               << R"(,"merge_compute_ms":)" << merge_event.merge_compute_ms
               << R"(,"merge_ms":)" << merge_event.merge_ms
               << R"(,"codebook_rebuild_ms":)" << merge_event.codebook_rebuild_ms
               << R"(,"merge_assignment_us":)"
               << merge_event.profiling.merge_delta_to_main_assignment_us
               << R"(,"merge_assignment_distance_us":)"
               << merge_event.profiling.merge_assignment_distance_us
               << R"(,"merge_assignment_top_r_us":)"
               << merge_event.profiling.merge_assignment_top_r_us
               << R"(,"merge_assignment_balance_us":)"
               << merge_event.profiling.merge_assignment_balance_us
               << R"(,"merge_assignment_materialize_us":)"
               << merge_event.profiling.merge_assignment_materialize_us
               << R"(,"merge_top_r_neighbor_us":)"
               << merge_event.profiling.top_r_neighbor_us
               << R"(,"merge_fetch_main_records_us":)"
               << merge_event.profiling.fetch_main_records_us
               << R"(,"merge_repartition_us":)" << merge_event.profiling.repartition_us
               << R"(,"merge_patch_prepare_us":)"
               << merge_event.profiling.patch_prepare_us
               << R"(,"merge_commit_us":)" << merge_event.profiling.commit_us
               << R"(,"prepare_total_us":)" << merge_event.profiling.prepare_total_us
               << R"(,"prepare_fetch_us":)" << merge_event.profiling.prepare_fetch_us
               << R"(,"prepare_distance_us":)" << merge_event.profiling.prepare_distance_us
               << R"(,"prepare_balance_us":)" << merge_event.profiling.prepare_balance_us
               << R"(,"compact_emit_us":)" << merge_event.profiling.compact_emit_us
               << R"(,"compact_patch_records":)"
               << merge_event.profiling.compact_patch_records
               << R"(,"compact_patch_estimated_bytes":)"
               << merge_event.profiling.compact_patch_estimated_bytes
               << R"(,"commit_total_us":)" << merge_event.profiling.commit_total_us
               << R"(,"commit_validation_us":)"
               << merge_event.profiling.commit_validation_us
               << R"(,"commit_reuse_classify_us":)"
               << merge_event.profiling.commit_reuse_classify_us
               << R"(,"commit_materialize_us":)"
               << merge_event.profiling.commit_materialize_us
               << R"(,"commit_materialized_rows":)"
               << merge_event.profiling.commit_materialized_rows
               << R"(,"commit_materialized_bytes":)"
               << merge_event.profiling.commit_materialized_bytes
               << R"(,"commit_materialize_batches":)"
               << merge_event.profiling.commit_materialize_batches
               << R"(,"commit_max_materialize_rows":)"
               << merge_event.profiling.commit_max_materialize_rows
               << R"(,"commit_pq_encode_us":)"
               << merge_event.profiling.commit_pq_encode_us
               << R"(,"pq_codes_reused":)" << merge_event.profiling.pq_codes_reused
               << R"(,"pq_codes_reencoded":)"
               << merge_event.profiling.pq_codes_reencoded
               << R"(,"commit_apply_us":)" << merge_event.profiling.commit_apply_us
               << R"(,"pq_list_flatten_us":)"
               << merge_event.profiling.pq_list_flatten_us
               << R"(,"docmap_rebuild_us":)" << merge_event.profiling.docmap_rebuild_us
               << R"(,"commit_lock_hold_us":)"
               << merge_event.profiling.commit_lock_hold_us
               << R"(,"patched_partitions":)" << merge_event.patched_partitions
               << R"(,"append_partitions":)" << merge_event.append_partitions
               << R"(,"recluster_partitions":)" << merge_event.recluster_partitions
               << R"(,"moved_delta_ratio":)" << merge_event.moved_delta_ratio
               << R"(,"avg_assignment_dist_ratio":)" << merge_event.avg_assignment_dist_ratio
               << R"(,"max_assignment_dist_ratio":)" << merge_event.max_assignment_dist_ratio
               << R"(,"imbalance_before":)" << merge_event.imbalance_before
               << R"(,"imbalance_after":)" << merge_event.imbalance_after
               << R"(,"main_imbalance_after_real":)" << merge_event.main_imbalance_after_real
               << R"(,"trigger_reason":")" << JsonEscape(merge_event.trigger_reason) << R"(")"
               << R"(,"trigger_rows":)" << (merge_event.trigger_rows ? "true" : "false")
               << R"(,"trigger_structure":)" << (merge_event.trigger_structure ? "true" : "false")
               << R"(,"original_assignment_patch":true)"
               << R"(,"pq_codebook_retrained":false)"
               << R"(,"rss_bytes":)" << ReadProcStatusBytes("VmRSS:")
               << R"(,"peak_rss_bytes":)" << ReadProcStatusBytes("VmHWM:")
               << "}";
          write_progress_jsonl(line.str());
        }

        std::cout << "[MERGE-LARGE-MODIFIED] pre_replace+original_merge done: frozen_rows="
                  << merge_res.value().frozen_records
                  << ", evicted_main_rows=" << evicted_rows
                  << ", main_rows_before=" << main_rows_before
                  << ", main_rows_after=" << main_rows_current
                  << ", live_main_doc_range=[" << merge_event.live_main_begin_doc
                  << "," << merge_event.live_main_end_doc << ")"
                  << ", max_main=" << main_max_rows
                  << ", original_assignment_patch=true"
                  << ", pq_codebook_retrained=false"
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
                  << ", pre_replacement_ms=" << pre_replacement_ms
                  << ", merge_compute_ms=" << merge_res.value().merge_compute_ms
                  << ", codebook_rebuild_ms=" << merge_res.value().codebook_rebuild_ms
                  << ", merge_ms=" << merge_commit_ms << std::endl;
        const auto& merge_profile = merge_res.value().profiling;
        std::cout << "[MERGE_PROFILE] effective_nlist=" << merge_profile.effective_nlist
                  << ", frozen_records=" << merge_profile.frozen_records
                  << ", seed_partitions=" << merge_profile.seed_partitions
                  << ", neighborhoods=" << merge_profile.neighborhoods
                  << ", main_records_loaded=" << merge_profile.main_records_loaded
                  << ", pooled_records=" << merge_profile.pooled_records
                  << ", repartitioned_records=" << merge_profile.repartitioned_records
                  << ", patch_records=" << merge_profile.patch_records
                  << ", compact_patch_records=" << merge_profile.compact_patch_records
                  << ", compact_patch_estimated_bytes="
                  << merge_profile.compact_patch_estimated_bytes
                  << ", assignment_distance_evaluations="
                  << merge_profile.assignment_distance_evaluations
                  << ", assignment_workspace_bytes="
                  << merge_profile.assignment_workspace_bytes
                  << ", assignment_chunk_records="
                  << merge_profile.assignment_chunk_records
                  << ", assignment_chunk_count=" << merge_profile.assignment_chunk_count
                  << ", merge_delta_to_main_assignment_us="
                  << merge_profile.merge_delta_to_main_assignment_us
                  << ", merge_assignment_distance_us="
                  << merge_profile.merge_assignment_distance_us
                  << ", merge_assignment_top_r_us="
                  << merge_profile.merge_assignment_top_r_us
                  << ", merge_assignment_balance_us="
                  << merge_profile.merge_assignment_balance_us
                  << ", merge_assignment_materialize_us="
                  << merge_profile.merge_assignment_materialize_us
                  << ", stats_us=" << merge_profile.stats_us
                  << ", scoring_us=" << merge_profile.scoring_us
                  << ", top_r_neighbor_us=" << merge_profile.top_r_neighbor_us
                  << ", fetch_main_records_us=" << merge_profile.fetch_main_records_us
                  << ", repartition_us=" << merge_profile.repartition_us
                  << ", patch_prepare_us=" << merge_profile.patch_prepare_us
                  << ", prepare_total_us=" << merge_profile.prepare_total_us
                  << ", prepare_fetch_us=" << merge_profile.prepare_fetch_us
                  << ", prepare_distance_us=" << merge_profile.prepare_distance_us
                  << ", prepare_balance_us=" << merge_profile.prepare_balance_us
                  << ", compact_emit_us=" << merge_profile.compact_emit_us
                  << ", commit_us=" << merge_profile.commit_us
                  << ", pq_code_assignment_us=" << merge_profile.pq_code_assignment_us
                  << ", commit_total_us=" << merge_profile.commit_total_us
                  << ", commit_validation_us=" << merge_profile.commit_validation_us
                  << ", commit_reuse_classify_us="
                  << merge_profile.commit_reuse_classify_us
                  << ", commit_materialize_us=" << merge_profile.commit_materialize_us
                  << ", commit_materialized_rows="
                  << merge_profile.commit_materialized_rows
                  << ", commit_materialized_bytes="
                  << merge_profile.commit_materialized_bytes
                  << ", commit_materialize_batches="
                  << merge_profile.commit_materialize_batches
                  << ", commit_max_materialize_rows="
                  << merge_profile.commit_max_materialize_rows
                  << ", commit_pq_encode_us=" << merge_profile.commit_pq_encode_us
                  << ", pq_codes_reused=" << merge_profile.pq_codes_reused
                  << ", pq_codes_reencoded=" << merge_profile.pq_codes_reencoded
                  << ", commit_apply_us=" << merge_profile.commit_apply_us
                  << ", pq_list_flatten_us=" << merge_profile.pq_list_flatten_us
                  << ", docmap_rebuild_us=" << merge_profile.docmap_rebuild_us
                  << ", commit_lock_hold_us=" << merge_profile.commit_lock_hold_us
                  << std::endl;

        frozen_delta.reset();
        frozen_trigger_decision.reset();
        TrimAllocatorRetainedMemory("merge_commit_done");
        record_memory("merge_commit_done", next_insert_idx);
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
        pending_update_wall_ms = 0.0;
        pending_update_materialize_ms = 0.0;
        pending_update_whitening_ms = 0.0;
        pending_update_insert_ms = 0.0;
        pending_update_record_build_ms = 0.0;
        pending_update_insert_encode_ms = 0.0;
        pending_update_insert_entry_ms = 0.0;
        pending_update_insert_commit_ms = 0.0;
        pending_update_onlinepq_maintenance_ms = 0.0;
        pending_update_delete_ms = 0.0;
        pending_update_onlinepq_stats_ms = 0.0;
        pending_update_codebook_update_ms = 0.0;
        pending_update_reencode_ms = 0.0;
        pending_update_replace_ms = 0.0;
        pending_merge_compute_ms = 0.0;
        pending_merge_wall_ms = 0.0;
        pending_global_rebuild_ms = 0.0;
        pending_global_rebuild_wall_ms = 0.0;
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
      total_update_ms + total_update_replace_ms + total_merge_compute_ms +
      total_global_rebuild_ms;
  const double online_update_wall_total_ms =
      total_update_wall_ms + total_merge_wall_ms + total_global_rebuild_wall_ms;
  final_metrics.update_total_ms = online_update_total_ms;
  final_metrics.update_per_vector_ms =
      inserted_rows > 0 ? online_update_total_ms / static_cast<double>(inserted_rows) : 0.0;
  final_metrics.update_throughput_vecps =
      (inserted_rows > 0 && online_update_total_ms > 0.0)
          ? (1000.0 * static_cast<double>(inserted_rows) / online_update_total_ms)
          : 0.0;
  final_metrics.update_wall_total_ms = online_update_wall_total_ms;
  final_metrics.update_wall_per_vector_ms =
      inserted_rows > 0 ? online_update_wall_total_ms / static_cast<double>(inserted_rows) : 0.0;
  final_metrics.update_wall_throughput_vecps =
      (inserted_rows > 0 && online_update_wall_total_ms > 0.0)
          ? (1000.0 * static_cast<double>(inserted_rows) / online_update_wall_total_ms)
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
            << ", replace=" << total_update_replace_ms
            << ", merge_compute=" << total_merge_compute_ms
            << ", global_rebuild=" << total_global_rebuild_ms
            << "), per_vec=" << final_metrics.update_per_vector_ms << "ms; "
            << "Update wall total=" << final_metrics.update_wall_total_ms
            << "ms, wall_per_vec=" << final_metrics.update_wall_per_vector_ms << "ms; "
            << "Scanned avg=" << final_metrics.scanned_avg << ", p50=" << final_metrics.scanned_p50
            << ", p99=" << final_metrics.scanned_p99 << ", max=" << final_metrics.scanned_max
            << "; Query QPS=" << final_metrics.query_qps
            << ", Update throughput=" << final_metrics.update_throughput_vecps << " vec/s"
            << ", Update wall throughput=" << final_metrics.update_wall_throughput_vecps
            << " vec/s"
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
  std::vector<double> update_wall_throughput_values;
  std::vector<double> amortized_update_wall_throughput_values;
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
    if (snap.update_wall_throughput_vecps > 0.0) {
      update_wall_throughput_values.push_back(snap.update_wall_throughput_vecps);
    }
    if (snap.amortized_update_wall_throughput_vecps > 0.0) {
      amortized_update_wall_throughput_values.push_back(
          snap.amortized_update_wall_throughput_vecps);
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
    if (final_metrics.update_wall_throughput_vecps > 0.0) {
      update_wall_throughput_values.push_back(final_metrics.update_wall_throughput_vecps);
      amortized_update_wall_throughput_values.push_back(
          final_metrics.update_wall_throughput_vecps);
    }
  }
  const auto recall_summary = SummarizeSeries(recall_values);
  const auto qps_summary = SummarizeSeries(qps_values);
  const auto latency_summary = SummarizeSeries(latency_values);
  const auto e2e_latency_summary = SummarizeSeries(e2e_latency_values);
  const auto update_throughput_summary = SummarizeSeries(update_throughput_values);
  const auto amortized_update_throughput_summary =
      SummarizeSeries(amortized_update_throughput_values);
  const auto update_wall_throughput_summary = SummarizeSeries(update_wall_throughput_values);
  const auto amortized_update_wall_throughput_summary =
      SummarizeSeries(amortized_update_wall_throughput_values);
  const double total_maintenance_ms =
      total_update_ms + total_update_replace_ms + total_merge_compute_ms +
      total_global_rebuild_ms;
  const double total_wall_maintenance_ms =
      total_update_wall_ms + total_merge_wall_ms + total_global_rebuild_wall_ms;

  const auto write_timing_metrics = [&](const char* indent) {
    ofs << indent << "\"build_rebuild_ms\": " << rebuild_ms_total << ",\n";
    ofs << indent << "\"initial_used_prebuilt_index\": false,\n";
    ofs << indent << "\"initial_index_load_ms\": 0,\n";
    ofs << indent << "\"initial_materialize_ms\": " << init_materialize_ms << ",\n";
    ofs << indent << "\"initial_build_aligned_ms\": " << init_rebuild_ms << ",\n";
    ofs << indent << "\"initial_whitening_ms\": " << init_whitening_ms << ",\n";
    ofs << indent << "\"initial_whitening_transform_ms\": "
        << init_whitening_transform_ms << ",\n";
    ofs << indent << "\"initial_main_build_ms\": " << init_main_build_ms << ",\n";
    ofs << indent << "\"initial_main_add_ms\": " << init_main_add_ms << ",\n";
    ofs << indent << "\"initial_build_wall_ms\": " << init_total_wall_ms << ",\n";
    ofs << indent << "\"initial_delta_materialize_ms\": " << init_delta_materialize_ms
        << ",\n";
    ofs << indent << "\"initial_delta_build_ms\": " << init_delta_build_ms << ",\n";
    ofs << indent << "\"initial_delta_add_ms\": " << init_delta_add_ms << ",\n";
    ofs << indent << "\"initial_delta_wall_ms\": " << init_delta_wall_ms << ",\n";
    ofs << indent << "\"global_rebuild_count\": " << global_rebuild_count << ",\n";
    ofs << indent << "\"update_total_ms\": " << final_metrics.update_total_ms << ",\n";
    ofs << indent << "\"update_apply_ms\": " << total_update_ms << ",\n";
    ofs << indent << "\"update_wall_ms\": " << final_metrics.update_wall_total_ms << ",\n";
    ofs << indent << "\"update_materialize_ms\": " << total_update_materialize_ms << ",\n";
    ofs << indent << "\"update_whitening_ms\": " << total_update_whitening_ms << ",\n";
    ofs << indent << "\"update_insert_ms\": " << total_update_insert_ms << ",\n";
    ofs << indent << "\"update_record_build_ms\": " << total_update_record_build_ms << ",\n";
    ofs << indent << "\"update_insert_encode_ms\": " << total_update_insert_encode_ms
        << ",\n";
    ofs << indent << "\"update_insert_entry_ms\": " << total_update_insert_entry_ms
        << ",\n";
    ofs << indent << "\"update_insert_commit_ms\": " << total_update_insert_commit_ms
        << ",\n";
    ofs << indent << "\"update_onlinepq_maintenance_ms\": "
        << total_update_onlinepq_maintenance_ms << ",\n";
    ofs << indent << "\"update_delete_ms\": " << total_update_delete_ms << ",\n";
    ofs << indent << "\"update_onlinepq_stats_ms\": " << total_update_onlinepq_stats_ms
        << ",\n";
    ofs << indent << "\"update_codebook_update_ms\": " << total_update_codebook_update_ms
        << ",\n";
    ofs << indent << "\"update_reencode_ms\": " << total_update_reencode_ms << ",\n";
    ofs << indent << "\"update_replace_ms\": " << total_update_replace_ms << ",\n";
    ofs << indent << "\"update_merge_compute_ms\": " << total_merge_compute_ms << ",\n";
    ofs << indent << "\"update_merge_wall_ms\": " << total_merge_wall_ms << ",\n";
    ofs << indent << "\"update_global_rebuild_ms\": " << total_global_rebuild_ms << ",\n";
    ofs << indent << "\"update_global_rebuild_wall_ms\": "
        << total_global_rebuild_wall_ms << ",\n";
    ofs << indent << "\"activation_materialize_ms\": " << total_activation_materialize_ms
        << ",\n";
    ofs << indent << "\"activation_whitening_ms\": " << total_activation_whitening_ms
        << ",\n";
    ofs << indent << "\"activation_cache_ms\": " << total_activation_cache_ms << ",\n";
    ofs << indent << "\"activation_build_ms\": " << total_activation_build_ms << ",\n";
    ofs << indent << "\"activation_insert_ms\": " << total_activation_insert_ms << ",\n";
    ofs << indent << "\"activation_wall_ms\": " << total_activation_wall_ms << ",\n";
    ofs << indent << "\"total_maintenance_ms\": " << total_maintenance_ms << ",\n";
    ofs << indent << "\"total_wall_maintenance_ms\": " << total_wall_maintenance_ms << ",\n";
    ofs << indent << "\"update_per_vector_ms\": " << final_metrics.update_per_vector_ms
        << ",\n";
    ofs << indent << "\"update_wall_per_vector_ms\": "
        << final_metrics.update_wall_per_vector_ms << ",\n";
    ofs << indent << "\"update_throughput_vecps\": "
        << final_metrics.update_throughput_vecps << ",\n";
    ofs << indent << "\"update_wall_throughput_vecps\": "
        << final_metrics.update_wall_throughput_vecps << ",\n";
  };

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
  auto write_memory_trace = [&]() -> Status {
    TrimAllocatorRetainedMemory("final");
    record_memory("final", std::min<uint32_t>(next_insert_idx, nx));
    const std::vector<ann::eval_memory::MemoryComponent> metadata = {
        {"base_rows", nx},
        {"base_dim", config.dim},
        {"base_raw_full_dataset_bytes",
         ann::eval_memory::MatrixActiveBytes(nx, config.dim)},
        {"query_rows", static_cast<uint64_t>(Q.rows())},
        {"query_dim", static_cast<uint64_t>(Q.cols())},
        {"main_index_rows_initial", main_rows_initial},
        {"main_max_rows", main_max_rows},
        {"delta_train_rows", delta_train_rows},
        {"stream_start_row", stream_start_idx},
        {"stream_rows", total_stream_rows},
        {"memory_block_rows", kAddBlockRows},
        {"raw_base_is_resident", 0}};
    return memory_trace.WriteJson(memory_trace_path.string(), metadata);
  };
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
    ofs << "    \"max_main\": " << main_max_rows << ",\n";
    ofs << "    \"main_rows_final\": " << main_rows_current << ",\n";
    ofs << "    \"evicted_main_rows_total\": " << evicted_main_rows_total << ",\n";
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
    write_timing_metrics("    ");
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
    ofs << "    \"update_wall_throughput_avg_vecps\": "
        << update_wall_throughput_summary.avg << ",\n";
    ofs << "    \"amortized_update_wall_throughput_avg_vecps\": "
        << amortized_update_wall_throughput_summary.avg << ",\n";
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
    ofs << "    \"evicted_main_rows_total\": " << evicted_main_rows_total << ",\n";
    ofs << "    \"live_main_begin_doc\": "
        << (main_window_doc_ids.empty() ? 0 : main_window_doc_ids.front()) << ",\n";
    ofs << "    \"live_main_end_doc\": "
        << (main_window_doc_ids.empty() ? 0 : main_window_doc_ids.back() + 1) << ",\n";
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
      ofs << "      \"evicted_main_rows\": " << ev.evicted_main_rows << ",\n";
      ofs << "      \"main_rows_before\": " << ev.main_rows_before << ",\n";
      ofs << "      \"main_rows_after\": " << ev.main_rows_after << ",\n";
      ofs << "      \"live_main_begin_doc\": " << ev.live_main_begin_doc << ",\n";
      ofs << "      \"live_main_end_doc\": " << ev.live_main_end_doc << ",\n";
      ofs << "      \"main_materialize_ms\": " << ev.main_materialize_ms << ",\n";
      ofs << "      \"main_build_ms\": " << ev.main_build_ms << ",\n";
      ofs << "      \"main_add_ms\": " << ev.main_add_ms << ",\n";
      ofs << "      \"pre_replacement_ms\": " << ev.pre_replacement_ms << ",\n";
      ofs << "      \"merge_compute_ms\": " << ev.merge_compute_ms << ",\n";
      ofs << "      \"merge_ms\": " << ev.merge_ms << ",\n";
      ofs << "      \"main_imbalance_after_real\": " << ev.main_imbalance_after_real << ",\n";
      ofs << "      \"profiling\": {\n";
      ofs << "        \"effective_nlist\": " << ev.profiling.effective_nlist << ",\n";
      ofs << "        \"frozen_records\": " << ev.profiling.frozen_records << ",\n";
      ofs << "        \"seed_partitions\": " << ev.profiling.seed_partitions << ",\n";
      ofs << "        \"neighborhoods\": " << ev.profiling.neighborhoods << ",\n";
      ofs << "        \"main_records_loaded\": "
          << ev.profiling.main_records_loaded << ",\n";
      ofs << "        \"pooled_records\": " << ev.profiling.pooled_records << ",\n";
      ofs << "        \"repartitioned_records\": "
          << ev.profiling.repartitioned_records << ",\n";
      ofs << "        \"patch_records\": " << ev.profiling.patch_records << ",\n";
      ofs << "        \"compact_patch_records\": "
          << ev.profiling.compact_patch_records << ",\n";
      ofs << "        \"compact_patch_estimated_bytes\": "
          << ev.profiling.compact_patch_estimated_bytes << ",\n";
      ofs << "        \"assignment_distance_evaluations\": "
          << ev.profiling.assignment_distance_evaluations << ",\n";
      ofs << "        \"assignment_workspace_bytes\": "
          << ev.profiling.assignment_workspace_bytes << ",\n";
      ofs << "        \"assignment_chunk_records\": "
          << ev.profiling.assignment_chunk_records << ",\n";
      ofs << "        \"assignment_chunk_count\": "
          << ev.profiling.assignment_chunk_count << ",\n";
      ofs << "        \"merge_delta_to_main_assignment_us\": "
          << ev.profiling.merge_delta_to_main_assignment_us << ",\n";
      ofs << "        \"merge_assignment_distance_us\": "
          << ev.profiling.merge_assignment_distance_us << ",\n";
      ofs << "        \"merge_assignment_top_r_us\": "
          << ev.profiling.merge_assignment_top_r_us << ",\n";
      ofs << "        \"merge_assignment_balance_us\": "
          << ev.profiling.merge_assignment_balance_us << ",\n";
      ofs << "        \"merge_assignment_materialize_us\": "
          << ev.profiling.merge_assignment_materialize_us << ",\n";
      ofs << "        \"stats_us\": " << ev.profiling.stats_us << ",\n";
      ofs << "        \"scoring_us\": " << ev.profiling.scoring_us << ",\n";
      ofs << "        \"top_r_neighbor_us\": " << ev.profiling.top_r_neighbor_us << ",\n";
      ofs << "        \"fetch_main_records_us\": "
          << ev.profiling.fetch_main_records_us << ",\n";
      ofs << "        \"repartition_us\": " << ev.profiling.repartition_us << ",\n";
      ofs << "        \"patch_prepare_us\": " << ev.profiling.patch_prepare_us << ",\n";
      ofs << "        \"commit_us\": " << ev.profiling.commit_us << ",\n";
      ofs << "        \"pq_code_assignment_us\": "
          << ev.profiling.pq_code_assignment_us << ",\n";
      ofs << "        \"prepare_total_us\": " << ev.profiling.prepare_total_us << ",\n";
      ofs << "        \"prepare_fetch_us\": " << ev.profiling.prepare_fetch_us << ",\n";
      ofs << "        \"prepare_distance_us\": " << ev.profiling.prepare_distance_us
          << ",\n";
      ofs << "        \"prepare_balance_us\": " << ev.profiling.prepare_balance_us
          << ",\n";
      ofs << "        \"compact_emit_us\": " << ev.profiling.compact_emit_us << ",\n";
      ofs << "        \"commit_total_us\": " << ev.profiling.commit_total_us << ",\n";
      ofs << "        \"commit_validation_us\": " << ev.profiling.commit_validation_us
          << ",\n";
      ofs << "        \"commit_reuse_classify_us\": "
          << ev.profiling.commit_reuse_classify_us << ",\n";
      ofs << "        \"commit_materialize_us\": "
          << ev.profiling.commit_materialize_us << ",\n";
      ofs << "        \"commit_materialized_rows\": "
          << ev.profiling.commit_materialized_rows << ",\n";
      ofs << "        \"commit_materialized_bytes\": "
          << ev.profiling.commit_materialized_bytes << ",\n";
      ofs << "        \"commit_materialize_batches\": "
          << ev.profiling.commit_materialize_batches << ",\n";
      ofs << "        \"commit_max_materialize_rows\": "
          << ev.profiling.commit_max_materialize_rows << ",\n";
      ofs << "        \"commit_pq_encode_us\": " << ev.profiling.commit_pq_encode_us
          << ",\n";
      ofs << "        \"pq_codes_reused\": " << ev.profiling.pq_codes_reused << ",\n";
      ofs << "        \"pq_codes_reencoded\": " << ev.profiling.pq_codes_reencoded
          << ",\n";
      ofs << "        \"commit_apply_us\": " << ev.profiling.commit_apply_us << ",\n";
      ofs << "        \"pq_list_flatten_us\": " << ev.profiling.pq_list_flatten_us
          << ",\n";
      ofs << "        \"docmap_rebuild_us\": " << ev.profiling.docmap_rebuild_us
          << ",\n";
      ofs << "        \"commit_lock_hold_us\": " << ev.profiling.commit_lock_hold_us
          << "\n";
      ofs << "      }\n";
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
      ofs << "      \"main_rows\": " << snap.main_rows << ",\n";
      ofs << "      \"frozen_delta_docs\": " << snap.frozen_delta_docs << ",\n";
      ofs << "      \"active_delta_docs\": " << snap.active_delta_docs << ",\n";
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
      ofs << "      \"update_wall_ms\": " << snap.update_wall_ms << ",\n";
      ofs << "      \"update_materialize_ms\": " << snap.update_materialize_ms << ",\n";
      ofs << "      \"update_whitening_ms\": " << snap.update_whitening_ms << ",\n";
      ofs << "      \"update_insert_ms\": " << snap.update_insert_ms << ",\n";
      ofs << "      \"update_record_build_ms\": " << snap.update_record_build_ms << ",\n";
      ofs << "      \"update_insert_encode_ms\": " << snap.update_insert_encode_ms << ",\n";
      ofs << "      \"update_insert_entry_ms\": " << snap.update_insert_entry_ms << ",\n";
      ofs << "      \"update_insert_commit_ms\": " << snap.update_insert_commit_ms << ",\n";
      ofs << "      \"update_onlinepq_maintenance_ms\": "
          << snap.update_onlinepq_maintenance_ms << ",\n";
      ofs << "      \"update_delete_ms\": " << snap.update_delete_ms << ",\n";
      ofs << "      \"update_onlinepq_stats_ms\": " << snap.update_onlinepq_stats_ms << ",\n";
      ofs << "      \"update_codebook_update_ms\": " << snap.update_codebook_update_ms
          << ",\n";
      ofs << "      \"update_reencode_ms\": " << snap.update_reencode_ms << ",\n";
      ofs << "      \"update_replace_ms\": " << snap.update_replace_ms << ",\n";
      ofs << "      \"merge_compute_ms\": " << snap.merge_compute_ms << ",\n";
      ofs << "      \"merge_wall_ms\": " << snap.merge_wall_ms << ",\n";
      ofs << "      \"global_rebuild_ms\": " << snap.global_rebuild_ms << ",\n";
      ofs << "      \"global_rebuild_wall_ms\": " << snap.global_rebuild_wall_ms << ",\n";
      ofs << "      \"snapshot_total_ms\": " << snap.snapshot_total_ms << ",\n";
      ofs << "      \"snapshot_wall_total_ms\": " << snap.snapshot_wall_total_ms << ",\n";
      ofs << "      \"throughput\": " << snap.update_throughput_vecps << ",\n";
      ofs << "      \"amortized_update_throughput_vecps\": "
          << snap.amortized_update_throughput_vecps << ",\n";
      ofs << "      \"update_wall_throughput_vecps\": "
          << snap.update_wall_throughput_vecps << ",\n";
      ofs << "      \"amortized_update_wall_throughput_vecps\": "
          << snap.amortized_update_wall_throughput_vecps << ",\n";
      ofs << "      \"rss_bytes\": " << snap.rss_bytes << ",\n";
      ofs << "      \"peak_rss_bytes\": " << snap.peak_rss_bytes << "\n";
      ofs << "    }";
      if (i + 1 < snapshots.size()) {
        ofs << ",";
      }
      ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";

    Status memory_status = write_memory_trace();
    if (!memory_status.ok()) {
      std::cerr << memory_status.ToString() << std::endl;
      return 1;
    }
    std::cout << "Saved metrics to " << result_path << std::endl;
    std::cout << "Saved memory trace to " << memory_trace_path << std::endl;
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
  ofs << "    \"max_main\": " << main_max_rows << ",\n";
  ofs << "    \"main_rows_final\": " << main_rows_current << ",\n";
  ofs << "    \"evicted_main_rows_total\": " << evicted_main_rows_total << ",\n";
  ofs << "    \"live_main_begin_doc\": "
      << (main_window_doc_ids.empty() ? 0 : main_window_doc_ids.front()) << ",\n";
  ofs << "    \"live_main_end_doc\": "
      << (main_window_doc_ids.empty() ? 0 : main_window_doc_ids.back() + 1) << ",\n";
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
  write_timing_metrics("    ");
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
  ofs << "    \"total_wall_maintenance_ms\": " << total_wall_maintenance_ms << ",\n";
  ofs << "    \"update_throughput_avg_vecps\": "
      << update_throughput_summary.avg << ",\n";
  ofs << "    \"amortized_update_throughput_avg_vecps\": "
      << amortized_update_throughput_summary.avg << ",\n";
  ofs << "    \"update_wall_throughput_avg_vecps\": "
      << update_wall_throughput_summary.avg << ",\n";
  ofs << "    \"amortized_update_wall_throughput_avg_vecps\": "
      << amortized_update_wall_throughput_summary.avg << ",\n";
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
    ofs << "      \"evicted_main_rows\": " << ev.evicted_main_rows << ",\n";
    ofs << "      \"main_rows_before\": " << ev.main_rows_before << ",\n";
    ofs << "      \"main_rows_after\": " << ev.main_rows_after << ",\n";
    ofs << "      \"live_main_begin_doc\": " << ev.live_main_begin_doc << ",\n";
    ofs << "      \"live_main_end_doc\": " << ev.live_main_end_doc << ",\n";
    ofs << "      \"main_materialize_ms\": " << ev.main_materialize_ms << ",\n";
    ofs << "      \"main_build_ms\": " << ev.main_build_ms << ",\n";
    ofs << "      \"main_add_ms\": " << ev.main_add_ms << ",\n";
    ofs << "      \"pre_replacement_ms\": " << ev.pre_replacement_ms << ",\n";
    ofs << "      \"fixed_ivf_centroids\": true,\n";
    ofs << "      \"pq_codebook_retrained\": false,\n";
    ofs << "      \"original_assignment_patch\": true,\n";
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
    ofs << "      \"merge_ms\": " << ev.merge_ms << ",\n";
    ofs << "      \"profiling\": {\n";
    ofs << "        \"effective_nlist\": " << ev.profiling.effective_nlist << ",\n";
    ofs << "        \"frozen_records\": " << ev.profiling.frozen_records << ",\n";
    ofs << "        \"seed_partitions\": " << ev.profiling.seed_partitions << ",\n";
    ofs << "        \"neighborhoods\": " << ev.profiling.neighborhoods << ",\n";
    ofs << "        \"main_records_loaded\": "
        << ev.profiling.main_records_loaded << ",\n";
    ofs << "        \"pooled_records\": " << ev.profiling.pooled_records << ",\n";
    ofs << "        \"repartitioned_records\": "
        << ev.profiling.repartitioned_records << ",\n";
    ofs << "        \"patch_records\": " << ev.profiling.patch_records << ",\n";
    ofs << "        \"compact_patch_records\": "
        << ev.profiling.compact_patch_records << ",\n";
    ofs << "        \"compact_patch_estimated_bytes\": "
        << ev.profiling.compact_patch_estimated_bytes << ",\n";
    ofs << "        \"assignment_distance_evaluations\": "
        << ev.profiling.assignment_distance_evaluations << ",\n";
    ofs << "        \"assignment_workspace_bytes\": "
        << ev.profiling.assignment_workspace_bytes << ",\n";
    ofs << "        \"assignment_chunk_records\": "
        << ev.profiling.assignment_chunk_records << ",\n";
    ofs << "        \"assignment_chunk_count\": "
        << ev.profiling.assignment_chunk_count << ",\n";
    ofs << "        \"merge_delta_to_main_assignment_us\": "
        << ev.profiling.merge_delta_to_main_assignment_us << ",\n";
    ofs << "        \"merge_assignment_distance_us\": "
        << ev.profiling.merge_assignment_distance_us << ",\n";
    ofs << "        \"merge_assignment_top_r_us\": "
        << ev.profiling.merge_assignment_top_r_us << ",\n";
    ofs << "        \"merge_assignment_balance_us\": "
        << ev.profiling.merge_assignment_balance_us << ",\n";
    ofs << "        \"merge_assignment_materialize_us\": "
        << ev.profiling.merge_assignment_materialize_us << ",\n";
    ofs << "        \"stats_us\": " << ev.profiling.stats_us << ",\n";
    ofs << "        \"scoring_us\": " << ev.profiling.scoring_us << ",\n";
    ofs << "        \"top_r_neighbor_us\": " << ev.profiling.top_r_neighbor_us << ",\n";
    ofs << "        \"fetch_main_records_us\": "
        << ev.profiling.fetch_main_records_us << ",\n";
    ofs << "        \"repartition_us\": " << ev.profiling.repartition_us << ",\n";
    ofs << "        \"patch_prepare_us\": " << ev.profiling.patch_prepare_us << ",\n";
    ofs << "        \"commit_us\": " << ev.profiling.commit_us << ",\n";
    ofs << "        \"pq_code_assignment_us\": "
        << ev.profiling.pq_code_assignment_us << ",\n";
    ofs << "        \"prepare_total_us\": " << ev.profiling.prepare_total_us << ",\n";
    ofs << "        \"prepare_fetch_us\": " << ev.profiling.prepare_fetch_us << ",\n";
    ofs << "        \"prepare_distance_us\": " << ev.profiling.prepare_distance_us
        << ",\n";
    ofs << "        \"prepare_balance_us\": " << ev.profiling.prepare_balance_us << ",\n";
    ofs << "        \"compact_emit_us\": " << ev.profiling.compact_emit_us << ",\n";
    ofs << "        \"commit_total_us\": " << ev.profiling.commit_total_us << ",\n";
    ofs << "        \"commit_validation_us\": " << ev.profiling.commit_validation_us
        << ",\n";
    ofs << "        \"commit_reuse_classify_us\": "
        << ev.profiling.commit_reuse_classify_us << ",\n";
    ofs << "        \"commit_materialize_us\": "
        << ev.profiling.commit_materialize_us << ",\n";
    ofs << "        \"commit_materialized_rows\": "
        << ev.profiling.commit_materialized_rows << ",\n";
    ofs << "        \"commit_materialized_bytes\": "
        << ev.profiling.commit_materialized_bytes << ",\n";
    ofs << "        \"commit_materialize_batches\": "
        << ev.profiling.commit_materialize_batches << ",\n";
    ofs << "        \"commit_max_materialize_rows\": "
        << ev.profiling.commit_max_materialize_rows << ",\n";
    ofs << "        \"commit_pq_encode_us\": " << ev.profiling.commit_pq_encode_us
        << ",\n";
    ofs << "        \"pq_codes_reused\": " << ev.profiling.pq_codes_reused << ",\n";
    ofs << "        \"pq_codes_reencoded\": " << ev.profiling.pq_codes_reencoded
        << ",\n";
    ofs << "        \"commit_apply_us\": " << ev.profiling.commit_apply_us << ",\n";
    ofs << "        \"pq_list_flatten_us\": " << ev.profiling.pq_list_flatten_us
        << ",\n";
    ofs << "        \"docmap_rebuild_us\": " << ev.profiling.docmap_rebuild_us << ",\n";
    ofs << "        \"commit_lock_hold_us\": " << ev.profiling.commit_lock_hold_us
        << "\n";
    ofs << "      }\n";
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
    ofs << "      \"materialize_ms\": " << ev.materialize_ms << ",\n";
    ofs << "      \"whitening_ms\": " << ev.whitening_ms << ",\n";
    ofs << "      \"whitening_transform_ms\": " << ev.whitening_transform_ms << ",\n";
    ofs << "      \"query_transform_ms\": " << ev.query_transform_ms << ",\n";
    ofs << "      \"cache_ms\": " << ev.cache_ms << ",\n";
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
    ofs << "      \"update_wall_ms\": " << snap.update_wall_ms << ",\n";
    ofs << "      \"update_materialize_ms\": " << snap.update_materialize_ms << ",\n";
    ofs << "      \"update_whitening_ms\": " << snap.update_whitening_ms << ",\n";
    ofs << "      \"update_insert_ms\": " << snap.update_insert_ms << ",\n";
    ofs << "      \"update_record_build_ms\": " << snap.update_record_build_ms << ",\n";
    ofs << "      \"update_insert_encode_ms\": " << snap.update_insert_encode_ms << ",\n";
    ofs << "      \"update_insert_entry_ms\": " << snap.update_insert_entry_ms << ",\n";
    ofs << "      \"update_insert_commit_ms\": " << snap.update_insert_commit_ms << ",\n";
    ofs << "      \"update_onlinepq_maintenance_ms\": "
        << snap.update_onlinepq_maintenance_ms << ",\n";
    ofs << "      \"update_delete_ms\": " << snap.update_delete_ms << ",\n";
    ofs << "      \"update_onlinepq_stats_ms\": " << snap.update_onlinepq_stats_ms << ",\n";
    ofs << "      \"update_codebook_update_ms\": " << snap.update_codebook_update_ms
        << ",\n";
    ofs << "      \"update_reencode_ms\": " << snap.update_reencode_ms << ",\n";
    ofs << "      \"update_replace_ms\": " << snap.update_replace_ms << ",\n";
    ofs << "      \"merge_compute_ms\": " << snap.merge_compute_ms << ",\n";
    ofs << "      \"merge_wall_ms\": " << snap.merge_wall_ms << ",\n";
    ofs << "      \"global_rebuild_ms\": " << snap.global_rebuild_ms << ",\n";
    ofs << "      \"global_rebuild_wall_ms\": " << snap.global_rebuild_wall_ms << ",\n";
    ofs << "      \"query_eval_ms\": " << snap.query_eval_ms << ",\n";
    ofs << "      \"snapshot_total_ms\": " << snap.snapshot_total_ms << ",\n";
    ofs << "      \"snapshot_wall_total_ms\": " << snap.snapshot_wall_total_ms << ",\n";
    ofs << "      \"update_throughput_vecps\": " << snap.update_throughput_vecps << ",\n";
    ofs << "      \"amortized_update_throughput_vecps\": "
        << snap.amortized_update_throughput_vecps << ",\n";
    ofs << "      \"update_wall_throughput_vecps\": "
        << snap.update_wall_throughput_vecps << ",\n";
    ofs << "      \"amortized_update_wall_throughput_vecps\": "
        << snap.amortized_update_wall_throughput_vecps << ",\n";
    ofs << "      \"rss_bytes\": " << snap.rss_bytes << ",\n";
    ofs << "      \"peak_rss_bytes\": " << snap.peak_rss_bytes << ",\n";
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
      ofs << "          \"update_wall_ms\": " << mb.update_wall_ms << ",\n";
      ofs << "          \"update_materialize_ms\": " << mb.update_materialize_ms << ",\n";
      ofs << "          \"update_whitening_ms\": " << mb.update_whitening_ms << ",\n";
      ofs << "          \"update_insert_ms\": " << mb.update_insert_ms << ",\n";
      ofs << "          \"update_record_build_ms\": " << mb.update_record_build_ms << ",\n";
      ofs << "          \"update_insert_encode_ms\": " << mb.update_insert_encode_ms << ",\n";
      ofs << "          \"update_insert_entry_ms\": " << mb.update_insert_entry_ms << ",\n";
      ofs << "          \"update_insert_commit_ms\": " << mb.update_insert_commit_ms << ",\n";
      ofs << "          \"update_onlinepq_maintenance_ms\": "
          << mb.update_onlinepq_maintenance_ms << ",\n";
      ofs << "          \"update_delete_ms\": " << mb.update_delete_ms << ",\n";
      ofs << "          \"update_onlinepq_stats_ms\": " << mb.update_onlinepq_stats_ms
          << ",\n";
      ofs << "          \"update_codebook_update_ms\": " << mb.update_codebook_update_ms
          << ",\n";
      ofs << "          \"update_reencode_ms\": " << mb.update_reencode_ms << ",\n";
      ofs << "          \"update_throughput_vecps\": " << mb.update_throughput_vecps << ",\n";
      ofs << "          \"update_wall_throughput_vecps\": "
          << mb.update_wall_throughput_vecps << ",\n";
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

  Status memory_status = write_memory_trace();
  if (!memory_status.ok()) {
    std::cerr << memory_status.ToString() << std::endl;
    return 1;
  }
  std::cout << "Saved metrics to " << result_path << std::endl;
  std::cout << "Saved memory trace to " << memory_trace_path << std::endl;
  return 0;
}
