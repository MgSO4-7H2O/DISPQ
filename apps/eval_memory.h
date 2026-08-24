#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "common/dataset.h"
#include "common/result.h"
#include "common/status.h"
#include "common/types.h"
#include "index/ivf.h"
#include "whitening/whitening.h"

namespace ann::eval_memory {

using MatrixRD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline uint64_t MatrixBytes(const MatrixRM& matrix) {
  return static_cast<uint64_t>(matrix.rows()) *
         static_cast<uint64_t>(matrix.cols()) * sizeof(float);
}

inline uint64_t MatrixActiveBytes(uint32_t rows, uint32_t dim) {
  return static_cast<uint64_t>(rows) * static_cast<uint64_t>(dim) * sizeof(float);
}

inline uint64_t VectorBytes(const Eigen::VectorXf& vector) {
  return static_cast<uint64_t>(vector.size()) * sizeof(float);
}

inline uint64_t ReadProcStatusBytesLocal(const std::string& key) {
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

struct BaseVectorSource {
  std::optional<std::string> fvecs_path;
  MatrixRM resident;
  uint32_t rows{0};
  uint32_t dim{0};

  bool UsesResidentMatrix() const {
    return !fvecs_path.has_value();
  }

  uint64_t ResidentBytes() const {
    return UsesResidentMatrix() ? MatrixBytes(resident) : 0;
  }

  Result<MatrixRM> LoadRange(uint32_t begin, uint32_t count) const {
    if (begin > rows || count > rows - begin) {
      return Status::InvalidArgument("BaseVectorSource::LoadRange: out of bounds");
    }
    if (fvecs_path.has_value()) {
      return LoadFvecsRange(*fvecs_path, begin, count);
    }
    return MatrixRM(resident.middleRows(static_cast<Eigen::Index>(begin),
                                        static_cast<Eigen::Index>(count)));
  }
};

inline Result<BaseVectorSource> MakeFvecsBaseSource(const std::string& path) {
  auto meta_res = InspectFvecs(path);
  if (!meta_res.ok()) {
    return meta_res.status();
  }
  BaseVectorSource source;
  source.fvecs_path = path;
  source.rows = meta_res.value().rows;
  source.dim = meta_res.value().dim;
  return source;
}

inline BaseVectorSource MakeResidentBaseSource(MatrixRM matrix) {
  BaseVectorSource source;
  source.rows = static_cast<uint32_t>(matrix.rows());
  source.dim = static_cast<uint32_t>(matrix.cols());
  source.resident = std::move(matrix);
  return source;
}

struct WhiteningMomentAccumulator {
  uint32_t dim{0};
  uint64_t count{0};
  Eigen::VectorXd sum;
  MatrixRD cross;

  explicit WhiteningMomentAccumulator(uint32_t d)
      : dim(d),
        sum(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d))),
        cross(MatrixRD::Zero(static_cast<Eigen::Index>(d),
                             static_cast<Eigen::Index>(d))) {}

  Status AddRawBatch(Eigen::Ref<const MatrixRM> raw) {
    if (raw.cols() != static_cast<Eigen::Index>(dim)) {
      return Status::InvalidArgument("WhiteningMomentAccumulator: dim mismatch");
    }
    if (raw.rows() == 0) {
      return Status::OK();
    }
    MatrixRD raw_d = raw.cast<double>();
    sum += raw_d.colwise().sum().transpose();
    cross.noalias() += raw_d.transpose() * raw_d;
    count += static_cast<uint64_t>(raw.rows());
    return Status::OK();
  }

  Result<VersionId> FitInto(const std::shared_ptr<WhiteningModel>& whitening) const {
    if (!whitening) {
      return Status::InvalidArgument("WhiteningMomentAccumulator: null whitening model");
    }
    if (count == 0) {
      return Status::InvalidArgument("WhiteningMomentAccumulator: no rows");
    }
    const double denom = static_cast<double>(count);
    const Eigen::VectorXd mean_d = sum / denom;
    MatrixRD cov_d = cross / denom - mean_d * mean_d.transpose();
    Eigen::VectorXf mean = mean_d.cast<float>();
    MatrixRM cov = cov_d.cast<float>();
    return whitening->FitFromMeanCov(mean, cov);
  }
};

inline Result<VersionId> FitWhiteningFromSourcePrefix(
    const BaseVectorSource& source,
    uint32_t rows,
    uint32_t block_rows,
    const std::shared_ptr<WhiteningModel>& whitening) {
  if (rows == 0 || rows > source.rows) {
    return Status::InvalidArgument("FitWhiteningFromSourcePrefix: invalid rows");
  }
  WhiteningMomentAccumulator acc(source.dim);
  const uint32_t step = std::max<uint32_t>(1, block_rows);
  for (uint32_t begin = 0; begin < rows; begin += step) {
    const uint32_t count = std::min<uint32_t>(step, rows - begin);
    auto raw_res = source.LoadRange(begin, count);
    if (!raw_res.ok()) {
      return raw_res.status();
    }
    Status st = acc.AddRawBatch(raw_res.value());
    if (!st.ok()) {
      return st;
    }
  }
  return acc.FitInto(whitening);
}

inline Result<VersionId> FitWhiteningFromWhitenedPrefix(
    const MatrixRM& whitened,
    uint32_t rows,
    VersionId old_version,
    uint32_t block_rows,
    const std::shared_ptr<WhiteningModel>& whitening) {
  if (!whitening) {
    return Status::InvalidArgument("FitWhiteningFromWhitenedPrefix: null whitening model");
  }
  if (rows == 0 || rows > static_cast<uint32_t>(whitened.rows())) {
    return Status::InvalidArgument("FitWhiteningFromWhitenedPrefix: invalid rows");
  }
  const uint32_t dim = static_cast<uint32_t>(whitened.cols());
  WhiteningMomentAccumulator acc(dim);
  const uint32_t step = std::max<uint32_t>(1, block_rows);
  for (uint32_t begin = 0; begin < rows; begin += step) {
    const uint32_t count = std::min<uint32_t>(step, rows - begin);
    auto raw_res = whitening->InverseTransformBatch(
        whitened.middleRows(static_cast<Eigen::Index>(begin),
                            static_cast<Eigen::Index>(count)),
        old_version);
    if (!raw_res.ok()) {
      return raw_res.status();
    }
    Status st = acc.AddRawBatch(raw_res.value());
    if (!st.ok()) {
      return st;
    }
  }
  return acc.FitInto(whitening);
}

inline Status NormalizeRowsIfNeeded(MatrixRM* matrix, bool use_cosine) {
  if (matrix == nullptr) {
    return Status::InvalidArgument("NormalizeRowsIfNeeded: null matrix");
  }
  if (!use_cosine) {
    return Status::OK();
  }
  for (Eigen::Index i = 0; i < matrix->rows(); ++i) {
    const float norm = matrix->row(i).norm();
    if (norm > 0.0f) {
      matrix->row(i) /= norm;
    }
  }
  return Status::OK();
}

inline Result<MatrixRM> TransformSourceRangeToWhitened(
    const BaseVectorSource& source,
    uint32_t begin,
    uint32_t rows,
    VersionId version,
    const std::shared_ptr<WhiteningModel>& whitening,
    bool use_cosine) {
  auto raw_res = source.LoadRange(begin, rows);
  if (!raw_res.ok()) {
    return raw_res.status();
  }
  auto whitened_res = whitening->TransformBatch(raw_res.value(), version);
  if (!whitened_res.ok()) {
    return whitened_res.status();
  }
  MatrixRM whitened = std::move(whitened_res.value());
  Status norm = NormalizeRowsIfNeeded(&whitened, use_cosine);
  if (!norm.ok()) {
    return norm;
  }
  return whitened;
}

inline Status TransformSourcePrefixToWhitened(
    const BaseVectorSource& source,
    uint32_t rows,
    uint32_t block_rows,
    VersionId version,
    const std::shared_ptr<WhiteningModel>& whitening,
    bool use_cosine,
    MatrixRM* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("TransformSourcePrefixToWhitened: null output");
  }
  if (rows > source.rows) {
    return Status::InvalidArgument("TransformSourcePrefixToWhitened: invalid rows");
  }
  out->resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(source.dim));
  const uint32_t step = std::max<uint32_t>(1, block_rows);
  for (uint32_t begin = 0; begin < rows; begin += step) {
    const uint32_t count = std::min<uint32_t>(step, rows - begin);
    auto block_res = TransformSourceRangeToWhitened(
        source, begin, count, version, whitening, use_cosine);
    if (!block_res.ok()) {
      return block_res.status();
    }
    out->middleRows(static_cast<Eigen::Index>(begin),
                    static_cast<Eigen::Index>(count)) = block_res.value();
  }
  return Status::OK();
}

inline Status RetargetWhitenedPrefix(
    const MatrixRM& old_whitened,
    uint32_t rows,
    uint32_t block_rows,
    VersionId old_version,
    VersionId new_version,
    const std::shared_ptr<WhiteningModel>& whitening,
    bool use_cosine,
    MatrixRM* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("RetargetWhitenedPrefix: null output");
  }
  if (!whitening) {
    return Status::InvalidArgument("RetargetWhitenedPrefix: null whitening model");
  }
  if (rows > static_cast<uint32_t>(old_whitened.rows())) {
    return Status::InvalidArgument("RetargetWhitenedPrefix: invalid rows");
  }
  const uint32_t dim = static_cast<uint32_t>(old_whitened.cols());
  out->resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(dim));
  const uint32_t step = std::max<uint32_t>(1, block_rows);
  for (uint32_t begin = 0; begin < rows; begin += step) {
    const uint32_t count = std::min<uint32_t>(step, rows - begin);
    auto raw_res = whitening->InverseTransformBatch(
        old_whitened.middleRows(static_cast<Eigen::Index>(begin),
                                static_cast<Eigen::Index>(count)),
        old_version);
    if (!raw_res.ok()) {
      return raw_res.status();
    }
    auto new_res = whitening->TransformBatch(raw_res.value(), new_version);
    if (!new_res.ok()) {
      return new_res.status();
    }
    MatrixRM block = std::move(new_res.value());
    Status norm = NormalizeRowsIfNeeded(&block, use_cosine);
    if (!norm.ok()) {
      return norm;
    }
    out->middleRows(static_cast<Eigen::Index>(begin),
                    static_cast<Eigen::Index>(count)) = block;
  }
  return Status::OK();
}

struct MemoryComponent {
  std::string name;
  uint64_t bytes{0};
};

struct MemoryTraceEvent {
  uint32_t id{0};
  std::string stage;
  uint32_t active_rows{0};
  uint64_t process_rss_bytes{0};
  uint64_t process_peak_rss_bytes{0};
  uint64_t accounted_bytes{0};
  uint64_t transient_estimate_bytes{0};
  std::vector<MemoryComponent> components;
};

inline std::string JsonEscape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char ch : input) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

class MemoryTraceRecorder {
 public:
  void Record(std::string stage,
              uint32_t active_rows,
              std::vector<MemoryComponent> components,
              uint64_t transient_estimate_bytes = 0) {
    MemoryTraceEvent event;
    event.id = static_cast<uint32_t>(events_.size());
    event.stage = std::move(stage);
    event.active_rows = active_rows;
    event.process_rss_bytes = ReadProcStatusBytesLocal("VmRSS:");
    event.process_peak_rss_bytes = ReadProcStatusBytesLocal("VmHWM:");
    event.transient_estimate_bytes = transient_estimate_bytes;
    event.components = std::move(components);
    for (const auto& component : event.components) {
      event.accounted_bytes += component.bytes;
    }
    events_.push_back(std::move(event));
  }

  const std::vector<MemoryTraceEvent>& events() const {
    return events_;
  }

  Status WriteJson(const std::string& path,
                   const std::vector<MemoryComponent>& metadata) const {
    std::ofstream ofs(path);
    if (!ofs) {
      return Status::IOError("Failed to write memory trace: " + path);
    }
    uint64_t max_rss = 0;
    uint64_t max_peak_rss = 0;
    uint64_t max_accounted = 0;
    uint64_t max_transient = 0;
    for (const auto& event : events_) {
      max_rss = std::max(max_rss, event.process_rss_bytes);
      max_peak_rss = std::max(max_peak_rss, event.process_peak_rss_bytes);
      max_accounted = std::max(max_accounted, event.accounted_bytes);
      max_transient = std::max(max_transient, event.transient_estimate_bytes);
    }

    ofs << "{\n";
    ofs << "  \"summary\": {\n";
    ofs << "    \"event_count\": " << events_.size() << ",\n";
    ofs << "    \"max_process_rss_bytes\": " << max_rss << ",\n";
    ofs << "    \"max_process_peak_rss_bytes\": " << max_peak_rss << ",\n";
    ofs << "    \"max_accounted_bytes\": " << max_accounted << ",\n";
    ofs << "    \"max_transient_estimate_bytes\": " << max_transient << "\n";
    ofs << "  },\n";
    ofs << "  \"metadata\": {\n";
    for (size_t i = 0; i < metadata.size(); ++i) {
      ofs << "    \"" << JsonEscape(metadata[i].name) << "\": "
          << metadata[i].bytes;
      ofs << (i + 1 < metadata.size() ? ",\n" : "\n");
    }
    ofs << "  },\n";
    ofs << "  \"events\": [\n";
    for (size_t i = 0; i < events_.size(); ++i) {
      const auto& event = events_[i];
      ofs << "    {\n";
      ofs << "      \"id\": " << event.id << ",\n";
      ofs << "      \"stage\": \"" << JsonEscape(event.stage) << "\",\n";
      ofs << "      \"active_rows\": " << event.active_rows << ",\n";
      ofs << "      \"process_rss_bytes\": " << event.process_rss_bytes << ",\n";
      ofs << "      \"process_peak_rss_bytes\": " << event.process_peak_rss_bytes << ",\n";
      ofs << "      \"accounted_bytes\": " << event.accounted_bytes << ",\n";
      ofs << "      \"transient_estimate_bytes\": "
          << event.transient_estimate_bytes << ",\n";
      ofs << "      \"components\": {\n";
      for (size_t ci = 0; ci < event.components.size(); ++ci) {
        ofs << "        \"" << JsonEscape(event.components[ci].name) << "\": "
            << event.components[ci].bytes;
        ofs << (ci + 1 < event.components.size() ? ",\n" : "\n");
      }
      ofs << "      }\n";
      ofs << "    }" << (i + 1 < events_.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    return Status::OK();
  }

 private:
  std::vector<MemoryTraceEvent> events_;
};

inline void AddIVFMemoryComponents(const std::string& prefix,
                                   const std::shared_ptr<IVFIndex>& ivf,
                                   const VersionSet& versions,
                                   std::vector<MemoryComponent>* out) {
  if (out == nullptr || !ivf || versions.index_version == 0) {
    return;
  }
  auto usage_res = ivf->EstimateMemoryUsage(versions);
  if (!usage_res.ok()) {
    return;
  }
  const IVFMemoryUsage& u = usage_res.value();
  out->push_back({prefix + ".ivf.total", u.total_bytes});
  out->push_back({prefix + ".ivf.routing_centroids", u.routing_centroids_bytes});
  out->push_back({prefix + ".ivf.routing_centroid_norms", u.routing_centroid_norms_bytes});
  out->push_back({prefix + ".ivf.pq_codebooks", u.pq_codebooks_bytes});
  out->push_back({prefix + ".ivf.pq_codebooks_soa", u.pq_codebooks_soa_bytes});
  out->push_back({prefix + ".ivf.pq_precomputed_table", u.pq_precomputed_table_bytes});
  out->push_back({prefix + ".ivf.pq_counts", u.pq_counts_bytes});
  out->push_back({prefix + ".ivf.lists_vector", u.lists_vector_bytes});
  out->push_back({prefix + ".ivf.list_entry_structs", u.list_entry_struct_bytes});
  out->push_back({prefix + ".ivf.list_entry_vectors", u.list_entry_vectors_bytes});
  out->push_back({prefix + ".ivf.compact_doc_ids", u.compact_doc_ids_bytes});
  out->push_back({prefix + ".ivf.compact_pq_codes", u.compact_pq_codes_bytes});
  out->push_back({prefix + ".ivf.compact_pq_codes_soa", u.compact_pq_codes_soa_bytes});
  out->push_back({prefix + ".ivf.doc_to_list", u.doc_to_list_bytes});
}

}  // namespace ann::eval_memory
