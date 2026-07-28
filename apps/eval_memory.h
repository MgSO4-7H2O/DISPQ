#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/status.h"
#include "common/types.h"
#include "index/ivf.h"

namespace ann::eval_memory {

struct MemoryComponent {
  std::string name;
  uint64_t bytes{0};
};

inline uint64_t MatrixActiveBytes(uint64_t rows, uint64_t cols) {
  return rows * cols * static_cast<uint64_t>(sizeof(float));
}

inline uint64_t MatrixBytes(const MatrixRM& matrix) {
  return MatrixActiveBytes(static_cast<uint64_t>(matrix.rows()),
                           static_cast<uint64_t>(matrix.cols()));
}

inline uint64_t VectorBytes(const Eigen::VectorXf& vector) {
  return static_cast<uint64_t>(vector.size()) * static_cast<uint64_t>(sizeof(float));
}

template <typename T>
inline uint64_t StdVectorBytes(const std::vector<T>& vector) {
  return static_cast<uint64_t>(vector.capacity()) * static_cast<uint64_t>(sizeof(T));
}

inline uint64_t ReadProcStatusBytes(const std::string& key) {
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

inline std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

inline void AddIVFMemoryComponents(const std::string& prefix,
                                   const std::shared_ptr<IVFIndex>& index,
                                   const VersionSet& versions,
                                   std::vector<MemoryComponent>* components) {
  if (!index || components == nullptr || versions.index_version == 0) {
    return;
  }
  auto usage_res = index->EstimateMemoryUsage(versions);
  if (!usage_res.ok()) {
    components->push_back({prefix + ".ivf.estimate_error", 0});
    return;
  }
  const IVFMemoryUsage& usage = usage_res.value();
  components->push_back({prefix + ".ivf.routing_centroids", usage.routing_centroids_bytes});
  components->push_back(
      {prefix + ".ivf.routing_centroid_norms", usage.routing_centroid_norms_bytes});
  components->push_back({prefix + ".ivf.pq_codebooks", usage.pq_codebooks_bytes});
  components->push_back({prefix + ".ivf.pq_codebooks_soa", usage.pq_codebooks_soa_bytes});
  components->push_back({prefix + ".ivf.pq_counts", usage.pq_counts_bytes});
  components->push_back({prefix + ".ivf.pq_precomputed_table", usage.pq_precomputed_table_bytes});
  components->push_back({prefix + ".ivf.list_entries", usage.list_entries_bytes});
  components->push_back({prefix + ".ivf.list_vectors", usage.list_vectors_bytes});
  components->push_back({prefix + ".ivf.list_pq_codes", usage.list_pq_codes_bytes});
  components->push_back({prefix + ".ivf.compact_doc_ids", usage.compact_doc_ids_bytes});
  components->push_back({prefix + ".ivf.compact_pq_codes", usage.compact_pq_codes_bytes});
  components->push_back({prefix + ".ivf.soa_pq_codes", usage.soa_pq_codes_bytes});
  components->push_back({prefix + ".ivf.doc_locations", usage.doc_locations_bytes});
  components->push_back({prefix + ".ivf.container_overhead", usage.container_overhead_bytes});
  components->push_back({prefix + ".ivf.total", usage.total_bytes});
}

struct MemoryTraceEvent {
  std::string stage;
  uint64_t active_rows{0};
  uint64_t rss_bytes{0};
  uint64_t peak_rss_bytes{0};
  uint64_t estimated_component_bytes{0};
  uint64_t transient_bytes{0};
  std::vector<MemoryComponent> components;
};

class MemoryTraceRecorder {
 public:
  void Record(const std::string& stage,
              uint64_t active_rows,
              std::vector<MemoryComponent> components,
              uint64_t transient_bytes = 0) {
    MemoryTraceEvent event;
    event.stage = stage;
    event.active_rows = active_rows;
    event.rss_bytes = ReadProcStatusBytes("VmRSS:");
    event.peak_rss_bytes = ReadProcStatusBytes("VmHWM:");
    event.transient_bytes = transient_bytes;
    event.components = std::move(components);
    for (const MemoryComponent& component : event.components) {
      event.estimated_component_bytes += component.bytes;
    }
    event.estimated_component_bytes += transient_bytes;
    events_.push_back(std::move(event));
  }

  Status WriteJson(const std::string& path,
                   const std::vector<MemoryComponent>& metadata) const {
    std::filesystem::path out_path(path);
    if (!out_path.parent_path().empty()) {
      std::error_code ec;
      std::filesystem::create_directories(out_path.parent_path(), ec);
    }
    std::ofstream ofs(path);
    if (!ofs) {
      return Status::IOError("Failed to write memory trace: " + path);
    }
    ofs << "{\n";
    ofs << "  \"metadata\": {\n";
    for (size_t i = 0; i < metadata.size(); ++i) {
      ofs << "    \"" << JsonEscape(metadata[i].name) << "\": " << metadata[i].bytes;
      ofs << (i + 1 < metadata.size() ? ",\n" : "\n");
    }
    ofs << "  },\n";
    ofs << "  \"events\": [\n";
    for (size_t i = 0; i < events_.size(); ++i) {
      const MemoryTraceEvent& event = events_[i];
      ofs << "    {\n";
      ofs << "      \"stage\": \"" << JsonEscape(event.stage) << "\",\n";
      ofs << "      \"active_rows\": " << event.active_rows << ",\n";
      ofs << "      \"rss_bytes\": " << event.rss_bytes << ",\n";
      ofs << "      \"peak_rss_bytes\": " << event.peak_rss_bytes << ",\n";
      ofs << "      \"estimated_component_bytes\": " << event.estimated_component_bytes
          << ",\n";
      ofs << "      \"transient_bytes\": " << event.transient_bytes << ",\n";
      ofs << "      \"components\": {\n";
      for (size_t j = 0; j < event.components.size(); ++j) {
        ofs << "        \"" << JsonEscape(event.components[j].name)
            << "\": " << event.components[j].bytes;
        ofs << (j + 1 < event.components.size() ? ",\n" : "\n");
      }
      ofs << "      }\n";
      ofs << "    }";
      ofs << (i + 1 < events_.size() ? ",\n" : "\n");
    }
    ofs << "  ]\n";
    ofs << "}\n";
    return Status::OK();
  }

 private:
  std::vector<MemoryTraceEvent> events_;
};

}  // namespace ann::eval_memory
