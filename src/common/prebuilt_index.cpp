#include "common/prebuilt_index.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#include "common/serialization.h"

namespace ann {
namespace {

constexpr uint32_t kPrebuiltIndexFormatVersion = 1;
constexpr const char* kManifestFileName = "manifest.json";
constexpr const char* kWhiteningFileName = "whitening.bin";
constexpr const char* kMainIndexFileName = "main_ivf.bin";

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (ch == '\\' || ch == '"') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  return out;
}

Result<std::string> ReadTextFile(const std::filesystem::path& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    return Status::IOError("Failed to open file for read: " + path.string());
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

Result<uint32_t> ExtractUint32(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return Status::InvalidArgument("Missing manifest field: " + key);
  }
  try {
    return static_cast<uint32_t>(std::stoul(match[1].str()));
  } catch (const std::exception&) {
    return Status::InvalidArgument("Invalid manifest integer field: " + key);
  }
}

Result<std::string> ExtractString(const std::string& text, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return Status::InvalidArgument("Missing manifest field: " + key);
  }
  return match[1].str();
}

}  // namespace

Result<void> SavePrebuiltMainIndex(const std::string& dir,
                                   const PrebuiltMainIndexArtifact& artifact) {
  namespace fs = std::filesystem;
  const fs::path root(dir);
  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    return Status::IOError("Failed to create index directory: " + root.string() +
                           ": " + ec.message());
  }

  const auto whitening_path = root / kWhiteningFileName;
  auto save_whitening = SaveBinary(whitening_path.string(), artifact.whitening_bytes);
  if (!save_whitening.ok()) {
    return save_whitening.status();
  }

  const auto index_path = root / kMainIndexFileName;
  auto save_index = SaveBinary(index_path.string(), artifact.index_bytes);
  if (!save_index.ok()) {
    return save_index.status();
  }

  const auto manifest_path = root / kManifestFileName;
  std::ofstream ofs(manifest_path, std::ios::trunc);
  if (!ofs) {
    return Status::IOError("Failed to open file for write: " + manifest_path.string());
  }
  ofs << "{\n"
      << "  \"format\": \"sa-wrq-prebuilt-main-index\",\n"
      << "  \"format_version\": " << kPrebuiltIndexFormatVersion << ",\n"
      << "  \"dim\": " << artifact.metadata.dim << ",\n"
      << "  \"main_rows\": " << artifact.metadata.main_rows << ",\n"
      << "  \"total_rows\": " << artifact.metadata.total_rows << ",\n"
      << "  \"whiten_version\": " << artifact.metadata.whiten_version << ",\n"
      << "  \"index_version\": " << artifact.metadata.index_version << ",\n"
      << "  \"config_path\": \"" << JsonEscape(artifact.metadata.config_path) << "\",\n"
      << "  \"dataset_path\": \"" << JsonEscape(artifact.metadata.dataset_path) << "\"\n"
      << "}\n";
  if (!ofs) {
    return Status::IOError("Failed to write file: " + manifest_path.string());
  }
  return Result<void>();
}

Result<PrebuiltMainIndexArtifact> LoadPrebuiltMainIndex(const std::string& dir) {
  namespace fs = std::filesystem;
  const fs::path root(dir);
  auto manifest_res = ReadTextFile(root / kManifestFileName);
  if (!manifest_res.ok()) {
    return manifest_res.status();
  }
  const std::string& manifest = manifest_res.value();

  PrebuiltMainIndexArtifact artifact;
  auto format_version = ExtractUint32(manifest, "format_version");
  if (!format_version.ok()) {
    return format_version.status();
  }
  artifact.metadata.format_version = format_version.value();
  if (artifact.metadata.format_version != kPrebuiltIndexFormatVersion) {
    return Status::InvalidArgument("Unsupported prebuilt index format_version");
  }

  auto dim = ExtractUint32(manifest, "dim");
  auto main_rows = ExtractUint32(manifest, "main_rows");
  auto total_rows = ExtractUint32(manifest, "total_rows");
  auto whiten_version = ExtractUint32(manifest, "whiten_version");
  auto index_version = ExtractUint32(manifest, "index_version");
  auto config_path = ExtractString(manifest, "config_path");
  auto dataset_path = ExtractString(manifest, "dataset_path");
  if (!dim.ok()) return dim.status();
  if (!main_rows.ok()) return main_rows.status();
  if (!total_rows.ok()) return total_rows.status();
  if (!whiten_version.ok()) return whiten_version.status();
  if (!index_version.ok()) return index_version.status();
  if (!config_path.ok()) return config_path.status();
  if (!dataset_path.ok()) return dataset_path.status();

  artifact.metadata.dim = dim.value();
  artifact.metadata.main_rows = main_rows.value();
  artifact.metadata.total_rows = total_rows.value();
  artifact.metadata.whiten_version = whiten_version.value();
  artifact.metadata.index_version = index_version.value();
  artifact.metadata.config_path = config_path.value();
  artifact.metadata.dataset_path = dataset_path.value();

  auto whitening_res = LoadBinary((root / kWhiteningFileName).string());
  if (!whitening_res.ok()) {
    return whitening_res.status();
  }
  artifact.whitening_bytes = std::move(whitening_res.value());

  auto index_res = LoadBinary((root / kMainIndexFileName).string());
  if (!index_res.ok()) {
    return index_res.status();
  }
  artifact.index_bytes = std::move(index_res.value());
  return artifact;
}

}  // namespace ann
