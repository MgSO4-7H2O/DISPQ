#include "common/dataset.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace ann {
namespace {

bool EndsWith(const std::string& value, const std::string& suffix) {
  if (suffix.size() > value.size()) {
    return false;
  }
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

Result<MatrixRM> LoadFvecs(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return Status::IOError("Failed to open fvecs file: " + path);
  }

  std::vector<float> values;
  int32_t dim = -1;
  size_t num_vecs = 0;
  while (true) {
    int32_t cur_dim = 0;
    ifs.read(reinterpret_cast<char*>(&cur_dim), sizeof(int32_t));
    if (!ifs) {
      if (ifs.eof() && ifs.gcount() == 0) {
        break;  // Finished reading cleanly.
      }
      return Status::IOError("Failed to read vector dimension from: " + path);
    }
    if (cur_dim <= 0) {
      return Status::InvalidArgument("Invalid dimension in fvecs file: " + path);
    }
    if (dim == -1) {
      dim = cur_dim;
    } else if (cur_dim != dim) {
      return Status::InvalidArgument("Mixed dimensions in fvecs file: " + path);
    }
    std::vector<float> buffer(cur_dim);
    ifs.read(reinterpret_cast<char*>(buffer.data()), sizeof(float) * cur_dim);
    if (!ifs) {
      return Status::IOError("Failed to read vector payload from: " + path);
    }
    values.insert(values.end(), buffer.begin(), buffer.end());
    ++num_vecs;
  }

  if (dim <= 0 || num_vecs == 0) {
    return Status::InvalidArgument("fvecs file contains no vectors: " + path);
  }

  MatrixRM matrix(num_vecs, dim);
  size_t offset = 0;
  for (size_t i = 0; i < num_vecs; ++i) {
    for (int32_t j = 0; j < dim; ++j) {
      matrix(i, j) = values[offset++];
    }
  }
  return matrix;
}

Result<std::string> ResolveFvecsPath(const std::string& path_or_dir,
                                     const std::string& required_suffix) {
  namespace fs = std::filesystem;
  fs::path candidate(path_or_dir);
  std::error_code ec;

  if (!fs::exists(candidate, ec)) {
    return Status::NotFound("Path does not exist: " + path_or_dir);
  }

  if (fs::is_regular_file(candidate, ec)) {
    const std::string file_name = candidate.filename().string();
    if (!required_suffix.empty() && !EndsWith(file_name, required_suffix)) {
      return Status::InvalidArgument("Expected a file ending with " + required_suffix + ": " + file_name);
    }
    return candidate.string();
  }

  if (fs::is_directory(candidate, ec)) {
    for (const auto& entry : fs::directory_iterator(candidate, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string name = entry.path().filename().string();
      if (!required_suffix.empty() && EndsWith(name, required_suffix)) {
        return entry.path().string();
      }
    }
    return Status::NotFound("No file ending with " + required_suffix + " found under " + path_or_dir);
  }

  return Status::InvalidArgument("Unsupported path type: " + path_or_dir);
}

}  // namespace ann
