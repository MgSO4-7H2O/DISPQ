#pragma once

#include <string>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct FvecsFileMeta {
  uint32_t rows{0};
  uint32_t dim{0};
};

// Loads an .fvecs file into a row-major matrix (rows = #vectors, cols = dim).
Result<MatrixRM> LoadFvecs(const std::string& path);

// Reads fvecs metadata without loading all vector payloads.
Result<FvecsFileMeta> InspectFvecs(const std::string& path);

// Loads a contiguous row range from an .fvecs file.
Result<MatrixRM> LoadFvecsRange(const std::string& path,
                                uint32_t begin,
                                uint32_t count);

// Resolves a user supplied file or directory path into a concrete file whose
// name ends with the provided suffix (e.g. "_base.fvecs").
Result<std::string> ResolveFvecsPath(const std::string& path_or_dir, const std::string& required_suffix);

}  // namespace ann
