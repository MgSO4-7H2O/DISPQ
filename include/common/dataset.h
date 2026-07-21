#pragma once

#include <string>

#include "common/result.h"
#include "common/types.h"

namespace ann {

// Loads an .fvecs file into a row-major matrix (rows = #vectors, cols = dim).
Result<MatrixRM> LoadFvecs(const std::string& path);

// Resolves a user supplied file or directory path into a concrete file whose
// name ends with the provided suffix (e.g. "_base.fvecs").
Result<std::string> ResolveFvecsPath(const std::string& path_or_dir, const std::string& required_suffix);

}  // namespace ann

