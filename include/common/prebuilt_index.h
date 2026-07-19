#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct PrebuiltMainIndexMetadata {
  uint32_t format_version{1};
  uint32_t dim{0};
  uint32_t main_rows{0};
  uint32_t total_rows{0};
  VersionId whiten_version{0};
  VersionId index_version{0};
  std::string config_path;
  std::string dataset_path;
};

struct PrebuiltMainIndexArtifact {
  PrebuiltMainIndexMetadata metadata;
  std::vector<uint8_t> whitening_bytes;
  std::vector<uint8_t> index_bytes;
};

Result<void> SavePrebuiltMainIndex(const std::string& dir,
                                   const PrebuiltMainIndexArtifact& artifact);

Result<PrebuiltMainIndexArtifact> LoadPrebuiltMainIndex(const std::string& dir);

}  // namespace ann
