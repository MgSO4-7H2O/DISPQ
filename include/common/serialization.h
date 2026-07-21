#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/result.h"

namespace ann {

Result<void> SaveBinary(const std::string& path, const std::vector<uint8_t>& bytes);
Result<std::vector<uint8_t>> LoadBinary(const std::string& path);

}  // namespace ann
