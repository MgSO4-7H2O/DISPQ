#pragma once

#include <cstddef>
#include <cstdint>

namespace ann::internal {

uint32_t AssignPQSoA8x256(const float* subvector,
                          const float* codebook_soa,
                          float* best_distance);

}  // namespace ann::internal
