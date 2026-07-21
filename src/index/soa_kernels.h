#pragma once

#include <cstdint>

namespace ann::internal {

uint32_t AssignPQSoA(const float* subvector,
                     uint32_t dsub,
                     const float* codebook_soa,
                     uint32_t ks,
                     float* best_distance);

}  // namespace ann::internal
