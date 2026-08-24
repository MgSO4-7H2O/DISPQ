#pragma once

#include <cstddef>
#include <cstdint>

namespace ann::internal {

uint32_t AssignPQSoA(const float* subvector,
                     uint32_t dsub,
                     const float* codebook_soa,
                     uint32_t ks,
                     float* best_distance);

void BuildPQDistanceTableSoA(const float* subvector,
                             uint32_t dsub,
                             const float* codebook_soa,
                             uint32_t ks,
                             float* distances);

void BuildPQNeg2DotTableSoA(const float* subvector,
                            uint32_t dsub,
                            const float* codebook_soa,
                            uint32_t ks,
                            float* neg2_dot);

void ScanPQCodesSoA(const float* distance_table,
                    const uint8_t* codes_soa,
                    uint32_t M,
                    uint32_t Ks,
                    size_t code_stride,
                    size_t count,
                    float* distances);

bool HasVectorizedPQCodeSoAScan();

}  // namespace ann::internal
