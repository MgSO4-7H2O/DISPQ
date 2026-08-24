#include "soa_kernels.h"

#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ann::internal {
namespace {

[[maybe_unused]] constexpr uint32_t kAvx512Width = 16;
[[maybe_unused]] constexpr uint32_t kAvx2Width = 8;

[[maybe_unused]] uint32_t AssignPQSoAScalar(const float* subvector,
                                            uint32_t dsub,
                                            const float* codebook_soa,
                                            uint32_t ks,
                                            float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  for (uint32_t k = 0; k < ks; ++k) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      const float diff = subvector[d] - codebook_soa[d * ks + k];
      dist += diff * diff;
    }
    if (dist < best) {
      best = dist;
      best_id = k;
    }
  }
  if (best_distance != nullptr) {
    *best_distance = best;
  }
  return best_id;
}

#if defined(__AVX512F__)
uint32_t AssignPQSoAAvx512(const float* subvector,
                           uint32_t dsub,
                           const float* codebook_soa,
                           uint32_t ks,
                           float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;

  for (uint32_t c0 = 0; c0 < ks; c0 += kAvx512Width) {
    const uint32_t remaining = ks - c0;
    const __mmask16 lane_mask =
        (remaining >= kAvx512Width)
            ? static_cast<__mmask16>(0xffffu)
            : static_cast<__mmask16>((1u << remaining) - 1u);
    __m512 dist = _mm512_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m512 x = _mm512_set1_ps(subvector[d]);
      const __m512 c = _mm512_maskz_loadu_ps(lane_mask, codebook_soa + d * ks + c0);
      const __m512 diff = _mm512_sub_ps(x, c);
      dist = _mm512_fmadd_ps(diff, diff, dist);
    }

    dist = _mm512_mask_mov_ps(
        _mm512_set1_ps(std::numeric_limits<float>::max()), lane_mask, dist);
    const float block_best = _mm512_reduce_min_ps(dist);
    const __mmask16 equal_mask = _mm512_cmp_ps_mask(
        dist, _mm512_set1_ps(block_best), _CMP_EQ_OQ);
    const uint32_t lane = static_cast<uint32_t>(
        __builtin_ctz(static_cast<unsigned int>(equal_mask)));
    const uint32_t block_id = c0 + lane;
    if (block_best < best) {
      best = block_best;
      best_id = block_id;
    }
  }

  if (best_distance != nullptr) {
    *best_distance = best;
  }
  return best_id;
}
#endif

#if defined(__AVX2__)
[[maybe_unused]] uint32_t AssignPQSoAAvx2(const float* subvector,
                                          uint32_t dsub,
                                          const float* codebook_soa,
                                          uint32_t ks,
                                          float* best_distance) {
  alignas(32) float block[kAvx2Width];
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  uint32_t c0 = 0;

  for (; c0 + kAvx2Width <= ks; c0 += kAvx2Width) {
    __m256 dist = _mm256_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m256 x = _mm256_set1_ps(subvector[d]);
      const __m256 c = _mm256_loadu_ps(codebook_soa + d * ks + c0);
      const __m256 diff = _mm256_sub_ps(x, c);
      dist = _mm256_add_ps(dist, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(block, dist);
    for (uint32_t lane = 0; lane < kAvx2Width; ++lane) {
      if (block[lane] < best) {
        best = block[lane];
        best_id = c0 + lane;
      }
    }
  }

  for (; c0 < ks; ++c0) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      const float diff = subvector[d] - codebook_soa[d * ks + c0];
      dist += diff * diff;
    }
    if (dist < best) {
      best = dist;
      best_id = c0;
    }
  }

  if (best_distance != nullptr) {
    *best_distance = best;
  }
  return best_id;
}
#endif

[[maybe_unused]] void BuildPQDistanceTableSoAScalar(const float* subvector,
                                                    uint32_t dsub,
                                                    const float* codebook_soa,
                                                    uint32_t ks,
                                                    float* distances) {
  for (uint32_t k = 0; k < ks; ++k) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      const float diff = subvector[d] - codebook_soa[d * ks + k];
      dist += diff * diff;
    }
    distances[k] = dist;
  }
}

[[maybe_unused]] void BuildPQNeg2DotTableSoAScalar(const float* subvector,
                                                   uint32_t dsub,
                                                   const float* codebook_soa,
                                                   uint32_t ks,
                                                   float* neg2_dot) {
  for (uint32_t k = 0; k < ks; ++k) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      dot += subvector[d] * codebook_soa[d * ks + k];
    }
    neg2_dot[k] = -2.0f * dot;
  }
}

[[maybe_unused]] void ScanPQCodesSoAScalar(const float* distance_table,
                                           const uint8_t* codes_soa,
                                           uint32_t M,
                                           uint32_t Ks,
                                           size_t code_stride,
                                           size_t count,
                                           float* distances) {
  for (size_t i = 0; i < count; ++i) {
    float sum = 0.0f;
    for (uint32_t m = 0; m < M; ++m) {
      const uint8_t code = codes_soa[static_cast<size_t>(m) * code_stride + i];
      sum += distance_table[static_cast<size_t>(m) * Ks + code];
    }
    distances[i] = sum;
  }
}

#if defined(__AVX512F__)
void BuildPQDistanceTableSoAAvx512(const float* subvector,
                                   uint32_t dsub,
                                   const float* codebook_soa,
                                   uint32_t ks,
                                   float* distances) {
  for (uint32_t c0 = 0; c0 < ks; c0 += kAvx512Width) {
    const uint32_t remaining = ks - c0;
    const __mmask16 lane_mask =
        (remaining >= kAvx512Width)
            ? static_cast<__mmask16>(0xffffu)
            : static_cast<__mmask16>((1u << remaining) - 1u);
    __m512 dist = _mm512_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m512 x = _mm512_set1_ps(subvector[d]);
      const __m512 c = _mm512_maskz_loadu_ps(lane_mask, codebook_soa + d * ks + c0);
      const __m512 diff = _mm512_sub_ps(x, c);
      dist = _mm512_fmadd_ps(diff, diff, dist);
    }
    _mm512_mask_storeu_ps(distances + c0, lane_mask, dist);
  }
}

void BuildPQNeg2DotTableSoAAvx512(const float* subvector,
                                  uint32_t dsub,
                                  const float* codebook_soa,
                                  uint32_t ks,
                                  float* neg2_dot) {
  for (uint32_t c0 = 0; c0 < ks; c0 += kAvx512Width) {
    const uint32_t remaining = ks - c0;
    const __mmask16 lane_mask =
        (remaining >= kAvx512Width)
            ? static_cast<__mmask16>(0xffffu)
            : static_cast<__mmask16>((1u << remaining) - 1u);
    __m512 dot = _mm512_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m512 scale = _mm512_set1_ps(-2.0f * subvector[d]);
      const __m512 c = _mm512_maskz_loadu_ps(lane_mask, codebook_soa + d * ks + c0);
      dot = _mm512_fmadd_ps(scale, c, dot);
    }
    _mm512_mask_storeu_ps(neg2_dot + c0, lane_mask, dot);
  }
}
#endif

#if defined(__AVX2__)
[[maybe_unused]] void BuildPQDistanceTableSoAAvx2(const float* subvector,
                                                  uint32_t dsub,
                                                  const float* codebook_soa,
                                                  uint32_t ks,
                                                  float* distances) {
  uint32_t c0 = 0;
  for (; c0 + kAvx2Width <= ks; c0 += kAvx2Width) {
    __m256 dist = _mm256_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m256 x = _mm256_set1_ps(subvector[d]);
      const __m256 c = _mm256_loadu_ps(codebook_soa + d * ks + c0);
      const __m256 diff = _mm256_sub_ps(x, c);
      dist = _mm256_add_ps(dist, _mm256_mul_ps(diff, diff));
    }
    _mm256_storeu_ps(distances + c0, dist);
  }
  for (; c0 < ks; ++c0) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      const float diff = subvector[d] - codebook_soa[d * ks + c0];
      dist += diff * diff;
    }
    distances[c0] = dist;
  }
}

[[maybe_unused]] void BuildPQNeg2DotTableSoAAvx2(const float* subvector,
                                                 uint32_t dsub,
                                                 const float* codebook_soa,
                                                 uint32_t ks,
                                                 float* neg2_dot) {
  uint32_t c0 = 0;
  for (; c0 + kAvx2Width <= ks; c0 += kAvx2Width) {
    __m256 dot = _mm256_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m256 scale = _mm256_set1_ps(-2.0f * subvector[d]);
      const __m256 c = _mm256_loadu_ps(codebook_soa + d * ks + c0);
      dot = _mm256_add_ps(dot, _mm256_mul_ps(scale, c));
    }
    _mm256_storeu_ps(neg2_dot + c0, dot);
  }
  for (; c0 < ks; ++c0) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      dot += subvector[d] * codebook_soa[d * ks + c0];
    }
    neg2_dot[c0] = -2.0f * dot;
  }
}

void ScanPQCodesSoAAvx2(const float* distance_table,
                        const uint8_t* codes_soa,
                        uint32_t M,
                        uint32_t Ks,
                        size_t code_stride,
                        size_t count,
                        float* distances) {
  size_t i = 0;
  for (; i + kAvx2Width <= count; i += kAvx2Width) {
    __m256 sum = _mm256_setzero_ps();
    for (uint32_t m = 0; m < M; ++m) {
      uint64_t packed_codes = 0;
      std::memcpy(&packed_codes,
                  codes_soa + static_cast<size_t>(m) * code_stride + i,
                  sizeof(packed_codes));
      const __m128i code_bytes =
          _mm_cvtsi64_si128(static_cast<long long>(packed_codes));
      __m256i offsets = _mm256_cvtepu8_epi32(code_bytes);
      offsets = _mm256_add_epi32(offsets, _mm256_set1_epi32(static_cast<int>(m * Ks)));
      const __m256 values = _mm256_i32gather_ps(distance_table, offsets, sizeof(float));
      sum = _mm256_add_ps(sum, values);
    }
    _mm256_storeu_ps(distances + i, sum);
  }
  if (i < count) {
    ScanPQCodesSoAScalar(distance_table, codes_soa + i, M, Ks, code_stride, count - i,
                         distances + i);
  }
}
#endif

}  // namespace

uint32_t AssignPQSoA(const float* subvector,
                     uint32_t dsub,
                     const float* codebook_soa,
                     uint32_t ks,
                     float* best_distance) {
  if (subvector == nullptr || codebook_soa == nullptr || dsub == 0 || ks == 0) {
    if (best_distance != nullptr) {
      *best_distance = std::numeric_limits<float>::max();
    }
    return 0;
  }
#if defined(__AVX512F__)
  return AssignPQSoAAvx512(subvector, dsub, codebook_soa, ks, best_distance);
#elif defined(__AVX2__)
  return AssignPQSoAAvx2(subvector, dsub, codebook_soa, ks, best_distance);
#else
  return AssignPQSoAScalar(subvector, dsub, codebook_soa, ks, best_distance);
#endif
}

void BuildPQDistanceTableSoA(const float* subvector,
                             uint32_t dsub,
                             const float* codebook_soa,
                             uint32_t ks,
                             float* distances) {
  if (subvector == nullptr || codebook_soa == nullptr || distances == nullptr ||
      dsub == 0 || ks == 0) {
    return;
  }
#if defined(__AVX512F__)
  BuildPQDistanceTableSoAAvx512(subvector, dsub, codebook_soa, ks, distances);
#elif defined(__AVX2__)
  BuildPQDistanceTableSoAAvx2(subvector, dsub, codebook_soa, ks, distances);
#else
  BuildPQDistanceTableSoAScalar(subvector, dsub, codebook_soa, ks, distances);
#endif
}

void BuildPQNeg2DotTableSoA(const float* subvector,
                            uint32_t dsub,
                            const float* codebook_soa,
                            uint32_t ks,
                            float* neg2_dot) {
  if (subvector == nullptr || codebook_soa == nullptr || neg2_dot == nullptr ||
      dsub == 0 || ks == 0) {
    return;
  }
#if defined(__AVX512F__)
  BuildPQNeg2DotTableSoAAvx512(subvector, dsub, codebook_soa, ks, neg2_dot);
#elif defined(__AVX2__)
  BuildPQNeg2DotTableSoAAvx2(subvector, dsub, codebook_soa, ks, neg2_dot);
#else
  BuildPQNeg2DotTableSoAScalar(subvector, dsub, codebook_soa, ks, neg2_dot);
#endif
}

void ScanPQCodesSoA(const float* distance_table,
                    const uint8_t* codes_soa,
                    uint32_t M,
                    uint32_t Ks,
                    size_t code_stride,
                    size_t count,
                    float* distances) {
  if (distance_table == nullptr || codes_soa == nullptr || distances == nullptr ||
      M == 0 || Ks == 0 || count == 0 || code_stride < count) {
    return;
  }
#if defined(__AVX2__)
  ScanPQCodesSoAAvx2(distance_table, codes_soa, M, Ks, code_stride, count, distances);
#else
  ScanPQCodesSoAScalar(distance_table, codes_soa, M, Ks, code_stride, count, distances);
#endif
}

bool HasVectorizedPQCodeSoAScan() {
#if defined(__AVX2__)
  return true;
#else
  return false;
#endif
}

}  // namespace ann::internal
