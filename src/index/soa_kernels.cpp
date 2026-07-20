#include "soa_kernels.h"

#include <cstdint>
#include <limits>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace ann::internal {
namespace {

constexpr uint32_t kDsub = 8;
constexpr uint32_t kKs = 256;
constexpr uint32_t kSimdWidth = 16;

#if !defined(__AVX512F__)
uint32_t AssignPQSoAScalar(const float* subvector,
                           const float* codebook_soa,
                           float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  for (uint32_t k = 0; k < kKs; ++k) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < kDsub; ++d) {
      const float diff = subvector[d] - codebook_soa[d * kKs + k];
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
#endif

#if defined(__AVX512F__)
uint32_t AssignPQSoAAvx512(const float* subvector,
                           const float* codebook_soa,
                           float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;

  for (uint32_t c0 = 0; c0 < kKs; c0 += kSimdWidth) {
    __m512 dist = _mm512_setzero_ps();
    for (uint32_t d = 0; d < kDsub; ++d) {
      const __m512 x = _mm512_set1_ps(subvector[d]);
      const __m512 c = _mm512_loadu_ps(codebook_soa + d * kKs + c0);
      const __m512 diff = _mm512_sub_ps(x, c);
      dist = _mm512_fmadd_ps(diff, diff, dist);
    }

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

}  // namespace

uint32_t AssignPQSoA8x256(const float* subvector,
                          const float* codebook_soa,
                          float* best_distance) {
#if defined(__AVX512F__)
  return AssignPQSoAAvx512(subvector, codebook_soa, best_distance);
#else
  return AssignPQSoAScalar(subvector, codebook_soa, best_distance);
#endif
}

}  // namespace ann::internal
