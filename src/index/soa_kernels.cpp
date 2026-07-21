#include "soa_kernels.h"

#include <limits>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace ann::internal {
namespace {

constexpr uint32_t kSimdWidth = 16;
constexpr uint32_t kFixedKs = 256;

#if !defined(__AVX512F__)
template <uint32_t Dsub>
uint32_t AssignPQSoAScalarFixed(const float* subvector,
                                const float* codebook_soa,
                                float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  for (uint32_t k = 0; k < kFixedKs; ++k) {
    float dist = 0.0f;
    for (uint32_t d = 0; d < Dsub; ++d) {
      const float diff = subvector[d] - codebook_soa[d * kFixedKs + k];
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

uint32_t AssignPQSoAScalar(const float* subvector,
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
#endif

#if defined(__AVX512F__)
template <uint32_t Dsub>
uint32_t AssignPQSoAAvx512Fixed(const float* subvector,
                                const float* codebook_soa,
                                float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  for (uint32_t c0 = 0; c0 < kFixedKs; c0 += kSimdWidth) {
    __m512 dist = _mm512_setzero_ps();
    for (uint32_t d = 0; d < Dsub; ++d) {
      const __m512 x = _mm512_set1_ps(subvector[d]);
      const __m512 c = _mm512_loadu_ps(codebook_soa + d * kFixedKs + c0);
      const __m512 diff = _mm512_sub_ps(x, c);
      dist = _mm512_fmadd_ps(diff, diff, dist);
    }
    const float block_best = _mm512_reduce_min_ps(dist);
    const __mmask16 equal_mask =
        _mm512_cmp_ps_mask(dist, _mm512_set1_ps(block_best), _CMP_EQ_OQ);
    const uint32_t lane = static_cast<uint32_t>(
        __builtin_ctz(static_cast<unsigned int>(equal_mask)));
    if (block_best < best) {
      best = block_best;
      best_id = c0 + lane;
    }
  }
  if (best_distance != nullptr) {
    *best_distance = best;
  }
  return best_id;
}

uint32_t AssignPQSoAAvx512(const float* subvector,
                           uint32_t dsub,
                           const float* codebook_soa,
                           uint32_t ks,
                           float* best_distance) {
  float best = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  for (uint32_t c0 = 0; c0 < ks; c0 += kSimdWidth) {
    const uint32_t remaining = ks - c0;
    const __mmask16 lane_mask =
        (remaining >= kSimdWidth)
            ? static_cast<__mmask16>(0xffffu)
            : static_cast<__mmask16>((1u << remaining) - 1u);
    __m512 dist = _mm512_setzero_ps();
    for (uint32_t d = 0; d < dsub; ++d) {
      const __m512 x = _mm512_set1_ps(subvector[d]);
      const __m512 c =
          _mm512_maskz_loadu_ps(lane_mask, codebook_soa + d * ks + c0);
      const __m512 diff = _mm512_sub_ps(x, c);
      dist = _mm512_fmadd_ps(diff, diff, dist);
    }
    dist = _mm512_mask_mov_ps(
        _mm512_set1_ps(std::numeric_limits<float>::max()), lane_mask, dist);
    const float block_best = _mm512_reduce_min_ps(dist);
    const __mmask16 equal_mask =
        _mm512_cmp_ps_mask(dist, _mm512_set1_ps(block_best), _CMP_EQ_OQ);
    const uint32_t lane = static_cast<uint32_t>(
        __builtin_ctz(static_cast<unsigned int>(equal_mask)));
    if (block_best < best) {
      best = block_best;
      best_id = c0 + lane;
    }
  }
  if (best_distance != nullptr) {
    *best_distance = best;
  }
  return best_id;
}
#endif

template <uint32_t Dsub>
uint32_t AssignPQSoAFixed(const float* subvector,
                          const float* codebook_soa,
                          float* best_distance) {
#if defined(__AVX512F__)
  return AssignPQSoAAvx512Fixed<Dsub>(subvector, codebook_soa, best_distance);
#else
  return AssignPQSoAScalarFixed<Dsub>(subvector, codebook_soa, best_distance);
#endif
}

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
  if (ks == kFixedKs) {
    if (dsub == 8) {
      return AssignPQSoAFixed<8>(subvector, codebook_soa, best_distance);
    }
    if (dsub == 4) {
      return AssignPQSoAFixed<4>(subvector, codebook_soa, best_distance);
    }
  }
#if defined(__AVX512F__)
  return AssignPQSoAAvx512(subvector, dsub, codebook_soa, ks, best_distance);
#else
  return AssignPQSoAScalar(subvector, dsub, codebook_soa, ks, best_distance);
#endif
}

}  // namespace ann::internal
