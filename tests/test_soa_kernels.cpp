#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "soa_kernels.h"

namespace {

void CheckKernel(uint32_t dsub, uint32_t ks) {
  std::mt19937 gen(1234 + dsub * 257 + ks);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> query(dsub);
  std::vector<float> aos(static_cast<size_t>(ks) * dsub);
  std::vector<float> soa(static_cast<size_t>(dsub) * ks);
  for (float& value : query) {
    value = dist(gen);
  }
  for (uint32_t k = 0; k < ks; ++k) {
    for (uint32_t d = 0; d < dsub; ++d) {
      const float value = dist(gen);
      aos[static_cast<size_t>(k) * dsub + d] = value;
      soa[static_cast<size_t>(d) * ks + k] = value;
    }
  }

  float expected_dist = std::numeric_limits<float>::max();
  uint32_t expected_id = 0;
  for (uint32_t k = 0; k < ks; ++k) {
    float value = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      const float diff = query[d] - aos[static_cast<size_t>(k) * dsub + d];
      value += diff * diff;
    }
    if (value < expected_dist) {
      expected_dist = value;
      expected_id = k;
    }
  }

  float actual_dist = 0.0f;
  assert(ann::internal::AssignPQSoA(
             query.data(), dsub, soa.data(), ks, &actual_dist) == expected_id);
  assert(std::fabs(actual_dist - expected_dist) <=
         1e-5f * std::max(1.0f, expected_dist));

  // Exact duplicate codewords must retain the scalar first-id tie break.
  for (uint32_t d = 0; d < dsub; ++d) {
    soa[static_cast<size_t>(d) * ks + 3] = query[d];
    soa[static_cast<size_t>(d) * ks + 5] = query[d];
  }
  assert(ann::internal::AssignPQSoA(query.data(), dsub, soa.data(), ks, nullptr) == 3);
}

}  // namespace

int main() {
  CheckKernel(4, 256);   // Fixed 4x256 path.
  CheckKernel(8, 256);   // Fixed 8x256 path.
  CheckKernel(6, 256);   // Generic MSTuring subspace.
  CheckKernel(7, 256);   // Generic MSTuring subspace.
  CheckKernel(2, 16);    // Generic short-tail path.
  CheckKernel(3, 32);    // Generic multi-block path.
  CheckKernel(5, 17);    // Generic non-power-of-two tail.

  float best_distance = 0.0f;
  assert(ann::internal::AssignPQSoA(nullptr, 4, nullptr, 256, &best_distance) == 0);
  assert(best_distance == std::numeric_limits<float>::max());
  return 0;
}
