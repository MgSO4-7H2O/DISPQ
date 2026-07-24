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

void CheckDistanceTables(uint32_t dsub, uint32_t ks) {
  std::mt19937 gen(4321 + dsub * 131 + ks);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> query(dsub);
  std::vector<float> soa(static_cast<size_t>(dsub) * ks);
  for (float& value : query) {
    value = dist(gen);
  }
  for (float& value : soa) {
    value = dist(gen);
  }

  std::vector<float> l2(ks, 0.0f);
  std::vector<float> neg2dot(ks, 0.0f);
  ann::internal::BuildPQDistanceTableSoA(query.data(), dsub, soa.data(), ks, l2.data());
  ann::internal::BuildPQNeg2DotTableSoA(query.data(), dsub, soa.data(), ks, neg2dot.data());
  for (uint32_t k = 0; k < ks; ++k) {
    float expected_l2 = 0.0f;
    float expected_neg2dot = 0.0f;
    for (uint32_t d = 0; d < dsub; ++d) {
      const float codebook_value = soa[static_cast<size_t>(d) * ks + k];
      const float diff = query[d] - codebook_value;
      expected_l2 += diff * diff;
      expected_neg2dot += query[d] * codebook_value;
    }
    expected_neg2dot *= -2.0f;
    assert(std::fabs(l2[k] - expected_l2) <=
           1e-5f * std::max(1.0f, expected_l2));
    assert(std::fabs(neg2dot[k] - expected_neg2dot) <=
           1e-5f * std::max(1.0f, std::fabs(expected_neg2dot)));
  }
}

void CheckCodeScan(uint32_t M, uint32_t ks, size_t count, size_t stride_extra) {
  std::mt19937 gen(9876 + M * 17 + ks + static_cast<uint32_t>(count));
  std::uniform_real_distribution<float> fdist(0.0f, 10.0f);
  std::uniform_int_distribution<uint32_t> cdist(0, ks - 1);
  const size_t stride = count + stride_extra;
  std::vector<float> table(static_cast<size_t>(M) * ks);
  std::vector<uint8_t> codes(static_cast<size_t>(M) * stride, 0);
  for (float& value : table) {
    value = fdist(gen);
  }
  for (uint32_t m = 0; m < M; ++m) {
    for (size_t i = 0; i < count; ++i) {
      codes[static_cast<size_t>(m) * stride + i] =
          static_cast<uint8_t>(cdist(gen));
    }
  }

  std::vector<float> actual(count, 0.0f);
  ann::internal::ScanPQCodesSoA(table.data(),
                                codes.data(),
                                M,
                                ks,
                                stride,
                                count,
                                actual.data());
  for (size_t i = 0; i < count; ++i) {
    float expected = 0.0f;
    for (uint32_t m = 0; m < M; ++m) {
      const uint8_t code = codes[static_cast<size_t>(m) * stride + i];
      expected += table[static_cast<size_t>(m) * ks + code];
    }
    assert(std::fabs(actual[i] - expected) <=
           1e-5f * std::max(1.0f, expected));
  }
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

  CheckDistanceTables(4, 256);
  CheckDistanceTables(7, 256);
  CheckDistanceTables(3, 17);
  CheckCodeScan(4, 256, 31, 0);
  CheckCodeScan(16, 256, 37, 5);
  return 0;
}
