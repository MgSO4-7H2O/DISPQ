#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace ann {

using DocId = uint32_t;
using VersionId = uint32_t;

using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

struct VersionSet {
  VersionId whiten_version{0};
  VersionId codebook_version{0};
  VersionId index_version{0};
};

struct Candidate {
  DocId doc_id{0};
  float approx_dist{0.0f};
  float rerank_dist{0.0f};
  VersionSet versions{};
  uint8_t from_new{0};
};

struct SearchResult {
  std::vector<Candidate> topk;
  uint64_t scanned_candidates{0};
};

struct VectorRecord {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DocId doc_id{0};
  uint32_t dim{0};
  VersionSet versions{};
  uint32_t ivf_id{0};
  std::vector<uint32_t> codes;
  Eigen::VectorXf x;
};

}  // namespace ann
