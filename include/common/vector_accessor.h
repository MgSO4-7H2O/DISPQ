#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/status.h"
#include "common/types.h"

namespace ann {

inline constexpr size_t kVectorAccessorMaterializeChunkRows = 65536;

class VectorAccessor {
 public:
  virtual ~VectorAccessor() = default;
  virtual uint32_t dim() const = 0;
  virtual Result<Eigen::VectorXf> GetVector(DocId doc_id) const = 0;
  virtual Result<float> GetNorm(DocId doc_id) const {
    auto vector_res = GetVector(doc_id);
    if (!vector_res.ok()) {
      return vector_res.status();
    }
    return vector_res.value().squaredNorm();
  }

  virtual Status Materialize(const std::vector<DocId>& doc_ids,
                             MatrixRM* out) const {
    if (out == nullptr) {
      return Status::InvalidArgument("VectorAccessor::Materialize: null output");
    }
    if (doc_ids.size() > kVectorAccessorMaterializeChunkRows) {
      return Status::InvalidArgument("VectorAccessor::Materialize: chunk too large");
    }
    out->resize(static_cast<Eigen::Index>(doc_ids.size()), dim());
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      auto vector_res = GetVector(doc_ids[i]);
      if (!vector_res.ok()) {
        return vector_res.status();
      }
      if (static_cast<uint32_t>(vector_res.value().size()) != dim()) {
        return Status::InvalidArgument("VectorAccessor::Materialize: dimension mismatch");
      }
      out->row(static_cast<Eigen::Index>(i)) = vector_res.value().transpose();
    }
    return Status::OK();
  }
};

class MatrixVectorAccessor final : public VectorAccessor {
 public:
  explicit MatrixVectorAccessor(const MatrixRM& vectors)
      : vectors_(vectors) {}

  MatrixVectorAccessor(const MatrixRM& vectors, const Eigen::VectorXf& norms)
      : vectors_(vectors), norms_(&norms) {}

  uint32_t dim() const override {
    return static_cast<uint32_t>(vectors_.cols());
  }

  Result<Eigen::VectorXf> GetVector(DocId doc_id) const override {
    if (static_cast<Eigen::Index>(doc_id) >= vectors_.rows()) {
      return Status::InvalidArgument("MatrixVectorAccessor::GetVector: doc_id out of range");
    }
    return Eigen::VectorXf(
        vectors_.row(static_cast<Eigen::Index>(doc_id)).transpose());
  }

  Result<float> GetNorm(DocId doc_id) const override {
    if (static_cast<Eigen::Index>(doc_id) >= vectors_.rows()) {
      return Status::InvalidArgument("MatrixVectorAccessor::GetNorm: doc_id out of range");
    }
    if (norms_ != nullptr) {
      if (static_cast<Eigen::Index>(doc_id) >= norms_->size()) {
        return Status::InvalidArgument("MatrixVectorAccessor::GetNorm: norm out of range");
      }
      return (*norms_)(static_cast<Eigen::Index>(doc_id));
    }
    return vectors_.row(static_cast<Eigen::Index>(doc_id)).squaredNorm();
  }

  Status Materialize(const std::vector<DocId>& doc_ids,
                     MatrixRM* out) const override {
    if (out == nullptr) {
      return Status::InvalidArgument("MatrixVectorAccessor::Materialize: null output");
    }
    if (doc_ids.size() > kVectorAccessorMaterializeChunkRows) {
      return Status::InvalidArgument("MatrixVectorAccessor::Materialize: chunk too large");
    }
    out->resize(static_cast<Eigen::Index>(doc_ids.size()), vectors_.cols());
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      const DocId doc_id = doc_ids[i];
      if (static_cast<Eigen::Index>(doc_id) >= vectors_.rows()) {
        return Status::InvalidArgument(
            "MatrixVectorAccessor::Materialize: doc_id out of range");
      }
      out->row(static_cast<Eigen::Index>(i)) =
          vectors_.row(static_cast<Eigen::Index>(doc_id));
    }
    return Status::OK();
  }

 private:
  const MatrixRM& vectors_;
  const Eigen::VectorXf* norms_{nullptr};
};

}  // namespace ann
