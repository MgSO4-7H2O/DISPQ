#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/status.h"
#include "common/types.h"

namespace ann {

class VectorAccessor {
 public:
  virtual ~VectorAccessor() = default;
  virtual uint32_t dim() const = 0;
  virtual Result<Eigen::VectorXf> GetVector(DocId doc_id) const = 0;
  virtual Result<float> GetNorm(DocId doc_id) const {
    auto vec_res = GetVector(doc_id);
    if (!vec_res.ok()) {
      return vec_res.status();
    }
    return vec_res.value().squaredNorm();
  }

  virtual Status Materialize(const std::vector<DocId>& doc_ids, MatrixRM* out) const {
    if (out == nullptr) {
      return Status::InvalidArgument("VectorAccessor::Materialize: null output");
    }
    MatrixRM rows(static_cast<Eigen::Index>(doc_ids.size()), dim());
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      auto vec_res = GetVector(doc_ids[i]);
      if (!vec_res.ok()) {
        return vec_res.status();
      }
      if (static_cast<uint32_t>(vec_res.value().size()) != dim()) {
        return Status::InvalidArgument("VectorAccessor::Materialize: dimension mismatch");
      }
      rows.row(static_cast<Eigen::Index>(i)) = vec_res.value().transpose();
    }
    *out = std::move(rows);
    return Status::OK();
  }
};

class MatrixVectorAccessor final : public VectorAccessor {
 public:
  explicit MatrixVectorAccessor(const MatrixRM& vectors) : vectors_(vectors) {}

  uint32_t dim() const override {
    return static_cast<uint32_t>(vectors_.cols());
  }

  Result<Eigen::VectorXf> GetVector(DocId doc_id) const override {
    if (static_cast<Eigen::Index>(doc_id) >= vectors_.rows()) {
      return Status::InvalidArgument("MatrixVectorAccessor: doc_id out of range");
    }
    return Eigen::VectorXf(vectors_.row(static_cast<Eigen::Index>(doc_id)).transpose());
  }

  Result<float> GetNorm(DocId doc_id) const override {
    if (static_cast<Eigen::Index>(doc_id) >= vectors_.rows()) {
      return Status::InvalidArgument("MatrixVectorAccessor: doc_id out of range");
    }
    return vectors_.row(static_cast<Eigen::Index>(doc_id)).squaredNorm();
  }

  Status Materialize(const std::vector<DocId>& doc_ids, MatrixRM* out) const override {
    if (out == nullptr) {
      return Status::InvalidArgument("MatrixVectorAccessor::Materialize: null output");
    }
    MatrixRM rows(static_cast<Eigen::Index>(doc_ids.size()), vectors_.cols());
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      const DocId doc_id = doc_ids[i];
      if (static_cast<Eigen::Index>(doc_id) >= vectors_.rows()) {
        return Status::InvalidArgument("MatrixVectorAccessor::Materialize: doc_id out of range");
      }
      rows.row(static_cast<Eigen::Index>(i)) =
          vectors_.row(static_cast<Eigen::Index>(doc_id));
    }
    *out = std::move(rows);
    return Status::OK();
  }

 private:
  const MatrixRM& vectors_;
};

}  // namespace ann
