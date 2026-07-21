#pragma once

#include <memory>
#include <shared_mutex>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

class WhiteningModel {
 public:
  virtual ~WhiteningModel() = default;

  // NTS: Fits statistics for a new version using batched data (X shape: n x dim, row-major).
  virtual Result<VersionId> Fit(Eigen::Ref<const MatrixRM> X) = 0;

  // TS: Applies the whitening transform corresponding to version -> out_xw.
  virtual Result<void> Transform(Eigen::Ref<const Eigen::VectorXf> x,
                                 VersionId version,
                                 Eigen::Ref<Eigen::VectorXf> out_xw) const = 0;

  // TS: Applies whitening transform to a batch matrix (each row a vector).
  virtual Result<MatrixRM> TransformBatch(Eigen::Ref<const MatrixRM> X,
                                          VersionId version) const = 0;

  // NTS: Updates running stats using a single sample.
  virtual Status UpdateStats(Eigen::Ref<const Eigen::VectorXf> x) = 0;

  // NTS: Finalizes accumulated stats into a new immutable version id.
  virtual Result<VersionId> FinalizeNewVersion() = 0;

  // TS: Returns bridge matrix that maps vectors from src -> dst whitening spaces (dim x dim, row-major).
  virtual Result<MatrixRM> Bridge(VersionId src_version, VersionId dst_version) const = 0;

  virtual Result<std::vector<uint8_t>> Serialize() const = 0;
  virtual Status Deserialize(const std::vector<uint8_t>& bytes) = 0;
};

std::shared_ptr<WhiteningModel> CreateWhiteningModel();

}  // namespace ann
