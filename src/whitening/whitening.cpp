#include "whitening/whitening.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include <Eigen/Eigenvalues>

namespace ann {
namespace {

constexpr float kDefaultEpsilon = 0;

struct VersionPayload {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VersionId version{0};
  uint32_t dim{0};
  float epsilon{kDefaultEpsilon};
  Eigen::VectorXf mean;
  MatrixRM transform;
  MatrixRM transform_inv;
};

class ZCAWhiteningModel : public WhiteningModel {
 public:
  Result<VersionId> Fit(Eigen::Ref<const MatrixRM> X) override {
    if (X.rows() == 0 || X.cols() == 0) {
      return Status::InvalidArgument("Input matrix must be non-empty");
    }
    VersionPayload payload;
    auto status = ComputeZCATransform(X, &payload);
    if (!status.ok()) {
      return status;
    }
    std::unique_lock lock(mu_);
    VersionId version = next_version_++;
    payload.version = version;
    payload.dim = static_cast<uint32_t>(X.cols());
    versions_[version] = std::move(payload);
    dim_ = versions_[version].dim;
    ResetPending();
    return version;
  }

  Result<void> Transform(Eigen::Ref<const Eigen::VectorXf> x,
                         VersionId version,
                         Eigen::Ref<Eigen::VectorXf> out_xw) const override {
    std::shared_lock lock(mu_);
    auto it = versions_.find(version);
    if (it == versions_.end()) {
      return Status::NotFound("Whitening version missing");
    }
    const VersionPayload& payload = it->second;
    if (x.size() != payload.mean.size() || out_xw.size() != x.size()) {
      return Status::InvalidArgument("Dimension mismatch for Transform");
    }
    out_xw = payload.transform * (x - payload.mean);
    return Result<void>();
  }

  Result<MatrixRM> TransformBatch(Eigen::Ref<const MatrixRM> X,
                                  VersionId version) const override {
    std::shared_lock lock(mu_);
    auto it = versions_.find(version);
    if (it == versions_.end()) {
      return Status::NotFound("Whitening version missing");
    }
    const VersionPayload& payload = it->second;
    if (X.cols() != payload.mean.size()) {
      return Status::InvalidArgument("Dimension mismatch for batch transform");
    }
    MatrixRM centered = X.rowwise() - payload.mean.transpose();
    MatrixRM whitened = centered * payload.transform.transpose();
    return whitened;
  }

  Status UpdateStats(Eigen::Ref<const Eigen::VectorXf> x) override {
    std::unique_lock lock(mu_);
    if (pending_count_ == 0) {
      pending_sum_ = Eigen::VectorXf::Zero(x.size());
      pending_cross_ = MatrixRM::Zero(x.size(), x.size());
      dim_ = static_cast<uint32_t>(x.size());
    }
    if (x.size() != pending_sum_.size()) {
      return Status::InvalidArgument("Dimension mismatch in UpdateStats");
    }
    pending_sum_ += x;
    pending_cross_ += x * x.transpose();
    pending_count_++;
    return Status::OK();
  }

  Result<VersionId> FinalizeNewVersion() override {
    std::unique_lock lock(mu_);
    if (pending_count_ == 0) {
      return Status::InvalidArgument("No stats accumulated");
    }
    Eigen::VectorXf mean = pending_sum_ / static_cast<float>(pending_count_);
    MatrixRM cov = (pending_cross_ / static_cast<float>(pending_count_)) - mean * mean.transpose();
    VersionPayload payload;
    payload.mean = mean;
    payload.epsilon = kDefaultEpsilon;
    auto status = ComputePayloadFromCov(cov, &payload);
    if (!status.ok()) {
      return status;
    }
    VersionId version = next_version_++;
    payload.version = version;
    payload.dim = dim_;
    versions_[version] = std::move(payload);
    ResetPending();
    return version;
  }

  Result<MatrixRM> Bridge(VersionId src_version, VersionId dst_version) const override {
    std::shared_lock lock(mu_);
    auto src_it = versions_.find(src_version);
    auto dst_it = versions_.find(dst_version);
    if (src_it == versions_.end() || dst_it == versions_.end()) {
      return Status::NotFound("Version missing for bridge");
    }
    const VersionPayload& src = src_it->second;
    const VersionPayload& dst = dst_it->second;
    if (src.dim != dst.dim) {
      return Status::InvalidArgument("Dimension mismatch in bridge");
    }
    MatrixRM bridge = dst.transform * src.transform_inv;
    return bridge;
  }

  Result<std::vector<uint8_t>> Serialize() const override {
    std::shared_lock lock(mu_);
    const uint32_t magic = 0x5a434137;  // 'ZCA7'
    std::vector<uint8_t> bytes;
    auto append = [&](const void* data, size_t len) {
      const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
      bytes.insert(bytes.end(), ptr, ptr + len);
    };
    uint32_t dim = dim_;
    uint32_t count = static_cast<uint32_t>(versions_.size());
    append(&magic, sizeof(uint32_t));
    append(&dim, sizeof(uint32_t));
    append(&count, sizeof(uint32_t));
    for (const auto& kv : versions_) {
      const VersionPayload& payload = kv.second;
      append(&payload.version, sizeof(VersionId));
      append(&payload.dim, sizeof(uint32_t));
      append(&payload.epsilon, sizeof(float));
      append(payload.mean.data(), sizeof(float) * payload.mean.size());
      append(payload.transform.data(), sizeof(float) * payload.transform.size());
    }
    return bytes;
  }

  Status Deserialize(const std::vector<uint8_t>& bytes) override {
    std::unique_lock lock(mu_);
    versions_.clear();
    dim_ = 0;
    if (bytes.size() < sizeof(uint32_t) * 3) {
      return Status::InvalidArgument("Invalid whitening payload");
    }
    const uint8_t* ptr = bytes.data();
    uint32_t magic = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    if (magic != 0x5a434137) {
      return Status::InvalidArgument("Bad whitening magic");
    }
    uint32_t dim = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    uint32_t count = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    size_t remaining = bytes.data() + bytes.size() - ptr;
    size_t per = sizeof(VersionId) + sizeof(uint32_t) + sizeof(float) +
                 sizeof(float) * dim + sizeof(float) * dim * dim;
    if (remaining < per * count) {
      return Status::InvalidArgument("Payload truncated");
    }
    for (uint32_t i = 0; i < count; ++i) {
      VersionPayload payload;
      payload.version = *reinterpret_cast<const VersionId*>(ptr);
      ptr += sizeof(VersionId);
      payload.dim = *reinterpret_cast<const uint32_t*>(ptr);
      ptr += sizeof(uint32_t);
      payload.epsilon = *reinterpret_cast<const float*>(ptr);
      ptr += sizeof(float);
      payload.mean = Eigen::VectorXf(payload.dim);
      std::memcpy(payload.mean.data(), ptr, sizeof(float) * payload.dim);
      ptr += sizeof(float) * payload.dim;
      payload.transform = MatrixRM(payload.dim, payload.dim);
      std::memcpy(payload.transform.data(), ptr, sizeof(float) * payload.transform.size());
      ptr += sizeof(float) * payload.transform.size();
      payload.transform_inv = payload.transform.inverse();
      versions_[payload.version] = std::move(payload);
    }
    dim_ = dim;
    VersionId max_version = 0;
    for (const auto& kv : versions_) {
      max_version = std::max(max_version, kv.first);
    }
    next_version_ = max_version + 1;
    ResetPending();
    return Status::OK();
  }

 private:
  Status ComputeZCATransform(Eigen::Ref<const MatrixRM> X, VersionPayload* payload) {
    payload->dim = static_cast<uint32_t>(X.cols());
    payload->epsilon = kDefaultEpsilon;
    payload->mean = X.colwise().mean();
    MatrixRM centered = X.rowwise() - payload->mean.transpose();
    MatrixRM cov = (centered.transpose() * centered) / static_cast<float>(X.rows());
    auto status = ComputePayloadFromCov(cov, payload);
    if (!status.ok()) {
      return status;
    }
    MatrixRM whitened = centered * payload->transform.transpose();
    Eigen::VectorXf mean_after = whitened.colwise().mean();
    MatrixRM identity = MatrixRM::Identity(payload->dim, payload->dim);
    MatrixRM cov_after = (whitened.transpose() * whitened) / static_cast<float>(X.rows());
    float mean_norm = mean_after.norm();
    float cov_diff = (cov_after - identity).norm();
    std::cout << "[Whitening] dim=" << payload->dim << " mean_norm=" << mean_norm
              << " cov_diff=" << cov_diff << std::endl;
    return Status::OK();
  }

  Status ComputePayloadFromCov(const MatrixRM& cov, VersionPayload* payload) {
    Eigen::SelfAdjointEigenSolver<MatrixRM> solver(cov);
    if (solver.info() != Eigen::Success) {
      return Status::Internal("Eigen decomposition failed");
    }
    Eigen::VectorXf eigenvalues = solver.eigenvalues();
    MatrixRM eigenvectors = solver.eigenvectors();
  
    // transform = sqrt(Sigma) = V * sqrt(D) * V^T
    Eigen::VectorXf sqrt_vals = eigenvalues.array().sqrt();
    payload->transform = eigenvectors * sqrt_vals.asDiagonal() * eigenvectors.transpose();
  
    // transform_inv = inv_sqrt(Sigma) = V * (1/sqrt(D)) * V^T
    Eigen::VectorXf inv_sqrt = eigenvalues.array().sqrt().inverse();
    payload->transform_inv = eigenvectors * inv_sqrt.asDiagonal() * eigenvectors.transpose();
  
    return Status::OK();
  }
  

  void ResetPending() {
    pending_sum_.resize(0);
    pending_cross_.resize(0, 0);
    pending_count_ = 0;
  }

  mutable std::shared_mutex mu_;
  std::unordered_map<VersionId, VersionPayload> versions_;
  VersionId next_version_{1};
  uint32_t dim_{0};

  // Running stats.
  Eigen::VectorXf pending_sum_;
  MatrixRM pending_cross_;
  uint64_t pending_count_{0};
};

}  // namespace

std::shared_ptr<WhiteningModel> CreateWhiteningModel() {
  return std::make_shared<ZCAWhiteningModel>();
}

}  // namespace ann
