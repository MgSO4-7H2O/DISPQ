#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct PQParams {
  bool enable{false};
  uint32_t M{0};
  uint32_t nbits{8};
  bool residual{true};
};

struct IVFParams {
  uint32_t nlist{1024};
  uint32_t dim{0};
  PQParams pq;
};

class IVFIndex {
 public:
  virtual ~IVFIndex() = default;

  virtual Result<VersionId> Build(Eigen::Ref<const MatrixRM> Xw,
                                  const std::vector<DocId>& ids,
                                  const IVFParams& p,
                                  VersionId index_version) = 0;

  // NTS: Adds vector records to the mutable shard.
  virtual Status Add(const AlignedVector<VectorRecord>& recs) = 0;

  // TS: Searches specified versions using whitened query.
  virtual Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                                      uint32_t topk,
                                      uint32_t nprobe,
                                      const VersionSet& route_versions,
                                      uint8_t from_new) const = 0;

  virtual Result<std::vector<uint8_t>> Serialize() const = 0;
  virtual Status Deserialize(const std::vector<uint8_t>& bytes) = 0;
};

std::shared_ptr<IVFIndex> CreateIVFIndex();

}  // namespace ann
