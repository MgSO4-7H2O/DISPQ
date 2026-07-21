#pragma once

#include <memory>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/result.h"
#include "common/types.h"
#include "index/ivf.h"
#include "whitening/whitening.h"

namespace ann {

struct SearchParams {
  uint32_t topk{10};
  uint32_t nprobe{8};
  bool use_whitening{true};
  bool enable_dual_route{true};
};

class HybridSearcher {
 public:
  virtual ~HybridSearcher() = default;

  // NTS: Injects the IVF index and associated route versions before serving traffic.
  virtual Status SetIndex(std::shared_ptr<IVFIndex> index, const VersionSet& versions) = 0;

  // NTS: Optionally injects a whitening model snapshot applied on every query.
  virtual Status SetWhitening(std::shared_ptr<WhiteningModel> model, VersionId version) = 0;

  // TS: Executes retrieval given a single query vector.
  virtual Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> query,
                                      const SearchParams& params) const = 0;
};

Result<std::unique_ptr<HybridSearcher>> CreateHybridSearcher(const Config& config);

}  // namespace ann
