#include "search/hybrid_search.h"

#include <memory>

namespace ann {
namespace {

class ConcreteHybridSearcher : public HybridSearcher {
 public:
  explicit ConcreteHybridSearcher(Config config) : config_(std::move(config)) {}

  Status SetIndex(std::shared_ptr<IVFIndex> index, const VersionSet& versions) override {
    if (!index) {
      return Status::InvalidArgument("index is null");
    }
    if (versions.index_version == 0) {
      return Status::InvalidArgument("index_version must be set");
    }
    ivf_ = std::move(index);
    route_versions_ = versions;
    return Status::OK();
  }

  Status SetWhitening(std::shared_ptr<WhiteningModel> model, VersionId version) override {
    if (!model) {
      whitening_.reset();
      whitening_version_ = 0;
      return Status::OK();
    }
    whitening_ = std::move(model);
    whitening_version_ = version;
    return Status::OK();
  }

  Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> query,
                              const SearchParams& params) const override {
    if (static_cast<uint32_t>(query.size()) != config_.dim) {
      return Status::InvalidArgument("Query dim mismatch");
    }
    if (params.topk == 0) {
      return Status::InvalidArgument("topk must be positive");
    }
    if (!ivf_) {
      return Status::Unavailable("IVF index not initialized");
    }
    const uint8_t from_new = params.enable_dual_route ? 1 : 0;
    if (params.use_whitening && whitening_) {
      Eigen::VectorXf query_buf(query.size());
      auto tstatus = whitening_->Transform(query, whitening_version_, query_buf);
      if (!tstatus.ok()) {
        return tstatus.status();
      }
      return ivf_->Search(query_buf, params.topk, params.nprobe, route_versions_, from_new);
    }
    return ivf_->Search(query, params.topk, params.nprobe, route_versions_, from_new);
  }

 private:
  Config config_;
  std::shared_ptr<IVFIndex> ivf_;
  VersionSet route_versions_{};
  std::shared_ptr<WhiteningModel> whitening_;
  VersionId whitening_version_{0};
};

}  // namespace

Result<std::unique_ptr<HybridSearcher>> CreateHybridSearcher(const Config& config) {
  return std::unique_ptr<HybridSearcher>(new ConcreteHybridSearcher(config));
}

}  // namespace ann
