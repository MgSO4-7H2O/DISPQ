#include "index/postings.h"

#include <mutex>
#include <shared_mutex>

namespace ann {

PostingStore::PostingStore() = default;
PostingStore::~PostingStore() = default;

Result<PostingListView> PostingStore::GetPostingList(VersionId version, uint32_t ivf_id) const {
  std::shared_lock lock(mu_);
  auto vit = shards_.find(version);
  if (vit == shards_.end()) {
    return Status::NotFound("Version not found");
  }
  const auto& shard = vit->second;
  auto sit = shard.find(ivf_id);
  if (sit == shard.end()) {
    return Status::NotFound("IVF list missing");
  }
  const auto& vec = sit->second;
  return PostingListView{vec.data(), vec.size()};
}

Status PostingStore::AddPosting(VersionId version, uint32_t ivf_id, const VectorRecord& record) {
  std::unique_lock lock(mu_);
  auto& shard = shards_[version];
  auto& vec = shard[ivf_id];
  vec.push_back(record);
  return Status::OK();
}

}  // namespace ann
