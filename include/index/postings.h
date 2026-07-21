#pragma once

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct PostingListView {
  const VectorRecord* data{nullptr};
  size_t size{0};
};

class PostingStore {
 public:
  PostingStore();
  ~PostingStore();

  // TS: Provides a read-only snapshot of a posting list for version/ivf_id.
  Result<PostingListView> GetPostingList(VersionId version, uint32_t ivf_id) const;

  // NTS: Appends a posting for a target shard.
  Status AddPosting(VersionId version, uint32_t ivf_id, const VectorRecord& record);

 private:
  using Shard = std::unordered_map<uint32_t, AlignedVector<VectorRecord>>;
  mutable std::shared_mutex mu_;
  std::unordered_map<VersionId, Shard> shards_;
};

}  // namespace ann
