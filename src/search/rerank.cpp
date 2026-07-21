#include "search/rerank.h"

#include <algorithm>

namespace ann {

Result<void> RerankL2(
    Eigen::Ref<const Eigen::VectorXf> query,
    std::vector<Candidate>* candidates_inout,
    const std::function<Result<Eigen::VectorXf>(DocId)>& get_vector_callback) {
  if (candidates_inout == nullptr) {
    return Status::InvalidArgument("candidates_inout is null");
  }
  if (!get_vector_callback) {
    return Status::InvalidArgument("Callback missing");
  }
  for (auto& cand : *candidates_inout) {
    auto vec_res = get_vector_callback(cand.doc_id);
    if (!vec_res.ok()) {
      return vec_res.status();
    }
    const Eigen::VectorXf& vec = vec_res.value();
    if (vec.size() != query.size()) {
      return Status::InvalidArgument("Vector dim mismatch in rerank");
    }
    cand.rerank_dist = (vec - query).squaredNorm();
  }
  std::sort(candidates_inout->begin(), candidates_inout->end(),
            [](const Candidate& a, const Candidate& b) { return a.rerank_dist < b.rerank_dist; });
  return Result<void>();
}

}  // namespace ann
