#include "search/rerank.h"

#include <algorithm>

namespace ann {
namespace {

float SquaredL2FromNormDot(float a_norm, float b_norm, float dot) {
  const float dist = a_norm + b_norm - 2.0f * dot;
  return dist >= 0.0f ? dist : 0.0f;
}

}  // namespace

Result<void> RerankL2(
    Eigen::Ref<const Eigen::VectorXf> query,
    std::vector<Candidate>* candidates_inout,
    const std::function<Result<Eigen::VectorXf>(DocId)>& get_vector_callback) {
  return RerankL2(query, candidates_inout, get_vector_callback, {});
}

Result<void> RerankL2(
    Eigen::Ref<const Eigen::VectorXf> query,
    std::vector<Candidate>* candidates_inout,
    const std::function<Result<Eigen::VectorXf>(DocId)>& get_vector_callback,
    const std::function<Result<float>(DocId)>& get_norm_callback) {
  if (candidates_inout == nullptr) {
    return Status::InvalidArgument("candidates_inout is null");
  }
  if (!get_vector_callback) {
    return Status::InvalidArgument("Callback missing");
  }
  const float query_norm = query.squaredNorm();
  for (auto& cand : *candidates_inout) {
    auto vec_res = get_vector_callback(cand.doc_id);
    if (!vec_res.ok()) {
      return vec_res.status();
    }
    const Eigen::VectorXf& vec = vec_res.value();
    if (vec.size() != query.size()) {
      return Status::InvalidArgument("Vector dim mismatch in rerank");
    }
    float vec_norm = 0.0f;
    if (get_norm_callback) {
      auto norm_res = get_norm_callback(cand.doc_id);
      if (!norm_res.ok()) {
        return norm_res.status();
      }
      vec_norm = norm_res.value();
    } else {
      vec_norm = vec.squaredNorm();
    }
    cand.rerank_dist = SquaredL2FromNormDot(query_norm, vec_norm, query.dot(vec));
  }
  std::sort(candidates_inout->begin(), candidates_inout->end(),
            [](const Candidate& a, const Candidate& b) { return a.rerank_dist < b.rerank_dist; });
  return Result<void>();
}

}  // namespace ann
