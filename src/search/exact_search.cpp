#include "search/exact_search.h"

#include <algorithm>
#include <vector>

namespace ann {

Result<std::vector<std::vector<DocId>>> ExactSearchBatch(Eigen::Ref<const MatrixRM> queries,
                                                         Eigen::Ref<const MatrixRM> database,
                                                         uint32_t topk) {
  if (queries.cols() == 0 || database.cols() == 0) {
    return Status::InvalidArgument("empty matrices");
  }
  if (queries.cols() != database.cols()) {
    return Status::InvalidArgument("dimension mismatch between queries and database");
  }
  if (database.rows() == 0) {
    return Status::InvalidArgument("database has no vectors");
  }
  if (topk == 0) {
    return Status::InvalidArgument("topk must be positive");
  }

  Eigen::MatrixXf db = database;
  Eigen::MatrixXf q = queries;
  const Eigen::VectorXf base_norms = db.rowwise().squaredNorm();
  const Eigen::VectorXf query_norms = q.rowwise().squaredNorm();
  Eigen::MatrixXf dots = q * db.transpose();

  std::vector<std::vector<DocId>> all_ids(queries.rows());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int qi = 0; qi < queries.rows(); ++qi) {
    std::vector<std::pair<float, DocId>> dist_pairs(database.rows());
    const float qnorm = query_norms(qi);
    for (int bi = 0; bi < database.rows(); ++bi) {
      float dist = qnorm + base_norms(bi) - 2.0f * dots(qi, bi);
      dist_pairs[bi] = {dist, static_cast<DocId>(bi)};
    }
    const size_t limit = std::min<size_t>(topk, dist_pairs.size());
    if (dist_pairs.size() > limit) {
      std::nth_element(dist_pairs.begin(), dist_pairs.begin() + limit, dist_pairs.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      dist_pairs.resize(limit);
    }
    std::sort(dist_pairs.begin(), dist_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<DocId> ids(limit);
    for (size_t idx = 0; idx < limit; ++idx) {
      ids[idx] = dist_pairs[idx].second;
    }
    all_ids[qi] = std::move(ids);
  }
  return all_ids;
}

}  // namespace ann
