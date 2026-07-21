#include "eval/gt.h"

#include <algorithm>

namespace ann {

Result<std::vector<std::vector<DocId>>> ComputeGroundTruth(Eigen::Ref<const MatrixRM> queries,
                                                            Eigen::Ref<const MatrixRM> database,
                                                            uint32_t topk) {
  if (queries.cols() == 0 || database.cols() == 0) {
    return Status::InvalidArgument("Empty matrices");
  }
  if (database.rows() == 0) {
    return Status::InvalidArgument("Database must contain rows");
  }
  if (queries.cols() != database.cols()) {
    return Status::InvalidArgument("Dim mismatch between Q and X");
  }
  if (topk == 0) {
    return Status::InvalidArgument("topk must be positive");
  }
  std::vector<std::vector<DocId>> gt(queries.rows());
  for (int qi = 0; qi < queries.rows(); ++qi) {
    std::vector<std::pair<float, DocId>> dists;
    dists.reserve(database.rows());
    for (int xi = 0; xi < database.rows(); ++xi) {
      float dist = (queries.row(qi) - database.row(xi)).squaredNorm();
      dists.emplace_back(dist, static_cast<DocId>(xi));
    }
    std::partial_sort(dists.begin(),
                      dists.begin() + std::min<size_t>(topk, dists.size()),
                      dists.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    const size_t limit = std::min<size_t>(topk, dists.size());
    auto& row = gt[qi];
    row.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
      row.push_back(dists[i].second);
    }
  }
  return gt;
}

}  // namespace ann
