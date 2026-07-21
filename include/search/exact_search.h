#pragma once

#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

// Batched helper returning only doc-id lists for evaluation baselines.
Result<std::vector<std::vector<DocId>>> ExactSearchBatch(Eigen::Ref<const MatrixRM> queries,
                                                         Eigen::Ref<const MatrixRM> database,
                                                         uint32_t topk);

}  // namespace ann
