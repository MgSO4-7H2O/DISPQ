#pragma once

#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

Result<std::vector<std::vector<DocId>>> ComputeGroundTruth(Eigen::Ref<const MatrixRM> queries,
                                                            Eigen::Ref<const MatrixRM> database,
                                                            uint32_t topk);

}  // namespace ann
