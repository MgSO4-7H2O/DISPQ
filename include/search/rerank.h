#pragma once

#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

Result<void> RerankL2(
    Eigen::Ref<const Eigen::VectorXf> query,
    std::vector<Candidate>* candidates_inout,
    const std::function<Result<Eigen::VectorXf>(DocId)>& get_vector_callback);

}  // namespace ann
