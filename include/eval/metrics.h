#pragma once

#include <vector>

#include "common/result.h"
#include "common/timer.h"
#include "common/types.h"

namespace ann {

Result<float> RecallAtK(const std::vector<std::vector<DocId>>& gt,
                        const std::vector<std::vector<DocId>>& pred,
                        uint32_t k);

struct BenchmarkSummary {
  double p50_ms{0.0};
  double p99_ms{0.0};
};

Result<BenchmarkSummary> SummarizeLatencies(const std::vector<double>& latencies_ms);

}  // namespace ann
