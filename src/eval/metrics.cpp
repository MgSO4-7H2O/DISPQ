#include "eval/metrics.h"

#include <algorithm>
#include <unordered_set>

namespace ann {

Result<float> RecallAtK(const std::vector<std::vector<DocId>>& gt,
                        const std::vector<std::vector<DocId>>& pred,
                        uint32_t k) {
  if (gt.size() != pred.size()) {
    return Status::InvalidArgument("gt/pred size mismatch");
  }
  if (k == 0) {
    return Status::InvalidArgument("k must be positive");
  }
  if (gt.empty()) {
    return 0.0f;
  }
  float total = 0.0f;
  for (size_t i = 0; i < gt.size(); ++i) {
    const auto& gt_row = gt[i];
    const auto& pred_row = pred[i];
    uint32_t hits = 0;
    std::unordered_set<DocId> counted;
    counted.reserve(std::min<uint32_t>(k, static_cast<uint32_t>(pred_row.size())) * 2 + 1);
    for (uint32_t j = 0; j < std::min<uint32_t>(k, static_cast<uint32_t>(pred_row.size())); ++j) {
      if (counted.insert(pred_row[j]).second &&
          std::find(gt_row.begin(), gt_row.end(), pred_row[j]) != gt_row.end()) {
        ++hits;
      }
    }
    const uint32_t denominator =
        std::min<uint32_t>(k, static_cast<uint32_t>(gt_row.size()));
    if (denominator > 0) {
      total += static_cast<float>(hits) / static_cast<float>(denominator);
    }
  }
  return total / static_cast<float>(gt.size());
}

Result<BenchmarkSummary> SummarizeLatencies(const std::vector<double>& latencies_ms) {
  if (latencies_ms.empty()) {
    return Status::InvalidArgument("latencies empty");
  }
  std::vector<double> sorted = latencies_ms;
  std::sort(sorted.begin(), sorted.end());
  auto pick = [&](double quantile) {
    const size_t idx = static_cast<size_t>(quantile * (sorted.size() - 1));
    return sorted[idx];
  };
  BenchmarkSummary summary;
  summary.p50_ms = pick(0.5);
  summary.p99_ms = pick(0.99);
  return summary;
}

}  // namespace ann
