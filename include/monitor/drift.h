#pragma once

#include <deque>
#include <shared_mutex>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct DriftParams {
  uint32_t nlist{256};
  uint32_t pq_m{16};
  uint32_t pq_nbits{8};
  uint32_t confirm_k{3};
  uint32_t confirm_m{5};
  double confirm_ratio{0.6};
  uint32_t min_delta_lifetime_windows{2};
  double soft_npd_ratio{1.2};
  double soft_gain{0.05};
  double hard_nre_ratio{1.5};
  double hard_cm_z{3.0};
  double hard_lds{0.2};
  uint32_t max_closed_deltas{3};
  double max_closed_ratio{0.15};
  uint32_t active_delta_max_docs{20000};
};

struct DriftContext {
  uint32_t total_docs{0};
  uint32_t active_delta_docs{0};
  uint32_t closed_delta_count{0};
  uint32_t closed_delta_docs{0};
};

struct DriftMetrics {
  double nre{0.0};
  double npd{0.0};
  double cm{0.0};
  double lds{0.0};
  double cue{0.0};
  double nre_ratio{1.0};
  double npd_ratio{1.0};
  double cm_z{0.0};
  double cue_ratio{1.0};
};

struct DriftDecision {
  DriftMetrics metrics{};
  bool soft_hit{false};
  bool hard_hit{false};
  bool trigger_new_delta{false};
  bool trigger_seal_delta{false};
  bool trigger_rebuild_main{false};
  double candidate_gain{0.0};
  double candidate_npd{0.0};
};

class DriftMonitor {
 public:
  DriftMonitor();
  ~DriftMonitor();

  // NTS: Initializes baseline statistics and monitoring models.
  Status Initialize(Eigen::Ref<const MatrixRM> baseline_vectors, const DriftParams& params);

  // NTS: Evaluates a streaming window and returns metrics + trigger decisions.
  Result<DriftDecision> ObserveWindow(Eigen::Ref<const MatrixRM> window_vectors,
                                      const DriftContext& context);

  // NTS: Resets delta-lifetime guards after rotating active delta.
  // If promote_candidate is true, candidate PQ becomes the active drift PQ.
  Status OnNewDeltaPromoted(bool promote_candidate);

  // NTS: Rebuilds baseline after main rebuild.
  Status OnMainRebuilt(Eigen::Ref<const MatrixRM> baseline_vectors);

  // TS: Returns baseline metrics used for ratio / z-score normalization.
  Result<DriftMetrics> BaselineMetrics() const;

  // Legacy APIs kept for compatibility with old tests.
  Status ObserveResidual(Eigen::Ref<const MatrixRM> residuals);
  Status ObserveSearchSignal(Eigen::Ref<const Eigen::VectorXf> margins);
  Result<bool> ShouldUpdateWhiten() const;
  Result<bool> ShouldUpdateTail() const;

 private:
  struct PQModel {
    uint32_t M{0};
    uint32_t nbits{8};
    uint32_t dsub{0};
    std::vector<MatrixRM> codebooks;

    bool valid() const { return M > 0 && dsub > 0 && codebooks.size() == M; }
  };

  struct WindowEval {
    DriftMetrics metrics{};
    MatrixRM residuals;
  };

  Status FitBaselineLocked(Eigen::Ref<const MatrixRM> baseline_vectors);
  Result<WindowEval> EvaluateWindowLocked(Eigen::Ref<const MatrixRM> window_vectors) const;
  Result<PQModel> TrainPQModelLocked(Eigen::Ref<const MatrixRM> residuals, uint32_t pq_m) const;
  Result<double> ComputeNPDLocked(Eigen::Ref<const MatrixRM> residuals,
                                  const PQModel& pq,
                                  double* cue_out) const;
  Result<double> EstimateCandidateGainLocked(Eigen::Ref<const MatrixRM> residuals,
                                             double npd_old,
                                             double* npd_new_out);
  bool TriggerByHistoryLocked(const std::deque<bool>& history) const;

  mutable std::shared_mutex mu_;
  DriftParams params_{};
  bool initialized_{false};
  MatrixRM coarse_centroids_;
  std::vector<double> baseline_list_dist_;
  DriftMetrics baseline_metrics_{};
  double baseline_cm_std_{1e-6};
  PQModel active_pq_;
  PQModel candidate_pq_;
  bool has_candidate_pq_{false};
  std::deque<bool> soft_hit_history_;
  std::deque<bool> hard_hit_history_;
  uint32_t windows_since_delta_create_{0};
  bool request_whiten_{false};
  bool request_tail_{false};
};

}  // namespace ann
