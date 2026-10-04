#pragma once

#include <cstdint>

#include "index/ivf.h"

namespace ann::eval {

struct OnlinePQRollup {
  uint32_t batches{0};
  uint32_t warmup_batches{0};
  uint32_t triggered{0};
  uint32_t updated{0};
  uint32_t reencoded{0};
  uint32_t updated_subspaces{0};
  uint32_t updated_codewords{0};
  double sum_nqe_batch{0.0};
  double sum_qe_ratio{0.0};
  double sum_codebook_drift{0.0};
  double last_nqe_batch{0.0};
  double last_qe_ratio{1.0};
  double last_codebook_drift{0.0};
  uint32_t last_warmup_batches_left{0};
};

struct RunEvalCounterValues {
  double rebuild_ms_total{0.0};
  double total_update_ms{0.0};
  double total_update_whitening_ms{0.0};
  double total_update_insert_ms{0.0};
  double total_update_record_build_ms{0.0};
  double total_update_insert_encode_ms{0.0};
  double total_update_insert_commit_ms{0.0};
  double total_update_onlinepq_maintenance_ms{0.0};
  double total_update_delete_ms{0.0};
  double total_update_codebook_update_ms{0.0};
  double total_update_reencode_ms{0.0};
  double pending_update_ms{0.0};
  double pending_update_whitening_ms{0.0};
  double pending_update_insert_ms{0.0};
  double pending_update_record_build_ms{0.0};
  double pending_update_insert_encode_ms{0.0};
  double pending_update_insert_commit_ms{0.0};
  double pending_update_onlinepq_maintenance_ms{0.0};
  double pending_update_delete_ms{0.0};
  double pending_update_codebook_update_ms{0.0};
  double pending_update_reencode_ms{0.0};
  double total_delta_ingest_assignment_us{0.0};
  uint64_t total_delta_ingest_assignment_records{0};
  double total_merge_compute_ms{0.0};
  double pending_merge_compute_ms{0.0};
  double total_global_rebuild_ms{0.0};
  double pending_global_rebuild_ms{0.0};
  uint32_t global_rebuild_count{0};
  uint64_t eval_sequence{0};
  OnlinePQRollup online_pq_rollup;
};

class RunEvalCounters {
 public:
  explicit RunEvalCounters(double initial_rebuild_ms = 0.0);

  const RunEvalCounterValues& Values() const;
  void AddDeltaIngestAssignment(double assignment_us, uint64_t records);
  void AddUpdate(double update_ms,
                 double whitening_ms,
                 double insert_ms,
                 double record_build_ms,
                 double insert_encode_ms,
                 double insert_commit_ms,
                 double onlinepq_maintenance_ms,
                 double delete_ms,
                 double codebook_update_ms,
                 double reencode_ms);
  void AddMergeCompute(double merge_compute_ms);
  void AddMergeRebuildWall(double merge_wall_ms);
  void AddGlobalRebuild(double rebuild_ms);
  void ObserveOnlinePQ(const OnlinePQUpdateStats& stats);
  uint64_t NextEvalSequence();
  void ResetPending();

 private:
  RunEvalCounterValues values_;
};

struct RunEvalMsCounterValues {
  double init_ms{0.0};
  double total_update_ms{0.0};
  double total_update_whitening_ms{0.0};
  double total_update_insert_ms{0.0};
  double total_update_record_build_ms{0.0};
  double total_update_insert_encode_ms{0.0};
  double total_update_insert_commit_ms{0.0};
  double total_update_onlinepq_maintenance_ms{0.0};
  double total_update_delete_ms{0.0};
  double total_update_codebook_update_ms{0.0};
  double total_update_reencode_ms{0.0};
  double total_delta_ingest_assignment_us{0.0};
  uint64_t total_delta_ingest_assignment_records{0};
  double total_merge_ms{0.0};
  double total_merge_compute_ms{0.0};
  double total_global_rebuild_ms{0.0};
  double pending_update_ms{0.0};
  double pending_merge_compute_ms{0.0};
  double pending_global_rebuild_ms{0.0};
  uint32_t merge_count{0};
  uint32_t global_rebuild_count{0};
  uint32_t search_id{0};
};

class RunEvalMsCounters {
 public:
  const RunEvalMsCounterValues& Values() const;
  void AddInitializationMs(double elapsed_ms);
  void AddWhiteningUpdate(double elapsed_ms);
  void AddInsertElapsed(double elapsed_ms);
  void AddPendingInsertElapsed(double elapsed_ms);
  void AddInsertStats(const OnlinePQUpdateStats& stats);
  void AddDeltaIngestAssignment(double assignment_us, uint64_t records);
  void AddMerge(double elapsed_ms, double merge_compute_ms);
  void AddGlobalRebuild(double rebuild_ms);
  uint32_t NextSearchId();
  void ResetPending();

 private:
  RunEvalMsCounterValues values_;
};

}  // namespace ann::eval
