#include "eval/eval_counter.h"

namespace ann::eval {

RunEvalCounters::RunEvalCounters(double initial_rebuild_ms) {
  values_.rebuild_ms_total = initial_rebuild_ms;
}

const RunEvalCounterValues& RunEvalCounters::Values() const {
  return values_;
}

void RunEvalCounters::AddDeltaIngestAssignment(double assignment_us,
                                              uint64_t records) {
  values_.total_delta_ingest_assignment_us += assignment_us;
  values_.total_delta_ingest_assignment_records += records;
}

void RunEvalCounters::AddUpdate(double update_ms,
                                double whitening_ms,
                                double insert_ms,
                                double record_build_ms,
                                double insert_encode_ms,
                                double insert_commit_ms,
                                double onlinepq_maintenance_ms,
                                double delete_ms,
                                double codebook_update_ms,
                                double reencode_ms) {
  values_.total_update_ms += update_ms;
  values_.total_update_whitening_ms += whitening_ms;
  values_.total_update_insert_ms += insert_ms;
  values_.total_update_record_build_ms += record_build_ms;
  values_.total_update_insert_encode_ms += insert_encode_ms;
  values_.total_update_insert_commit_ms += insert_commit_ms;
  values_.total_update_onlinepq_maintenance_ms += onlinepq_maintenance_ms;
  values_.total_update_delete_ms += delete_ms;
  values_.total_update_codebook_update_ms += codebook_update_ms;
  values_.total_update_reencode_ms += reencode_ms;
  values_.pending_update_ms += update_ms;
  values_.pending_update_whitening_ms += whitening_ms;
  values_.pending_update_insert_ms += insert_ms;
  values_.pending_update_record_build_ms += record_build_ms;
  values_.pending_update_insert_encode_ms += insert_encode_ms;
  values_.pending_update_insert_commit_ms += insert_commit_ms;
  values_.pending_update_onlinepq_maintenance_ms += onlinepq_maintenance_ms;
  values_.pending_update_delete_ms += delete_ms;
  values_.pending_update_codebook_update_ms += codebook_update_ms;
  values_.pending_update_reencode_ms += reencode_ms;
}

void RunEvalCounters::AddMergeCompute(double merge_compute_ms) {
  values_.total_merge_compute_ms += merge_compute_ms;
  values_.pending_merge_compute_ms += merge_compute_ms;
}

void RunEvalCounters::AddMergeRebuildWall(double merge_wall_ms) {
  values_.rebuild_ms_total += merge_wall_ms;
}

void RunEvalCounters::AddGlobalRebuild(double rebuild_ms) {
  values_.rebuild_ms_total += rebuild_ms;
  values_.total_global_rebuild_ms += rebuild_ms;
  values_.pending_global_rebuild_ms += rebuild_ms;
  values_.global_rebuild_count++;
}

void RunEvalCounters::ObserveOnlinePQ(const OnlinePQUpdateStats& stats) {
  values_.online_pq_rollup.batches++;
  values_.online_pq_rollup.sum_nqe_batch += stats.nqe_batch;
  values_.online_pq_rollup.sum_qe_ratio += stats.qe_ratio;
  values_.online_pq_rollup.sum_codebook_drift += stats.codebook_drift_l2;
  values_.online_pq_rollup.last_nqe_batch = stats.nqe_batch;
  values_.online_pq_rollup.last_qe_ratio = stats.qe_ratio;
  values_.online_pq_rollup.last_codebook_drift = stats.codebook_drift_l2;
  values_.online_pq_rollup.last_warmup_batches_left = stats.warmup_batches_left;
  if (stats.in_warmup) {
    values_.online_pq_rollup.warmup_batches++;
  }
  if (stats.trigger_update) {
    values_.online_pq_rollup.triggered++;
  }
  if (stats.updated_codebook) {
    values_.online_pq_rollup.updated++;
  }
  if (stats.reencoded_batch) {
    values_.online_pq_rollup.reencoded++;
  }
  values_.online_pq_rollup.updated_subspaces += stats.updated_subspaces;
  values_.online_pq_rollup.updated_codewords += stats.updated_codewords;
}

uint64_t RunEvalCounters::NextEvalSequence() {
  return ++values_.eval_sequence;
}

void RunEvalCounters::ResetPending() {
  values_.pending_update_ms = 0.0;
  values_.pending_update_whitening_ms = 0.0;
  values_.pending_update_insert_ms = 0.0;
  values_.pending_update_record_build_ms = 0.0;
  values_.pending_update_insert_encode_ms = 0.0;
  values_.pending_update_insert_commit_ms = 0.0;
  values_.pending_update_onlinepq_maintenance_ms = 0.0;
  values_.pending_update_delete_ms = 0.0;
  values_.pending_update_codebook_update_ms = 0.0;
  values_.pending_update_reencode_ms = 0.0;
  values_.pending_merge_compute_ms = 0.0;
  values_.pending_global_rebuild_ms = 0.0;
}

const RunEvalMsCounterValues& RunEvalMsCounters::Values() const {
  return values_;
}

void RunEvalMsCounters::AddInitializationMs(double elapsed_ms) {
  values_.init_ms += elapsed_ms;
}

void RunEvalMsCounters::AddWhiteningUpdate(double elapsed_ms) {
  values_.total_update_ms += elapsed_ms;
  values_.total_update_whitening_ms += elapsed_ms;
  values_.pending_update_ms += elapsed_ms;
}

void RunEvalMsCounters::AddInsertElapsed(double elapsed_ms) {
  values_.total_update_ms += elapsed_ms;
}

void RunEvalMsCounters::AddPendingInsertElapsed(double elapsed_ms) {
  values_.pending_update_ms += elapsed_ms;
}

void RunEvalMsCounters::AddInsertStats(const OnlinePQUpdateStats& stats) {
  values_.total_update_insert_ms += stats.insert_ms;
  values_.total_update_record_build_ms += stats.record_build_ms;
  values_.total_update_insert_encode_ms += stats.insert_encode_ms;
  values_.total_update_insert_commit_ms += stats.insert_commit_ms;
  values_.total_update_delete_ms += stats.delete_ms;
  values_.total_update_codebook_update_ms += stats.codebook_update_ms;
  values_.total_update_reencode_ms += stats.reencode_ms;
  values_.total_update_onlinepq_maintenance_ms += stats.maintenance_ms;
}

void RunEvalMsCounters::AddDeltaIngestAssignment(double assignment_us,
                                                 uint64_t records) {
  values_.total_delta_ingest_assignment_us += assignment_us;
  values_.total_delta_ingest_assignment_records += records;
}

void RunEvalMsCounters::AddMerge(double elapsed_ms, double merge_compute_ms) {
  values_.total_merge_ms += elapsed_ms;
  values_.total_merge_compute_ms += merge_compute_ms;
  values_.pending_merge_compute_ms += merge_compute_ms;
  values_.merge_count++;
}

void RunEvalMsCounters::AddGlobalRebuild(double rebuild_ms) {
  values_.total_global_rebuild_ms += rebuild_ms;
  values_.pending_global_rebuild_ms += rebuild_ms;
  values_.global_rebuild_count++;
}

uint32_t RunEvalMsCounters::NextSearchId() {
  return ++values_.search_id;
}

void RunEvalMsCounters::ResetPending() {
  values_.pending_update_ms = 0.0;
  values_.pending_merge_compute_ms = 0.0;
  values_.pending_global_rebuild_ms = 0.0;
}

}  // namespace ann::eval
