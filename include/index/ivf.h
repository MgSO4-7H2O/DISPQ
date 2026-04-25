#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "common/result.h"
#include "common/types.h"

namespace ann {

struct PQParams {
  bool enable{false};
  uint32_t M{0};
  uint32_t nbits{8};
  bool residual{true};
};

struct IVFParams {
  uint32_t nlist{1024};
  uint32_t dim{0};
  uint32_t kmeans_iterations{20};
  PQParams pq;
  bool use_fixed_routing_centroids{false};
  MatrixRM fixed_routing_centroids;
  bool use_fixed_pq_codebooks{false};
  std::vector<MatrixRM> fixed_pq_codebooks;
  std::vector<std::vector<uint64_t>> fixed_pq_counts;
  double fixed_pq_baseline_nqe{-1.0};
  double fixed_pq_ema_nqe{-1.0};
};

struct OnlinePQUpdateOptions {
  bool enable{false};
  double qe_ratio_threshold{1.05};
  double ema_alpha{0.1};
  double nqe_eps{1e-6};
  bool warmup_enable{false};
  uint32_t warmup_batches{0};
  uint32_t force_update_interval{0};
  bool partial_top_alpha{false};
  double partial_alpha{1.0};
  bool partial_top_lambda{false};
  double partial_lambda{1.0};
  bool reencode_batch_after_update{true};
};

struct OnlinePQUpdateStats {
  bool use_online_pq{false};
  bool trigger_update{false};
  bool updated_codebook{false};
  bool reencoded_batch{false};
  bool in_warmup{false};
  uint32_t warmup_batches_left{0};
  uint32_t processed_vectors{0};
  uint32_t deleted_vectors{0};
  uint32_t updated_subspaces{0};
  uint32_t updated_codewords{0};
  double nqe_batch{0.0};
  double nqe_ema{0.0};
  double nqe_baseline{0.0};
  double qe_ratio{1.0};
  double codebook_drift_l2{0.0};
};

struct PQRuntimeState {
  bool use_pq{false};
  bool pq_residual{true};
  uint32_t M{0};
  uint32_t Ks{0};
  uint32_t dsub{0};
  std::vector<MatrixRM> codebooks;
  std::vector<std::vector<uint64_t>> counts;
  double nqe_baseline{0.0};
  double nqe_ema{0.0};
  double qe_ratio{1.0};
  uint64_t ntotal{0};
};

struct PartitionPatch {
  std::vector<uint32_t> partition_ids;
  std::vector<AlignedVector<VectorRecord>> replacement_records;
};

class IVFIndex {
 public:
  virtual ~IVFIndex() = default;

  virtual Result<VersionId> Build(Eigen::Ref<const MatrixRM> Xw,
                                  const std::vector<DocId>& ids,
                                  const IVFParams& p,
                                  VersionId index_version) = 0;

  // NTS: Adds vector records to the mutable shard.
  virtual Status Add(const AlignedVector<VectorRecord>& recs) = 0;

  // NTS: Adds records and optionally applies residual OnlinePQ codebook update.
  virtual Result<OnlinePQUpdateStats> AddWithOnlinePQ(
      const AlignedVector<VectorRecord>& recs,
      const OnlinePQUpdateOptions& options) = 0;

  // NTS: Sliding-window OnlinePQ step with insertion and deletion in one update.
  virtual Result<OnlinePQUpdateStats> AddWithOnlinePQSlidingWindow(
      const AlignedVector<VectorRecord>& recs,
      const std::vector<DocId>& delete_doc_ids,
      const OnlinePQUpdateOptions& options) = 0;

  // TS: Searches specified versions using whitened query.
  virtual Result<SearchResult> Search(Eigen::Ref<const Eigen::VectorXf> qw,
                                      uint32_t topk,
                                      uint32_t nprobe,
                                      const VersionSet& route_versions,
                                      uint8_t from_new) const = 0;

  virtual Result<MatrixRM> GetRoutingCentroids(const VersionSet& route_versions) const = 0;
  virtual Result<PQRuntimeState> GetPQRuntimeState(const VersionSet& route_versions) const = 0;
  virtual Result<std::vector<uint8_t>> GetDocPQCode(const VersionSet& route_versions,
                                                    DocId doc_id) const = 0;
  virtual Result<AlignedVector<VectorRecord>> SnapshotRecords(
      const VersionSet& route_versions) const = 0;
  virtual Result<AlignedVector<VectorRecord>> GetPartitionRecords(
      const VersionSet& route_versions,
      uint32_t partition_id) const = 0;
  virtual Result<std::vector<uint32_t>> GetPartitionSizes(
      const VersionSet& route_versions) const = 0;
  virtual Status CommitPartitionPatch(const VersionSet& route_versions,
                                      const PartitionPatch& patch) = 0;
  virtual Result<double> GetLastPatchPQReencodeMs(
      const VersionSet& route_versions) const = 0;

  virtual Result<std::vector<uint8_t>> Serialize() const = 0;
  virtual Status Deserialize(const std::vector<uint8_t>& bytes) = 0;
};

std::shared_ptr<IVFIndex> CreateIVFIndex();

}  // namespace ann
