#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "common/types.h"
#include "index/ivf.h"
#include "index/merge.h"

using namespace ann;

namespace {

AlignedVector<VectorRecord> MakeRecords(Eigen::Ref<const MatrixRM> X,
                                        const std::vector<DocId>& ids,
                                        uint32_t dim,
                                        const VersionSet& versions) {
  assert(ids.size() == static_cast<size_t>(X.rows()));
  AlignedVector<VectorRecord> out;
  out.reserve(ids.size());
  for (int64_t i = 0; i < X.rows(); ++i) {
    VectorRecord rec;
    rec.doc_id = ids[static_cast<size_t>(i)];
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = X.row(i).transpose();
    out.push_back(std::move(rec));
  }
  return out;
}

std::vector<MatrixRM> MakeMainCodebooks() {
  std::vector<MatrixRM> books(2, MatrixRM(4, 2));
  books[0] << -2.0f, -2.0f, -1.0f, -1.0f, 1.0f, 1.0f, 2.0f, 2.0f;
  books[1] << -2.0f, -2.0f, -1.0f, -1.0f, 1.0f, 1.0f, 2.0f, 2.0f;
  return books;
}

std::vector<MatrixRM> MakeDeltaCodebooks() {
  std::vector<MatrixRM> books(2, MatrixRM(4, 2));
  books[0] << 2.0f, 2.0f, 1.0f, 1.0f, -1.0f, -1.0f, -2.0f, -2.0f;
  books[1] << 2.0f, 2.0f, 1.0f, 1.0f, -1.0f, -1.0f, -2.0f, -2.0f;
  return books;
}

struct Fixture {
  std::shared_ptr<IVFIndex> main_ivf;
  VersionSet main_versions{};
  std::shared_ptr<IVFIndex> delta_ivf;
  VersionSet delta_versions{};
};

Fixture BuildFixture() {
  constexpr uint32_t kDim = 4;
  MatrixRM centroids(3, kDim);
  centroids << -10.0f, 0.0f, -10.0f, 0.0f,   // p0
      10.0f, 0.0f, 10.0f, 0.0f,              // p1
      0.0f, 10.0f, 0.0f, 10.0f;              // p2

  MatrixRM train = MatrixRM::Zero(6, kDim);
  std::vector<DocId> train_ids(static_cast<size_t>(train.rows()));
  for (size_t i = 0; i < train_ids.size(); ++i) {
    train_ids[i] = static_cast<DocId>(i);
  }

  IVFParams main_params;
  main_params.nlist = 3;
  main_params.dim = kDim;
  main_params.use_fixed_routing_centroids = true;
  main_params.fixed_routing_centroids = centroids;
  main_params.pq.enable = true;
  main_params.pq.M = 2;
  main_params.pq.nbits = 2;
  main_params.pq.residual = true;
  main_params.use_fixed_pq_codebooks = true;
  main_params.fixed_pq_codebooks = MakeMainCodebooks();

  auto main_ivf = CreateIVFIndex();
  auto main_ver = main_ivf->Build(train, train_ids, main_params, 101);
  assert(main_ver.ok());
  VersionSet main_versions{0, main_ver.value()};

  MatrixRM main_docs(4, kDim);
  main_docs << -11.0f, -1.0f, -10.5f, -1.0f,   // p0
      -9.0f, 1.0f, -9.0f, 1.0f,                // p0
      9.0f, 0.5f, 10.0f, 1.0f,                 // p1
      0.0f, 11.0f, 0.0f, 11.0f;                // p2
  std::vector<DocId> main_doc_ids = {0, 1, 2, 3};
  auto main_records = MakeRecords(main_docs, main_doc_ids, kDim, main_versions);
  assert(main_ivf->Add(main_records).ok());

  IVFParams delta_params = main_params;
  delta_params.fixed_pq_codebooks = MakeDeltaCodebooks();
  auto delta_ivf = CreateIVFIndex();
  auto delta_ver = delta_ivf->Build(train, train_ids, delta_params, 202);
  assert(delta_ver.ok());
  VersionSet delta_versions{0, delta_ver.value()};

  MatrixRM delta_docs(3, kDim);
  delta_docs << -12.0f, -2.0f, -11.0f, -2.0f,   // -> p0
      12.0f, 2.0f, 11.0f, 2.0f,                 // -> p1
      8.0f, 0.0f, 9.0f, 0.0f;                   // -> p1
  std::vector<DocId> delta_doc_ids = {100, 101, 102};
  auto delta_records = MakeRecords(delta_docs, delta_doc_ids, kDim, delta_versions);
  assert(delta_ivf->Add(delta_records).ok());

  Fixture fx;
  fx.main_ivf = main_ivf;
  fx.main_versions = main_versions;
  fx.delta_ivf = delta_ivf;
  fx.delta_versions = delta_versions;
  return fx;
}

Fixture BuildHealthierNeighborFixture() {
  constexpr uint32_t kDim = 2;
  MatrixRM centroids(2, kDim);
  centroids << 0.0f, 0.0f, 1.0f, 0.0f;

  MatrixRM train = MatrixRM::Zero(2, kDim);
  std::vector<DocId> train_ids = {0, 1};

  IVFParams params;
  params.nlist = 2;
  params.dim = kDim;
  params.use_fixed_routing_centroids = true;
  params.fixed_routing_centroids = centroids;
  params.pq.enable = false;

  auto main_ivf = CreateIVFIndex();
  auto main_ver = main_ivf->Build(train, train_ids, params, 301);
  assert(main_ver.ok());
  VersionSet main_versions{0, main_ver.value()};

  MatrixRM main_docs(10, kDim);
  main_docs << 0.00f, 0.00f,
      0.01f, 0.00f,
      0.02f, 0.00f,
      0.03f, 0.00f,
      0.04f, 0.00f,
      0.05f, 0.00f,
      0.06f, 0.00f,
      1.00f, 0.00f,
      1.01f, 0.00f,
      1.02f, 0.00f;
  std::vector<DocId> main_doc_ids = {10, 11, 12, 13, 14, 15, 16, 20, 21, 22};
  auto main_records = MakeRecords(main_docs, main_doc_ids, kDim, main_versions);
  assert(main_ivf->Add(main_records).ok());

  auto delta_ivf = CreateIVFIndex();
  auto delta_ver = delta_ivf->Build(train, train_ids, params, 302);
  assert(delta_ver.ok());
  VersionSet delta_versions{0, delta_ver.value()};

  MatrixRM delta_docs(1, kDim);
  delta_docs << 0.49f, 0.0f;
  std::vector<DocId> delta_doc_ids = {200};
  auto delta_records = MakeRecords(delta_docs, delta_doc_ids, kDim, delta_versions);
  assert(delta_ivf->Add(delta_records).ok());

  Fixture fx;
  fx.main_ivf = main_ivf;
  fx.main_versions = main_versions;
  fx.delta_ivf = delta_ivf;
  fx.delta_versions = delta_versions;
  return fx;
}

}  // namespace

int main() {
  Fixture fx = BuildFixture();

  // 1) Delta -> Main centroid assignment + freeze.
  auto frozen_res = freeze_delta(fx.delta_ivf, fx.delta_versions);
  assert(frozen_res.ok());
  const FrozenDelta frozen = frozen_res.value();
  assert(frozen.records.size() == 3);

  auto centroids_res = fx.main_ivf->GetRoutingCentroids(fx.main_versions);
  assert(centroids_res.ok());
  auto assign_res = assign_delta_to_main_centroids(frozen, centroids_res.value());
  assert(assign_res.ok());
  const PartitionAssignments assignments = assign_res.value();
  assert(assignments.size() == 3);

  std::unordered_map<DocId, uint32_t> assigned_part;
  for (uint32_t p = 0; p < static_cast<uint32_t>(assignments.size()); ++p) {
    for (const auto& item : assignments[static_cast<size_t>(p)]) {
      assigned_part[item.record.doc_id] = p;
    }
  }
  assert(assigned_part[100] == 0);
  assert(assigned_part[101] == 1);
  assert(assigned_part[102] == 1);

  // 2) partition scoring.
  auto sizes_before_res = fx.main_ivf->GetPartitionSizes(fx.main_versions);
  assert(sizes_before_res.ok());
  const std::vector<uint32_t> sizes_before = sizes_before_res.value();
  assert(sizes_before.size() == 3);
  assert(sizes_before[0] == 2 && sizes_before[1] == 1 && sizes_before[2] == 1);

  auto stats_res = compute_partition_stats(assignments, sizes_before);
  assert(stats_res.ok());
  auto score_res = score_partitions(stats_res.value(), 1.0, 0.0, 1.0);
  assert(score_res.ok());
  const PartitionScoreResult score = score_res.value();
  assert(score.append_partitions.size() == 1);
  assert(score.recluster_partitions.size() == 1);
  assert(score.append_partitions[0] == 0);
  assert(score.recluster_partitions[0] == 1);

  // 3) append merge path.
  auto p0_main_res = fx.main_ivf->GetPartitionRecords(fx.main_versions, 0);
  assert(p0_main_res.ok());
  auto append_res = merge_partition_append(0, p0_main_res.value(), assignments[0]);
  assert(append_res.ok());
  assert(append_res.value().size() == p0_main_res.value().size() + assignments[0].size());
  {
    bool has_100 = false;
    for (const auto& rec : append_res.value()) {
      if (rec.doc_id == 100) {
        has_100 = true;
      }
    }
    assert(has_100);
  }

  // 4) local recluster merge path.
  auto p1_main_res = fx.main_ivf->GetPartitionRecords(fx.main_versions, 1);
  assert(p1_main_res.ok());
  MergeOptions options;
  options.alpha = 1.0;
  options.beta = 0.0;
  options.recluster_threshold = 1.0;
  options.local_recluster_k = 2;
  options.local_kmeans_iterations = 5;
  auto recluster_res = merge_partition_recluster(1, p1_main_res.value(), assignments[1], options);
  assert(recluster_res.ok());
  assert(recluster_res.value().size() == p1_main_res.value().size() + assignments[1].size());
  {
    std::unordered_set<DocId> docs;
    for (const auto& rec : recluster_res.value()) {
      docs.insert(rec.doc_id);
    }
    assert(docs.find(2) != docs.end());
    assert(docs.find(101) != docs.end());
    assert(docs.find(102) != docs.end());
  }

  // 5) patch + commit + PQ re-encode in main codebook space.
  auto delta_code_before = fx.delta_ivf->GetDocPQCode(fx.delta_versions, 100);
  assert(delta_code_before.ok());

  auto patch_res =
      prepare_partition_patch(fx.main_ivf, fx.main_versions, assignments, score, options);
  assert(patch_res.ok());
  assert(patch_res.value().partition_ids.size() == 3);
  {
    std::unordered_set<uint32_t> patch_parts(patch_res.value().partition_ids.begin(),
                                             patch_res.value().partition_ids.end());
    assert(patch_parts.find(0) != patch_parts.end());
    assert(patch_parts.find(1) != patch_parts.end());
    assert(patch_parts.find(2) != patch_parts.end());
  }
  assert(commit_partition_patch(fx.main_ivf, fx.main_versions, patch_res.value()).ok());

  auto sizes_after_res = fx.main_ivf->GetPartitionSizes(fx.main_versions);
  assert(sizes_after_res.ok());
  const std::vector<uint32_t> sizes_after = sizes_after_res.value();
  assert(sizes_after[0] == 3);
  assert(sizes_after[1] == 3);
  assert(sizes_after[2] == 1);  // untouched partition remains unchanged.

  auto main_code_after = fx.main_ivf->GetDocPQCode(fx.main_versions, 100);
  assert(main_code_after.ok());
  assert(main_code_after.value() != delta_code_before.value());  // must be re-encoded by Main PQ.

  // 6) unified entry: freeze + assign/stats/score + patch + commit.
  Fixture fx2 = BuildFixture();
  auto report_res =
      merge_frozen_delta_into_main(fx2.main_ivf, fx2.main_versions, fx2.delta_ivf, fx2.delta_versions, options);
  assert(report_res.ok());
  assert(report_res.value().frozen_records == 3);
  assert(report_res.value().patch_partitions == 3);

  auto sizes_after_full_res = fx2.main_ivf->GetPartitionSizes(fx2.main_versions);
  assert(sizes_after_full_res.ok());
  const auto& sizes_after_full = sizes_after_full_res.value();
  assert(sizes_after_full[0] == 3);
  assert(sizes_after_full[1] == 3);
  assert(sizes_after_full[2] == 1);

  // 7) balanced_append should prefer a healthier nearby list when the nearest
  // list is already overloaded relative to local average.
  Fixture fx3 = BuildHealthierNeighborFixture();
  MergeOptions constrained_options;
  constrained_options.alpha = 0.0;
  constrained_options.beta = 0.0;
  constrained_options.recluster_threshold = 1.0;
  constrained_options.assignment_mode = "balanced_append";
  constrained_options.assignment_top_r = 2;
  constrained_options.assignment_gamma = 1.1;
  constrained_options.assignment_hard_cap_ratio = 1.5;
  constrained_options.assignment_lambda = 0.0;
  auto constrained_merge_res = merge_frozen_delta_into_main(
      fx3.main_ivf, fx3.main_versions, fx3.delta_ivf, fx3.delta_versions, constrained_options);
  assert(constrained_merge_res.ok());
  assert(constrained_merge_res.value().patch_partitions == 2);
  auto constrained_sizes_res = fx3.main_ivf->GetPartitionSizes(fx3.main_versions);
  assert(constrained_sizes_res.ok());
  const auto& constrained_sizes = constrained_sizes_res.value();
  assert(constrained_sizes.size() == 2);
  assert(constrained_sizes[0] == 7);
  assert(constrained_sizes[1] == 4);

  return 0;
}
