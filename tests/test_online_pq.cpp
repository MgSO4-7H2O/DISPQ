#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "common/types.h"
#include "index/ivf.h"

using namespace ann;

namespace {

AlignedVector<VectorRecord> MakeRecords(Eigen::Ref<const MatrixRM> X,
                                        DocId start_id,
                                        uint32_t dim,
                                        const VersionSet& versions) {
  AlignedVector<VectorRecord> out;
  out.reserve(static_cast<size_t>(X.rows()));
  for (int64_t i = 0; i < X.rows(); ++i) {
    VectorRecord rec;
    rec.doc_id = start_id + static_cast<DocId>(i);
    rec.dim = dim;
    rec.versions = versions;
    rec.ivf_id = 0;
    rec.x = X.row(i).transpose();
    out.push_back(std::move(rec));
  }
  return out;
}

double FindApproxDistByDoc(const SearchResult& result, DocId doc_id) {
  for (const auto& cand : result.topk) {
    if (cand.doc_id == doc_id) {
      return static_cast<double>(cand.approx_dist);
    }
  }
  return std::numeric_limits<double>::infinity();
}

bool MatrixAlmostEqual(Eigen::Ref<const MatrixRM> a, Eigen::Ref<const MatrixRM> b, double eps) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return false;
  }
  return (a - b).cwiseAbs().maxCoeff() <= static_cast<float>(eps);
}

}  // namespace

int main() {
  constexpr uint32_t kDim = 8;
  constexpr uint32_t kM = 4;
  constexpr uint32_t kNBits = 2;
  constexpr uint32_t kNList = 2;

  MatrixRM train = MatrixRM::Random(64, kDim);
  std::vector<DocId> train_ids(static_cast<size_t>(train.rows()));
  for (int64_t i = 0; i < train.rows(); ++i) {
    train_ids[static_cast<size_t>(i)] = static_cast<DocId>(i);
  }

  IVFParams params;
  params.nlist = kNList;
  params.dim = kDim;
  params.pq.enable = true;
  params.pq.M = kM;
  params.pq.nbits = kNBits;
  params.pq.residual = true;

  // 1) No-update path should match Add() behavior.
  auto ivf_ref = CreateIVFIndex();
  auto ivf_online = CreateIVFIndex();
  auto ver_ref = ivf_ref->Build(train, train_ids, params, 100);
  auto ver_online = ivf_online->Build(train, train_ids, params, 101);
  assert(ver_ref.ok() && ver_online.ok());
  VersionSet versions_ref{0, ver_ref.value()};
  VersionSet versions_online{0, ver_online.value()};

  MatrixRM batch_same = MatrixRM::Random(24, kDim);
  auto records_ref = MakeRecords(batch_same, 10000, kDim, versions_ref);
  auto records_online = MakeRecords(batch_same, 10000, kDim, versions_online);

  auto state_before_no_update = ivf_online->GetPQRuntimeState(versions_online);
  assert(state_before_no_update.ok());

  assert(ivf_ref->Add(records_ref).ok());

  OnlinePQUpdateOptions no_update_opt;
  no_update_opt.enable = true;
  no_update_opt.force_update_interval = 0;  // disabled
  no_update_opt.ema_alpha = 0.1;
  no_update_opt.nqe_eps = 1e-6;
  auto no_update_res = ivf_online->AddWithOnlinePQ(records_online, no_update_opt);
  assert(no_update_res.ok());
  assert(!no_update_res.value().updated_codebook);

  auto state_after_no_update = ivf_online->GetPQRuntimeState(versions_online);
  assert(state_after_no_update.ok());
  const auto& sb = state_before_no_update.value();
  const auto& sa = state_after_no_update.value();
  assert(sb.counts == sa.counts);
  assert(sb.codebooks.size() == sa.codebooks.size());
  for (size_t m = 0; m < sb.codebooks.size(); ++m) {
    assert(MatrixAlmostEqual(sb.codebooks[m], sa.codebooks[m], 1e-6));
  }

  for (int i = 0; i < 5; ++i) {
    Eigen::VectorXf q = MatrixRM::Random(1, kDim).row(0).transpose();
    auto sr_ref = ivf_ref->Search(q, 10, 2, versions_ref, 0);
    auto sr_online = ivf_online->Search(q, 10, 2, versions_online, 0);
    assert(sr_ref.ok() && sr_online.ok());
    assert(sr_ref.value().topk.size() == sr_online.value().topk.size());
    for (size_t j = 0; j < sr_ref.value().topk.size(); ++j) {
      assert(sr_ref.value().topk[j].doc_id == sr_online.value().topk[j].doc_id);
      assert(std::fabs(static_cast<double>(sr_ref.value().topk[j].approx_dist) -
                       static_cast<double>(sr_online.value().topk[j].approx_dist)) < 1e-5);
    }
  }

  // 2) main -> delta fixed residual PQ state initialization (codebook + counts).
  auto main_ivf = CreateIVFIndex();
  auto main_ver = main_ivf->Build(train, train_ids, params, 200);
  assert(main_ver.ok());
  VersionSet main_versions{0, main_ver.value()};
  auto main_state = main_ivf->GetPQRuntimeState(main_versions);
  assert(main_state.ok());
  auto centroids_res = main_ivf->GetRoutingCentroids(main_versions);
  assert(centroids_res.ok());

  IVFParams delta_params = params;
  delta_params.use_fixed_routing_centroids = true;
  delta_params.fixed_routing_centroids = centroids_res.value();
  delta_params.use_fixed_pq_codebooks = true;
  delta_params.fixed_pq_codebooks = main_state.value().codebooks;
  delta_params.fixed_pq_counts = main_state.value().counts;
  delta_params.fixed_pq_baseline_nqe = main_state.value().nqe_baseline;
  delta_params.fixed_pq_ema_nqe = main_state.value().nqe_ema;

  auto delta_ivf = CreateIVFIndex();
  auto delta_ver = delta_ivf->Build(train, train_ids, delta_params, 201);
  assert(delta_ver.ok());
  VersionSet delta_versions{0, delta_ver.value()};
  auto delta_state = delta_ivf->GetPQRuntimeState(delta_versions);
  assert(delta_state.ok());
  assert(delta_state.value().counts == main_state.value().counts);
  assert(delta_state.value().codebooks.size() == main_state.value().codebooks.size());
  for (size_t m = 0; m < main_state.value().codebooks.size(); ++m) {
    assert(MatrixAlmostEqual(main_state.value().codebooks[m], delta_state.value().codebooks[m], 1e-6));
  }

  // 3) Triggered update should change codebook and pq_counts.
  auto state_before_update = delta_ivf->GetPQRuntimeState(delta_versions);
  assert(state_before_update.ok());
  MatrixRM update_batch = MatrixRM::Random(96, kDim);
  update_batch.array() += 2.0f;
  auto update_records = MakeRecords(update_batch, 30000, kDim, delta_versions);
  OnlinePQUpdateOptions update_opt;
  update_opt.enable = true;
  update_opt.force_update_interval = 1;  // trigger every batch
  update_opt.ema_alpha = 1.0;
  update_opt.nqe_eps = 1e-6;
  update_opt.reencode_batch_after_update = true;
  auto update_res = delta_ivf->AddWithOnlinePQ(update_records, update_opt);
  assert(update_res.ok());
  assert(update_res.value().trigger_update);
  assert(update_res.value().updated_codebook);
  auto state_after_update = delta_ivf->GetPQRuntimeState(delta_versions);
  assert(state_after_update.ok());

  bool count_increased = false;
  bool codebook_changed = false;
  for (size_t m = 0; m < state_before_update.value().counts.size(); ++m) {
    for (size_t k = 0; k < state_before_update.value().counts[m].size(); ++k) {
      if (state_after_update.value().counts[m][k] > state_before_update.value().counts[m][k]) {
        count_increased = true;
      }
    }
    if (!MatrixAlmostEqual(state_before_update.value().codebooks[m],
                           state_after_update.value().codebooks[m],
                           1e-6)) {
      codebook_changed = true;
    }
  }
  assert(count_increased);
  assert(codebook_changed);

  // 4) Old codes are not rewritten, but search uses latest codebook.
  IVFParams small_params;
  small_params.nlist = 1;
  small_params.dim = kDim;
  small_params.pq.enable = true;
  small_params.pq.M = kM;
  small_params.pq.nbits = kNBits;
  small_params.pq.residual = true;

  MatrixRM small_train = MatrixRM::Random(32, kDim);
  std::vector<DocId> small_ids(static_cast<size_t>(small_train.rows()));
  for (int64_t i = 0; i < small_train.rows(); ++i) {
    small_ids[static_cast<size_t>(i)] = static_cast<DocId>(i);
  }

  auto old_test_ivf = CreateIVFIndex();
  auto old_ver = old_test_ivf->Build(small_train, small_ids, small_params, 300);
  assert(old_ver.ok());
  VersionSet old_versions{0, old_ver.value()};
  auto old_centroids = old_test_ivf->GetRoutingCentroids(old_versions);
  assert(old_centroids.ok());

  MatrixRM old_doc_mat(1, kDim);
  old_doc_mat.row(0) = small_train.row(0);
  auto old_doc_recs = MakeRecords(old_doc_mat, 90000, kDim, old_versions);
  assert(old_test_ivf->Add(old_doc_recs).ok());

  auto old_code_before = old_test_ivf->GetDocPQCode(old_versions, 90000);
  assert(old_code_before.ok());
  auto old_state_before = old_test_ivf->GetPQRuntimeState(old_versions);
  assert(old_state_before.ok());
  const Eigen::VectorXf q_old = old_doc_mat.row(0).transpose();
  auto search_before = old_test_ivf->Search(q_old, 128, 1, old_versions, 0);
  assert(search_before.ok());
  const double dist_before = FindApproxDistByDoc(search_before.value(), 90000);
  assert(std::isfinite(dist_before));

  // Create a batch near the old codewords so the same code indices are selected and updated.
  MatrixRM codeword_guided_batch(128, kDim);
  const Eigen::VectorXf centroid0 = old_centroids.value().row(0).transpose();
  for (int64_t i = 0; i < codeword_guided_batch.rows(); ++i) {
    Eigen::VectorXf residual(kDim);
    for (uint32_t m = 0; m < kM; ++m) {
      const uint8_t code = old_code_before.value()[static_cast<size_t>(m)];
      Eigen::VectorXf sub =
          old_state_before.value().codebooks[static_cast<size_t>(m)]
              .row(static_cast<Eigen::Index>(code))
              .transpose();
      sub.array() += 0.05f;
      const uint32_t offset = old_state_before.value().sub_offsets[static_cast<size_t>(m)];
      const uint32_t subdim = old_state_before.value().sub_offsets[static_cast<size_t>(m) + 1] -
                              offset;
      residual.segment(static_cast<Eigen::Index>(offset), static_cast<Eigen::Index>(subdim)) = sub;
    }
    codeword_guided_batch.row(i) = (residual + centroid0).transpose();
  }
  auto guided_recs = MakeRecords(codeword_guided_batch, 91000, kDim, old_versions);
  OnlinePQUpdateOptions guided_opt;
  guided_opt.enable = true;
  guided_opt.force_update_interval = 1;  // trigger every batch
  guided_opt.ema_alpha = 1.0;
  guided_opt.nqe_eps = 1e-6;
  guided_opt.reencode_batch_after_update = true;
  auto guided_update = old_test_ivf->AddWithOnlinePQ(guided_recs, guided_opt);
  assert(guided_update.ok());
  assert(guided_update.value().updated_codebook);

  auto old_code_after = old_test_ivf->GetDocPQCode(old_versions, 90000);
  assert(old_code_after.ok());
  assert(old_code_after.value() == old_code_before.value());  // old code index not rewritten

  auto old_state_after = old_test_ivf->GetPQRuntimeState(old_versions);
  assert(old_state_after.ok());
  bool old_codeword_changed = false;
  for (uint32_t m = 0; m < kM; ++m) {
    const uint8_t code = old_code_before.value()[static_cast<size_t>(m)];
    const Eigen::VectorXf before =
        old_state_before.value().codebooks[static_cast<size_t>(m)]
            .row(static_cast<Eigen::Index>(code))
            .transpose();
    const Eigen::VectorXf after =
        old_state_after.value().codebooks[static_cast<size_t>(m)]
            .row(static_cast<Eigen::Index>(code))
            .transpose();
    if ((before - after).norm() > 1e-6f) {
      old_codeword_changed = true;
      break;
    }
  }
  assert(old_codeword_changed);

  auto search_after = old_test_ivf->Search(q_old, 256, 1, old_versions, 0);
  assert(search_after.ok());
  const double dist_after = FindApproxDistByDoc(search_after.value(), 90000);
  assert(std::isfinite(dist_after));
  assert(std::fabs(dist_after - dist_before) > 1e-6);  // distance table uses updated codebook

  // 5) Sliding-window step supports insert + delete in one online update.
  auto sliding_ivf = CreateIVFIndex();
  auto sliding_ver = sliding_ivf->Build(small_train, small_ids, small_params, 400);
  assert(sliding_ver.ok());
  VersionSet sliding_versions{0, sliding_ver.value()};

  MatrixRM init_docs = small_train.topRows(8);
  auto init_recs = MakeRecords(init_docs, 100000, kDim, sliding_versions);
  assert(sliding_ivf->Add(init_recs).ok());
  auto state_before_slide = sliding_ivf->GetPQRuntimeState(sliding_versions);
  assert(state_before_slide.ok());
  assert(state_before_slide.value().ntotal == 8);

  MatrixRM ins_docs = MatrixRM::Random(4, kDim);
  auto ins_recs = MakeRecords(ins_docs, 200000, kDim, sliding_versions);
  std::vector<DocId> delete_ids{100000, 100001, 100002, 100003};
  OnlinePQUpdateOptions slide_opt;
  slide_opt.enable = true;
  slide_opt.force_update_interval = 1;
  slide_opt.ema_alpha = 1.0;
  slide_opt.nqe_eps = 1e-6;
  auto slide_res =
      sliding_ivf->AddWithOnlinePQSlidingWindow(ins_recs, delete_ids, slide_opt);
  assert(slide_res.ok());
  assert(slide_res.value().deleted_vectors == delete_ids.size());

  auto state_after_slide = sliding_ivf->GetPQRuntimeState(sliding_versions);
  assert(state_after_slide.ok());
  assert(state_after_slide.value().ntotal == 8);
  auto deleted_code = sliding_ivf->GetDocPQCode(sliding_versions, 100000);
  assert(!deleted_code.ok());
  auto inserted_code = sliding_ivf->GetDocPQCode(sliding_versions, 200000);
  assert(inserted_code.ok());

  // 6) Non-divisible layouts are rejected; MSTuring dim=100, M=50 uses generic 2x256 SoA.
  constexpr uint32_t kMSTDim = 100;
  constexpr uint32_t kMSTM = 50;
  MatrixRM mst_train = MatrixRM::Random(288, kMSTDim);
  std::vector<DocId> mst_train_ids(static_cast<size_t>(mst_train.rows()));
  for (int64_t i = 0; i < mst_train.rows(); ++i) {
    mst_train_ids[static_cast<size_t>(i)] = static_cast<DocId>(i);
  }
  IVFParams mst_params;
  mst_params.nlist = 4;
  mst_params.dim = kMSTDim;
  mst_params.kmeans_iterations = 1;
  mst_params.pq.enable = true;
  mst_params.pq.M = kMSTM;
  mst_params.pq.nbits = 8;
  mst_params.pq.residual = true;

  IVFParams non_divisible_params = mst_params;
  non_divisible_params.pq.M = 16;
  auto rejected_ivf = CreateIVFIndex();
  auto rejected_ver =
      rejected_ivf->Build(mst_train, mst_train_ids, non_divisible_params, 499);
  assert(!rejected_ver.ok());

  auto mst_ivf = CreateIVFIndex();
  auto mst_ver = mst_ivf->Build(mst_train, mst_train_ids, mst_params, 500);
  assert(mst_ver.ok());
  VersionSet mst_versions{0, mst_ver.value()};
  auto mst_state = mst_ivf->GetPQRuntimeState(mst_versions);
  assert(mst_state.ok());
  assert(mst_state.value().dsub == 2);
  assert(mst_state.value().sub_offsets.size() == kMSTM + 1);
  assert(mst_state.value().sub_offsets.front() == 0);
  assert(mst_state.value().sub_offsets.back() == kMSTDim);
  for (uint32_t m = 0; m < kMSTM; ++m) {
    const uint32_t subdim = mst_state.value().sub_offsets[static_cast<size_t>(m) + 1] -
                            mst_state.value().sub_offsets[static_cast<size_t>(m)];
    assert(subdim == 2);
    assert(mst_state.value().codebooks[static_cast<size_t>(m)].rows() == 256);
    assert(mst_state.value().codebooks[static_cast<size_t>(m)].cols() == subdim);
  }

  MatrixRM mst_initial = MatrixRM::Random(32, kMSTDim);
  auto mst_initial_recs = MakeRecords(mst_initial, 300000, kMSTDim, mst_versions);
  assert(mst_ivf->Add(mst_initial_recs).ok());
  auto mst_code = mst_ivf->GetDocPQCode(mst_versions, 300000);
  assert(mst_code.ok() && mst_code.value().size() == kMSTM);
  Eigen::VectorXf mst_query = mst_initial.row(0).transpose();
  auto mst_search = mst_ivf->Search(mst_query, 10, 4, mst_versions, 0);
  assert(mst_search.ok() && !mst_search.value().topk.empty());

  MatrixRM mst_insert = MatrixRM::Random(8, kMSTDim);
  auto mst_insert_recs = MakeRecords(mst_insert, 400000, kMSTDim, mst_versions);
  AlignedVector<VectorRecord> mst_delete_recs(
      mst_initial_recs.begin(), mst_initial_recs.begin() + 8);
  OnlinePQUpdateOptions mst_slide_opt;
  mst_slide_opt.enable = true;
  mst_slide_opt.force_update_interval = 1;
  mst_slide_opt.ema_alpha = 1.0;
  mst_slide_opt.nqe_eps = 1e-6;
  mst_slide_opt.reencode_batch_after_update = true;
  auto mst_slide = mst_ivf->AddWithOnlinePQSlidingWindowRecords(
      mst_insert_recs, mst_delete_recs, mst_slide_opt);
  assert(mst_slide.ok());
  assert(mst_slide.value().deleted_vectors == mst_delete_recs.size());
  auto mst_ids = mst_ivf->SnapshotDocIds(mst_versions);
  assert(mst_ids.ok() && mst_ids.value().size() == 32);
  assert(!mst_ivf->GetDocPQCode(mst_versions, 300000).ok());
  assert(mst_ivf->GetDocPQCode(mst_versions, 400000).ok());

  // Re-encode one complete partition through the merge patch path.
  uint32_t patched_partition = 0;
  std::vector<DocId> patched_ids;
  for (; patched_partition < mst_params.nlist; ++patched_partition) {
    auto ids_res = mst_ivf->GetPartitionDocIds(mst_versions, patched_partition);
    assert(ids_res.ok());
    if (!ids_res.value().empty()) {
      patched_ids = std::move(ids_res.value());
      break;
    }
  }
  assert(!patched_ids.empty());
  AlignedVector<VectorRecord> patch_records;
  patch_records.reserve(patched_ids.size());
  for (DocId doc_id : patched_ids) {
    VectorRecord patch_rec;
    patch_rec.doc_id = doc_id;
    patch_rec.dim = kMSTDim;
    patch_rec.versions = mst_versions;
    if (doc_id >= 300000 && doc_id < 300032) {
      patch_rec.x = mst_initial.row(static_cast<Eigen::Index>(doc_id - 300000)).transpose();
    } else {
      assert(doc_id >= 400000 && doc_id < 400008);
      patch_rec.x = mst_insert.row(static_cast<Eigen::Index>(doc_id - 400000)).transpose();
    }
    patch_records.push_back(std::move(patch_rec));
  }
  PartitionPatch mst_patch;
  mst_patch.partition_ids.push_back(patched_partition);
  mst_patch.replacement_records.push_back(std::move(patch_records));
  assert(mst_ivf->CommitPartitionPatch(mst_versions, mst_patch).ok());
  auto mst_ids_after_patch = mst_ivf->SnapshotDocIds(mst_versions);
  assert(mst_ids_after_patch.ok() && mst_ids_after_patch.value().size() == 32);

  // 7) SIFT layout remains the uniform 16x8 fast path.
  constexpr uint32_t kSiftDim = 128;
  IVFParams sift_params;
  sift_params.nlist = 1;
  sift_params.dim = kSiftDim;
  sift_params.pq.enable = true;
  sift_params.pq.M = 16;
  sift_params.pq.nbits = 8;
  sift_params.pq.residual = true;
  sift_params.use_fixed_routing_centroids = true;
  sift_params.fixed_routing_centroids = MatrixRM::Zero(1, kSiftDim);
  sift_params.use_fixed_pq_codebooks = true;
  sift_params.fixed_pq_codebooks.assign(16, MatrixRM::Random(256, 8));
  sift_params.fixed_pq_counts.assign(16, std::vector<uint64_t>(256, 1));

  MatrixRM sift_seed = MatrixRM::Random(1, kSiftDim);
  auto sift_ivf = CreateIVFIndex();
  auto sift_ver = sift_ivf->Build(sift_seed, std::vector<DocId>{0}, sift_params, 600);
  assert(sift_ver.ok());
  VersionSet sift_versions{0, sift_ver.value()};
  auto sift_state = sift_ivf->GetPQRuntimeState(sift_versions);
  assert(sift_state.ok());
  assert(sift_state.value().dsub == 8);
  assert(sift_state.value().sub_offsets.size() == 17);
  for (uint32_t m = 0; m < 16; ++m) {
    assert(sift_state.value().sub_offsets[static_cast<size_t>(m)] == m * 8);
  }
  auto sift_recs = MakeRecords(sift_seed, 500000, kSiftDim, sift_versions);
  assert(sift_ivf->Add(sift_recs).ok());
  auto sift_code = sift_ivf->GetDocPQCode(sift_versions, 500000);
  assert(sift_code.ok() && sift_code.value().size() == 16);
  Eigen::VectorXf sift_query = sift_seed.row(0).transpose();
  assert(sift_ivf->Search(sift_query, 1, 1, sift_versions, 0).ok());

  return 0;
}
