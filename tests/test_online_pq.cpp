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
      residual.segment(static_cast<Eigen::Index>(m * old_state_before.value().dsub),
                       static_cast<Eigen::Index>(old_state_before.value().dsub)) = sub;
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

  return 0;
}
