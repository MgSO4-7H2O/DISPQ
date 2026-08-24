#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/serialization.h"
#include "common/types.h"
#include "eval/gt.h"
#include "eval/metrics.h"
#include "index/ivf.h"
#include "index/postings.h"
#include "search/hybrid_search.h"
#include "search/rerank.h"
#include "whitening/whitening.h"

using namespace ann;

int main() {
  Config cfg;
  cfg.dim = 8;
  MatrixRM X = MatrixRM::Random(4, cfg.dim);

  auto whitening = CreateWhiteningModel();
  auto ivf = CreateIVFIndex();

  auto whiten_version = whitening->Fit(X);
  assert(whiten_version.ok());
  auto batch_whiten = whitening->TransformBatch(X, whiten_version.value());
  assert(batch_whiten.ok());
  MatrixRM whitened = batch_whiten.value();
  assert(whitened.rows() == X.rows());
  assert(whitened.cols() == X.cols());

  Eigen::VectorXf sample = X.row(0).transpose();
  Eigen::VectorXf xw(cfg.dim);
  auto transform_status = whitening->Transform(sample, whiten_version.value(), xw);
  assert(transform_status.ok());
  auto bridge = whitening->Bridge(whiten_version.value(), whiten_version.value());
  assert(bridge.ok());

  std::vector<DocId> ids = {0, 1, 2, 3};
  IVFParams ivf_params;
  ivf_params.nlist = 2;
  ivf_params.dim = cfg.dim;
  auto ivf_version = ivf->Build(whitened, ids, ivf_params, 1);
  assert(ivf_version.ok());

  VectorRecord rec;
  rec.doc_id = 0;
  rec.dim = cfg.dim;
  rec.versions = VersionSet{whiten_version.value(), ivf_version.value()};
  rec.ivf_id = 0;
  rec.x = xw;
  AlignedVector<VectorRecord> recs = {rec};
  assert(ivf->Add(recs).ok());

  VersionSet routes = rec.versions;
  auto ingest_profile = ivf->GetLastIngestProfiling(routes);
  assert(ingest_profile.ok());
  assert(ingest_profile.value().records == recs.size());
  assert(ingest_profile.value().assignment_us >= 0.0);

  auto search_res = ivf->Search(xw, 2, 1, routes, 0);
  assert(search_res.ok());
  assert(search_res.value().scanned_candidates >= search_res.value().topk.size());

  VersionSet bad_routes = routes;
  bad_routes.index_version = 999;
  auto bad_search = ivf->Search(xw, 1, 1, bad_routes, 0);
  assert(!bad_search.ok());

  auto hybrid_res = CreateHybridSearcher(cfg);
  assert(hybrid_res.ok());
  auto searcher = std::move(hybrid_res.value());
  VersionSet route_versions{whiten_version.value(), ivf_version.value()};
  assert(searcher->SetIndex(ivf, route_versions).ok());
  assert(searcher->SetWhitening(whitening, whiten_version.value()).ok());
  SearchParams search_params;
  search_params.topk = 2;
  search_params.nprobe = 1;
  search_params.use_whitening = true;
  auto hybrid_search = searcher->Search(sample, search_params);
  assert(hybrid_search.ok());

  PostingStore posting_store;
  assert(posting_store.AddPosting(routes.index_version, 0, rec).ok());
  auto posting_view = posting_store.GetPostingList(routes.index_version, 0);
  assert(posting_view.ok());

  std::vector<uint8_t> payload = {1, 2, 3};
  const std::string tmp_file = "test_serialization.bin";
  auto save_status = SaveBinary(tmp_file, payload);
  assert(save_status.ok());
  auto load_res = LoadBinary(tmp_file);
  assert(load_res.ok());
  std::remove(tmp_file.c_str());

  std::vector<std::vector<DocId>> gt = {{0, 1}, {1, 2}};
  std::vector<std::vector<DocId>> pred = {{1, 0}, {0, 2}};
  auto recall = RecallAtK(gt, pred, 2);
  assert(recall.ok());

  auto gt_res = ComputeGroundTruth(X, X, 2);
  assert(gt_res.ok());

  std::vector<Candidate> cands(1);
  cands[0].doc_id = 0;
  auto rerank_status = RerankL2(sample, &cands, [&](DocId) -> Result<Eigen::VectorXf> {
    return sample;
  }, [&](DocId) -> Result<float> {
    return sample.squaredNorm();
  });
  assert(rerank_status.ok());
  assert(std::fabs(cands[0].rerank_dist) < 1e-6f);

  for (bool residual : {true, false}) {
    constexpr uint32_t kSoADim = 8;
    MatrixRM training = MatrixRM::Zero(1, kSoADim);
    std::vector<DocId> training_ids = {900};

    IVFParams pq_params;
    pq_params.nlist = 1;
    pq_params.dim = kSoADim;
    pq_params.kmeans_iterations = 1;
    pq_params.pq.enable = true;
    pq_params.pq.M = 1;
    pq_params.pq.nbits = 8;
    pq_params.pq.residual = residual;
    pq_params.use_fixed_routing_centroids = true;
    pq_params.fixed_routing_centroids = MatrixRM::Zero(1, kSoADim);
    pq_params.use_fixed_pq_codebooks = true;
    MatrixRM codebook = MatrixRM::Zero(256, kSoADim);
    for (int k = 0; k < 256; ++k) {
      codebook(k, 0) = static_cast<float>(k);
    }
    pq_params.fixed_pq_codebooks = {codebook};

    auto pq_ivf = CreateIVFIndex();
    auto pq_version = pq_ivf->Build(training, training_ids, pq_params, residual ? 201 : 202);
    assert(pq_version.ok());
    VersionSet pq_routes{1, pq_version.value()};

    AlignedVector<VectorRecord> pq_records;
    for (const auto& doc_value : {std::pair<DocId, float>{100, 30.0f},
                                  std::pair<DocId, float>{101, 10.0f},
                                  std::pair<DocId, float>{102, 20.0f}}) {
      VectorRecord r;
      r.doc_id = doc_value.first;
      r.dim = kSoADim;
      r.versions = pq_routes;
      r.ivf_id = 0;
      r.x = Eigen::VectorXf::Zero(kSoADim);
      r.x(0) = doc_value.second;
      pq_records.push_back(std::move(r));
    }
    assert(pq_ivf->Add(pq_records).ok());

    Eigen::VectorXf pq_query = Eigen::VectorXf::Zero(kSoADim);
    pq_query(0) = 10.0f;
    auto pq_search = pq_ivf->Search(pq_query, 3, 1, pq_routes, 0);
    assert(pq_search.ok());
    assert(!pq_search.value().topk.empty());
    assert(pq_search.value().topk[0].doc_id == 101);
    assert(std::fabs(pq_search.value().topk[0].approx_dist) < 1e-5f);

    auto payload_res = pq_ivf->Serialize();
    assert(payload_res.ok());
    auto restored = CreateIVFIndex();
    assert(restored->Deserialize(payload_res.value()).ok());
    auto restored_search = restored->Search(pq_query, 3, 1, pq_routes, 0);
    assert(restored_search.ok());
    assert(!restored_search.value().topk.empty());
    assert(restored_search.value().topk[0].doc_id == 101);
    assert(std::fabs(restored_search.value().topk[0].approx_dist) < 1e-5f);
  }

  {
    const std::string tmp_fvecs = "test_vectors.fvecs";
    {
      std::ofstream ofs(tmp_fvecs, std::ios::binary | std::ios::trunc);
      auto write_vec = [&](std::initializer_list<float> vals) {
        int32_t dim = static_cast<int32_t>(vals.size());
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
        for (float v : vals) {
          ofs.write(reinterpret_cast<const char*>(&v), sizeof(float));
        }
      };
      write_vec({1.0f, 2.0f});
      write_vec({3.0f, 4.0f});
    }
    auto matrix_res = LoadFvecs(tmp_fvecs);
    assert(matrix_res.ok());
    const MatrixRM mat = matrix_res.value();
    assert(mat.rows() == 2);
    assert(mat.cols() == 2);
    assert(mat(0, 0) == 1.0f);
    std::remove(tmp_fvecs.c_str());
  }

  return 0;
}
