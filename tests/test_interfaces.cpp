#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

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
  });
  assert(rerank_status.ok());

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
