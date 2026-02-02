#include <algorithm>
#include <iostream>
#include <optional>
#include <random>

#include <Eigen/Dense>

#include "common/config.h"
#include "common/dataset.h"
#include "common/result.h"
#include "common/types.h"
#include "index/ivf.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

MatrixRM GenerateRandomMatrix(uint32_t rows, uint32_t cols, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  MatrixRM m(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      m(r, c) = dist(gen);
    }
  }
  return m;
}

void LogStatus(const Status& status, const std::string& step) {
  if (!status.ok()) {
    std::cerr << "[ERROR] " << step << ": " << status.ToString() << std::endl;
  } else {
    std::cout << "[INFO] " << step << " OK" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1) ? argv[1] : "configs/base.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  auto whitening = CreateWhiteningModel();
  auto ivf = CreateIVFIndex();

  std::optional<std::string> base_dataset_path;
  if (argc > 2) {
    auto resolved = ResolveFvecsPath(argv[2], "_base.fvecs");
    if (!resolved.ok()) {
      std::cerr << resolved.status().ToString() << std::endl;
      return 1;
    }
    base_dataset_path = resolved.value();
    std::cout << "[INFO] Using base dataset: " << *base_dataset_path << std::endl;
  }

  MatrixRM X;
  uint32_t num_vectors = 0;
  if (base_dataset_path) {
    auto matrix_res = LoadFvecs(*base_dataset_path);
    if (!matrix_res.ok()) {
      std::cerr << matrix_res.status().ToString() << std::endl;
      return 1;
    }
    X = matrix_res.value();
    num_vectors = static_cast<uint32_t>(X.rows());
    if (num_vectors == 0) {
      std::cerr << "Base dataset contains no vectors." << std::endl;
      return 1;
    }
    if (config.dim != static_cast<uint32_t>(X.cols())) {
      std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << X.cols() << std::endl;
      config.dim = static_cast<uint32_t>(X.cols());
    }
  } else {
    num_vectors = 32;
    X = GenerateRandomMatrix(num_vectors, config.dim, config.seed);
  }

  MatrixRM X_for_ivf = X;
  VersionId whiten_version = 0;
  if (config.use_whitening) {
    auto whiten_version_res = whitening->Fit(X);
    if (!whiten_version_res.ok()) {
      std::cerr << whiten_version_res.status().ToString() << std::endl;
      return 1;
    }
    whiten_version = whiten_version_res.value();
    auto batch_res = whitening->TransformBatch(X, whiten_version);
    if (!batch_res.ok()) {
      std::cerr << batch_res.status().ToString() << std::endl;
      return 1;
    }
    X_for_ivf = batch_res.value();
  }

  std::vector<DocId> ids(num_vectors);
  for (uint32_t i = 0; i < num_vectors; ++i) {
    ids[i] = i;
  }

  IVFParams ivf_params;
  ivf_params.nlist = std::max(1u, config.ivf_nlist);
  ivf_params.dim = config.dim;
  ivf_params.pq.enable = config.pq_enable;
  ivf_params.pq.M = config.pq_m;
  ivf_params.pq.nbits = config.pq_nbits;
  ivf_params.pq.residual = config.pq_residual;
  auto ivf_version_res = ivf->Build(X_for_ivf, ids, ivf_params, 1);
  if (!ivf_version_res.ok()) {
    std::cerr << ivf_version_res.status().ToString() << std::endl;
    return 1;
  }
  VersionId ivf_version = ivf_version_res.value();

  AlignedVector<VectorRecord> records;
  records.reserve(num_vectors);
  for (uint32_t i = 0; i < num_vectors; ++i) {
    VectorRecord rec;
    rec.doc_id = ids[i];
    rec.dim = config.dim;
    rec.versions = VersionSet{whiten_version, ivf_version};
    rec.ivf_id = i % std::max(1u, ivf_params.nlist);
    Eigen::VectorXf original = X.row(i).transpose();
    Eigen::VectorXf xw = original;
    if (config.use_whitening) {
      auto transform_res = whitening->Transform(original, whiten_version, xw);
      if (!transform_res.ok()) {
        std::cerr << transform_res.status().ToString() << std::endl;
        return 1;
      }
    }
    rec.x = xw;

    records.push_back(rec);
  }

  Status add_status = ivf->Add(records);
  LogStatus(add_status, "IVF::Add");

  auto bytes = whitening->Serialize();
  if (!bytes.ok()) {
    std::cerr << bytes.status().ToString() << std::endl;
  }
  std::cout << "Serialized whitening bytes: " << bytes.value().size() << std::endl;

  return add_status.ok() ? 0 : 1;
}
