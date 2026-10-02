#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/config.h"
#include "common/dataset.h"
#include "common/prebuilt_index.h"
#include "common/timer.h"
#include "common/types.h"
#include "index/ivf.h"
#include "whitening/whitening.h"

using namespace ann;

namespace {

constexpr uint32_t kAddBlockRows = 65536;

MatrixRM GenerateRandom(uint32_t rows, uint32_t cols, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  MatrixRM m(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      m(r, c) = dist(gen);
    }
  }
  return m;
}

void NormalizeRowsL2(MatrixRM* X) {
  if (X == nullptr) {
    return;
  }
  for (Eigen::Index i = 0; i < X->rows(); ++i) {
    const float norm = X->row(i).norm();
    if (norm > 0.0f) {
      X->row(i) /= norm;
    }
  }
}

uint32_t ResolveMainRows(const Config& config, uint32_t total_rows) {
  if (total_rows == 0) {
    return 0;
  }
  if (!config.enable_streaming) {
    return total_rows;
  }
  uint32_t main_rows = config.main_index_rows;
  if (main_rows == 0) {
    main_rows = std::max<uint32_t>(1, total_rows / 2);
  }
  if (total_rows > 1 && main_rows >= total_rows) {
    main_rows = total_rows - 1;
  }
  return std::max<uint32_t>(1, main_rows);
}

std::string SanitizePathPart(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) || ch == '-' || ch == '_') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "default" : out;
}

std::string DatasetLabelFromPath(const std::string& base_dataset_path) {
  std::filesystem::path ds_path(base_dataset_path);
  const std::string parent_name = ds_path.parent_path().filename().string();
  if (!parent_name.empty()) {
    return parent_name;
  }
  const std::string stem = ds_path.stem().string();
  return stem.empty() ? "dataset" : stem;
}

std::string DefaultPrebuiltIndexDir(const std::string& config_path,
                                    const std::string& dataset_label) {
  const std::string config_stem =
      std::filesystem::path(config_path).stem().string().empty()
          ? "config"
          : std::filesystem::path(config_path).stem().string();
  return (std::filesystem::path("index") / SanitizePathPart(dataset_label) /
          SanitizePathPart(config_stem))
      .string();
}

Status AddRangeToIndex(const std::shared_ptr<IVFIndex>& ivf,
                       Eigen::Ref<const MatrixRM> x_whitened,
                       uint32_t begin,
                       uint32_t end,
                       uint32_t dim,
                       const VersionSet& versions) {
  if (!ivf) {
    return Status::InvalidArgument("AddRangeToIndex: null index");
  }
  if (begin > end || static_cast<Eigen::Index>(end) > x_whitened.rows()) {
    return Status::InvalidArgument("AddRangeToIndex: invalid row range");
  }
  for (uint32_t block_begin = begin; block_begin < end;) {
    const uint32_t block_end =
        std::min<uint32_t>(end, block_begin + kAddBlockRows);
    const uint32_t count = block_end - block_begin;
    AlignedVector<VectorRecord> records(static_cast<size_t>(count));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t offset = 0; offset < static_cast<int64_t>(count); ++offset) {
      const uint32_t row = block_begin + static_cast<uint32_t>(offset);
      VectorRecord rec;
      rec.doc_id = row;
      rec.dim = dim;
      rec.versions = versions;
      rec.ivf_id = 0;
      rec.x = x_whitened.row(static_cast<Eigen::Index>(row)).transpose();
      records[static_cast<size_t>(offset)] = std::move(rec);
    }
    const bool finalize = (block_end == end);
    Status status = ivf->AddBatch(records, finalize);
    if (!status.ok()) {
      return status;
    }
    block_begin = block_end;
  }
  return Status::OK();
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [config.json] [dataset_dir_or_base.fvecs] [output_index_dir]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string config_path = (argc > 1) ? argv[1] : "configs/sift/sift.json";
  auto config_res = LoadConfigFromJson(config_path);
  if (!config_res.ok()) {
    std::cerr << config_res.status().ToString() << std::endl;
    return 1;
  }
  Config config = config_res.value();
  std::cout << "Loaded " << config.ToString() << std::endl;

  std::optional<std::string> dataset_spec =
      (argc > 2) ? std::optional<std::string>(argv[2]) : std::nullopt;
  std::string dataset_label = "synthetic";
  std::optional<std::string> base_dataset_path;
  if (dataset_spec) {
    auto resolved = ResolveFvecsPath(*dataset_spec, "_base.fvecs");
    if (!resolved.ok()) {
      std::cerr << resolved.status().ToString() << std::endl;
      return 1;
    }
    base_dataset_path = resolved.value();
    dataset_label = DatasetLabelFromPath(*base_dataset_path);
    std::cout << "[INFO] Using base dataset: " << *base_dataset_path << std::endl;
  }

  MatrixRM X;
  uint32_t nx = 0;
  if (base_dataset_path) {
    auto load_res = LoadFvecs(*base_dataset_path);
    if (!load_res.ok()) {
      std::cerr << load_res.status().ToString() << std::endl;
      return 1;
    }
    X = load_res.value();
    nx = static_cast<uint32_t>(X.rows());
    if (nx == 0) {
      std::cerr << "Base dataset contains no vectors." << std::endl;
      return 1;
    }
    if (config.dim != static_cast<uint32_t>(X.cols())) {
      std::cout << "[INFO] Overriding config dim " << config.dim << " -> " << X.cols()
                << std::endl;
      config.dim = static_cast<uint32_t>(X.cols());
    }
  } else {
    nx = 64;
    X = GenerateRandom(nx, config.dim, config.seed);
    std::cout << "[WARN] No dataset supplied; building synthetic index." << std::endl;
  }

  const uint32_t main_rows = ResolveMainRows(config, nx);
  if (main_rows == 0 || main_rows > nx) {
    std::cerr << "Invalid main_rows resolved from config." << std::endl;
    return 1;
  }
  const std::string output_dir =
      (argc > 3) ? argv[3] : DefaultPrebuiltIndexDir(config_path, dataset_label);

  IVFParams ivf_params;
  ivf_params.nlist = std::max(1u, config.ivf_nlist);
  ivf_params.dim = config.dim;
  ivf_params.pq.enable = config.pq_enable;
  ivf_params.pq.M = config.pq_m;
  ivf_params.pq.nbits = config.pq_nbits;
  ivf_params.pq.residual = config.pq_residual;
  ivf_params.defer_pq_stats_to_add = true;

  Timer total_timer;
  auto whitening = CreateWhiteningModel();
  VersionId whiten_version = 0;
  double whitening_ms = 0.0;
  if (config.use_whitening) {
    Timer fit_timer;
    auto whiten_version_res = whitening->Fit(X.topRows(main_rows));
    if (!whiten_version_res.ok()) {
      std::cerr << whiten_version_res.status().ToString() << std::endl;
      return 1;
    }
    whiten_version = whiten_version_res.value();
    whitening_ms = fit_timer.ElapsedMillis();
  }

  MatrixRM X_index_space;
  double whitening_transform_ms = 0.0;
  if (config.use_whitening) {
    Timer transform_timer;
    auto xw_res = whitening->TransformBatch(X.topRows(main_rows), whiten_version);
    if (!xw_res.ok()) {
      std::cerr << xw_res.status().ToString() << std::endl;
      return 1;
    }
    X_index_space = std::move(xw_res.value());
    whitening_transform_ms = transform_timer.ElapsedMillis();
  } else {
    X_index_space = X.topRows(main_rows);
  }
  if (config.use_cosine) {
    NormalizeRowsL2(&X_index_space);
  }

  std::vector<DocId> main_ids(static_cast<size_t>(main_rows));
  std::iota(main_ids.begin(), main_ids.end(), 0);
  auto main_ivf = CreateIVFIndex();
  Timer build_timer;
  auto main_version_res = main_ivf->Build(X_index_space.topRows(main_rows),
                                          main_ids,
                                          ivf_params,
                                          0);
  if (!main_version_res.ok()) {
    std::cerr << main_version_res.status().ToString() << std::endl;
    return 1;
  }
  const VersionId main_index_version = main_version_res.value();
  const double main_build_ms = build_timer.ElapsedMillis();

  Timer add_timer;
  const Status add_status = AddRangeToIndex(main_ivf,
                                            X_index_space,
                                            0,
                                            main_rows,
                                            config.dim,
                                            VersionSet{whiten_version, main_index_version});
  if (!add_status.ok()) {
    std::cerr << add_status.ToString() << std::endl;
    return 1;
  }
  const double main_add_ms = add_timer.ElapsedMillis();

  std::vector<uint8_t> whitening_bytes;
  if (config.use_whitening) {
    auto whitening_bytes_res = whitening->Serialize();
    if (!whitening_bytes_res.ok()) {
      std::cerr << whitening_bytes_res.status().ToString() << std::endl;
      return 1;
    }
    whitening_bytes = std::move(whitening_bytes_res.value());
  }
  auto index_bytes_res = main_ivf->Serialize();
  if (!index_bytes_res.ok()) {
    std::cerr << index_bytes_res.status().ToString() << std::endl;
    return 1;
  }

  PrebuiltMainIndexArtifact artifact;
  artifact.metadata.dim = config.dim;
  artifact.metadata.main_rows = main_rows;
  artifact.metadata.total_rows = nx;
  artifact.metadata.whiten_version = whiten_version;
  artifact.metadata.index_version = main_index_version;
  artifact.metadata.use_whitening = config.use_whitening;
  artifact.metadata.config_path = config_path;
  artifact.metadata.dataset_path = base_dataset_path.value_or("synthetic");
  artifact.whitening_bytes = std::move(whitening_bytes);
  artifact.index_bytes = std::move(index_bytes_res.value());

  Timer save_timer;
  auto save_res = SavePrebuiltMainIndex(output_dir, artifact);
  if (!save_res.ok()) {
    std::cerr << save_res.status().ToString() << std::endl;
    return 1;
  }
  const double save_ms = save_timer.ElapsedMillis();
  const double total_ms = total_timer.ElapsedMillis();

  std::cout << "[BUILD_INDEX] output_dir=" << output_dir
            << ", total_rows=" << nx
            << ", main_rows=" << main_rows
            << ", whiten_version=" << whiten_version
            << ", index_version=" << main_index_version
            << ", whitening_ms=" << whitening_ms
            << ", whitening_transform_ms=" << whitening_transform_ms
            << ", main_build_ms=" << main_build_ms
            << ", main_add_ms=" << main_add_ms
            << ", save_ms=" << save_ms
            << ", total_ms=" << total_ms << std::endl;
  return 0;
}
