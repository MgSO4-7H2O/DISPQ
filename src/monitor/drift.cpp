#include "monitor/drift.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>

namespace ann {
namespace {

constexpr double kEps = 1e-6;
constexpr uint32_t kDefaultSeed = 42;
constexpr int kKMeansIterations = 15;

uint32_t ClampNBits(uint32_t nbits) { return std::max<uint32_t>(1, std::min<uint32_t>(8, nbits)); }

uint32_t ResolvePQM(uint32_t dim, uint32_t pq_m) {
  if (dim == 0) {
    return 1;
  }
  uint32_t m = pq_m;
  if (m == 0) {
    m = std::max<uint32_t>(1, std::min<uint32_t>(16, dim));
  }
  m = std::min<uint32_t>(m, dim);
  while (m > 1 && dim % m != 0) {
    --m;
  }
  return std::max<uint32_t>(1, m);
}

MatrixRM InitializeCentroids(Eigen::Ref<const MatrixRM> X, uint32_t nlist, uint32_t seed) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  MatrixRM centroids(nlist, dim);
  std::vector<int64_t> indices(static_cast<size_t>(num_vecs));
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 gen(seed);
  std::shuffle(indices.begin(), indices.end(), gen);
  for (uint32_t i = 0; i < nlist; ++i) {
    centroids.row(i) = X.row(indices[static_cast<size_t>(i % num_vecs)]);
  }
  return centroids;
}

void RunKMeans(Eigen::Ref<const MatrixRM> X, MatrixRM* centroids) {
  const int64_t num_vecs = X.rows();
  const int64_t dim = X.cols();
  const int64_t k = centroids->rows();
  std::vector<int> assignments(static_cast<size_t>(num_vecs), 0);

  for (int iter = 0; iter < kKMeansIterations; ++iter) {
    for (int64_t i = 0; i < num_vecs; ++i) {
      float best = std::numeric_limits<float>::max();
      int best_idx = 0;
      Eigen::VectorXf vec = X.row(i).transpose();
      for (int64_t c = 0; c < k; ++c) {
        const float dist = (centroids->row(c).transpose() - vec).squaredNorm();
        if (dist < best) {
          best = dist;
          best_idx = static_cast<int>(c);
        }
      }
      assignments[static_cast<size_t>(i)] = best_idx;
    }

    MatrixRM new_centroids = MatrixRM::Zero(k, dim);
    std::vector<int64_t> counts(static_cast<size_t>(k), 0);
    for (int64_t i = 0; i < num_vecs; ++i) {
      const int assign = assignments[static_cast<size_t>(i)];
      new_centroids.row(assign) += X.row(i);
      counts[static_cast<size_t>(assign)]++;
    }
    std::mt19937 gen(kDefaultSeed + static_cast<uint32_t>(iter));
    std::uniform_int_distribution<int64_t> dist_index(0, std::max<int64_t>(0, num_vecs - 1));
    for (int64_t c = 0; c < k; ++c) {
      const int64_t cnt = counts[static_cast<size_t>(c)];
      if (cnt > 0) {
        new_centroids.row(c) /= static_cast<float>(cnt);
      } else {
        new_centroids.row(c) = X.row(dist_index(gen));
      }
    }
    *centroids = std::move(new_centroids);
  }
}

double Log2Safe(double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  return std::log(x) / std::log(2.0);
}

double NormalizedEntropy(const std::vector<uint64_t>& counts, size_t symbols) {
  const uint64_t total = std::accumulate(counts.begin(), counts.end(), uint64_t{0});
  if (total == 0 || symbols <= 1) {
    return 0.0;
  }
  double ent = 0.0;
  for (uint64_t c : counts) {
    if (c == 0) {
      continue;
    }
    const double p = static_cast<double>(c) / static_cast<double>(total);
    ent -= p * Log2Safe(p);
  }
  return ent / std::max(1e-12, Log2Safe(static_cast<double>(symbols)));
}

double JSDivergence(const std::vector<double>& p, const std::vector<double>& q) {
  const size_t n = std::min(p.size(), q.size());
  if (n == 0) {
    return 0.0;
  }
  double js = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double pi = std::max(0.0, p[i]);
    const double qi = std::max(0.0, q[i]);
    const double m = 0.5 * (pi + qi);
    if (pi > 0.0) {
      js += 0.5 * pi * Log2Safe(pi / std::max(kEps, m));
    }
    if (qi > 0.0) {
      js += 0.5 * qi * Log2Safe(qi / std::max(kEps, m));
    }
  }
  return std::max(0.0, js);
}

struct CoarseStats {
  DriftMetrics metrics{};
  MatrixRM residuals;
  std::vector<double> list_dist;
};

Result<CoarseStats> ComputeCoarseStats(Eigen::Ref<const MatrixRM> X,
                                       Eigen::Ref<const MatrixRM> coarse_centroids) {
  if (X.rows() == 0 || X.cols() == 0) {
    return Status::InvalidArgument("ComputeCoarseStats: empty vectors");
  }
  if (coarse_centroids.rows() == 0 || coarse_centroids.cols() != X.cols()) {
    return Status::InvalidArgument("ComputeCoarseStats: invalid centroids");
  }
  const uint32_t nlist = static_cast<uint32_t>(coarse_centroids.rows());
  CoarseStats out;
  out.residuals = MatrixRM(X.rows(), X.cols());
  out.list_dist.assign(nlist, 0.0);
  std::vector<double> cm_values(static_cast<size_t>(X.rows()), 0.0);

  double nre_sum = 0.0;
  double cm_sum = 0.0;

  for (int64_t i = 0; i < X.rows(); ++i) {
    Eigen::VectorXf z = X.row(i).transpose();
    float d1 = std::numeric_limits<float>::max();
    float d2 = std::numeric_limits<float>::max();
    int best_idx = 0;
    for (int c = 0; c < coarse_centroids.rows(); ++c) {
      const float d = (z - coarse_centroids.row(c).transpose()).squaredNorm();
      if (d < d1) {
        d2 = d1;
        d1 = d;
        best_idx = c;
      } else if (d < d2) {
        d2 = d;
      }
    }
    Eigen::VectorXf r = z - coarse_centroids.row(best_idx).transpose();
    out.residuals.row(i) = r.transpose();
    out.list_dist[static_cast<size_t>(best_idx)] += 1.0;

    const double znorm = static_cast<double>(z.squaredNorm());
    const double rnorm = static_cast<double>(r.squaredNorm());
    const double nre = rnorm / (znorm + kEps);
    const double cm =
        static_cast<double>(d2 - d1) / (std::max(kEps, static_cast<double>(d1)));
    nre_sum += nre;
    cm_sum += cm;
    cm_values[static_cast<size_t>(i)] = cm;
  }

  const double inv_rows = 1.0 / static_cast<double>(X.rows());
  out.metrics.nre = nre_sum * inv_rows;
  out.metrics.cm = cm_sum * inv_rows;

  double cm_var = 0.0;
  for (double v : cm_values) {
    const double d = v - out.metrics.cm;
    cm_var += d * d;
  }
  cm_var *= inv_rows;
  out.metrics.cm_z = std::sqrt(std::max(0.0, cm_var));

  for (double& p : out.list_dist) {
    p *= inv_rows;
  }
  return out;
}

}  // namespace

DriftMonitor::DriftMonitor() = default;
DriftMonitor::~DriftMonitor() = default;

Status DriftMonitor::Initialize(Eigen::Ref<const MatrixRM> baseline_vectors,
                                const DriftParams& params) {
  if (baseline_vectors.rows() == 0 || baseline_vectors.cols() == 0) {
    return Status::InvalidArgument("DriftMonitor::Initialize: baseline vectors are empty");
  }
  if (params.confirm_k == 0 || params.confirm_m == 0 || params.confirm_k > params.confirm_m) {
    return Status::InvalidArgument("DriftMonitor::Initialize: invalid confirmation window params");
  }
  if (params.confirm_ratio <= 0.0 || params.confirm_ratio > 1.0) {
    return Status::InvalidArgument("DriftMonitor::Initialize: confirm_ratio must be in (0,1]");
  }

  std::unique_lock lock(mu_);
  params_ = params;
  params_.nlist = std::max<uint32_t>(1, params_.nlist);
  params_.pq_nbits = ClampNBits(params_.pq_nbits);
  soft_hit_history_.clear();
  hard_hit_history_.clear();
  windows_since_delta_create_ = 0;
  has_candidate_pq_ = false;
  request_whiten_ = false;
  request_tail_ = false;

  Status fit = FitBaselineLocked(baseline_vectors);
  if (!fit.ok()) {
    return fit;
  }
  initialized_ = true;
  return Status::OK();
}

Status DriftMonitor::FitBaselineLocked(Eigen::Ref<const MatrixRM> baseline_vectors) {
  const uint32_t target_nlist = std::max<uint32_t>(
      2, static_cast<uint32_t>(std::sqrt(static_cast<double>(baseline_vectors.rows())) * 4.0));
  const uint32_t nlist = std::min<uint32_t>(
      std::min<uint32_t>(params_.nlist, target_nlist),
      static_cast<uint32_t>(baseline_vectors.rows()));
  coarse_centroids_ = InitializeCentroids(baseline_vectors, nlist, kDefaultSeed);
  RunKMeans(baseline_vectors, &coarse_centroids_);

  auto coarse_res = ComputeCoarseStats(baseline_vectors, coarse_centroids_);
  if (!coarse_res.ok()) {
    return coarse_res.status();
  }
  CoarseStats coarse = coarse_res.value();
  baseline_list_dist_ = coarse.list_dist;

  auto pq_res = TrainPQModelLocked(coarse.residuals, params_.pq_m);
  if (!pq_res.ok()) {
    return pq_res.status();
  }
  active_pq_ = pq_res.value();
  has_candidate_pq_ = false;
  candidate_pq_ = PQModel{};

  double cue = 0.0;
  auto npd_res = ComputeNPDLocked(coarse.residuals, active_pq_, &cue);
  if (!npd_res.ok()) {
    return npd_res.status();
  }

  baseline_metrics_ = DriftMetrics{};
  baseline_metrics_.nre = coarse.metrics.nre;
  baseline_metrics_.npd = npd_res.value();
  baseline_metrics_.cm = coarse.metrics.cm;
  baseline_metrics_.lds = 0.0;
  baseline_metrics_.cue = cue;
  baseline_metrics_.nre_ratio = 1.0;
  baseline_metrics_.npd_ratio = 1.0;
  baseline_metrics_.cm_z = 0.0;
  baseline_metrics_.cue_ratio = 1.0;

  baseline_cm_std_ = std::max(coarse.metrics.cm_z, 1e-6);
  return Status::OK();
}

Result<DriftMonitor::PQModel> DriftMonitor::TrainPQModelLocked(Eigen::Ref<const MatrixRM> residuals,
                                                               uint32_t pq_m) const {
  if (residuals.rows() == 0 || residuals.cols() == 0) {
    return Status::InvalidArgument("TrainPQModelLocked: empty residuals");
  }
  const uint32_t dim = static_cast<uint32_t>(residuals.cols());
  const uint32_t m = ResolvePQM(dim, pq_m);
  const uint32_t dsub = dim / m;
  const uint32_t ks = 1u << ClampNBits(params_.pq_nbits);

  PQModel pq;
  pq.M = m;
  pq.nbits = ClampNBits(params_.pq_nbits);
  pq.dsub = dsub;
  pq.codebooks.resize(m);

  for (uint32_t i = 0; i < m; ++i) {
    const Eigen::Index offset = static_cast<Eigen::Index>(i * dsub);
    MatrixRM sub = residuals.block(0, offset, residuals.rows(), static_cast<Eigen::Index>(dsub));
    const uint32_t local_k = std::min<uint32_t>(ks, static_cast<uint32_t>(sub.rows()));
    MatrixRM codebook = InitializeCentroids(sub, local_k, kDefaultSeed + i + 17);
    RunKMeans(sub, &codebook);
    pq.codebooks[static_cast<size_t>(i)] = std::move(codebook);
  }
  return pq;
}

Result<double> DriftMonitor::ComputeNPDLocked(Eigen::Ref<const MatrixRM> residuals,
                                              const PQModel& pq,
                                              double* cue_out) const {
  if (!pq.valid()) {
    return Status::InvalidArgument("ComputeNPDLocked: invalid PQ model");
  }
  if (residuals.rows() == 0 || residuals.cols() == 0) {
    return Status::InvalidArgument("ComputeNPDLocked: empty residuals");
  }
  if (static_cast<uint32_t>(residuals.cols()) != pq.M * pq.dsub) {
    return Status::InvalidArgument("ComputeNPDLocked: dim mismatch");
  }

  std::vector<std::vector<uint64_t>> usage(pq.M);
  for (uint32_t m = 0; m < pq.M; ++m) {
    usage[static_cast<size_t>(m)].assign(
        static_cast<size_t>(pq.codebooks[static_cast<size_t>(m)].rows()), 0);
  }

  double npd_sum = 0.0;
  for (int64_t i = 0; i < residuals.rows(); ++i) {
    Eigen::VectorXf r = residuals.row(i).transpose();
    const double rnorm = static_cast<double>(r.squaredNorm());
    double err = 0.0;
    for (uint32_t m = 0; m < pq.M; ++m) {
      const MatrixRM& codebook = pq.codebooks[static_cast<size_t>(m)];
      Eigen::Map<const Eigen::VectorXf> sub(
          r.data() + static_cast<Eigen::Index>(m * pq.dsub),
          static_cast<Eigen::Index>(pq.dsub));
      double best = std::numeric_limits<double>::max();
      int best_idx = 0;
      for (int k = 0; k < codebook.rows(); ++k) {
        const double dist = static_cast<double>((sub - codebook.row(k).transpose()).squaredNorm());
        if (dist < best) {
          best = dist;
          best_idx = k;
        }
      }
      err += best;
      usage[static_cast<size_t>(m)][static_cast<size_t>(best_idx)]++;
    }
    npd_sum += err / (rnorm + kEps);
  }

  if (cue_out) {
    double cue = 0.0;
    for (uint32_t m = 0; m < pq.M; ++m) {
      cue += NormalizedEntropy(usage[static_cast<size_t>(m)],
                               usage[static_cast<size_t>(m)].size());
    }
    *cue_out = cue / static_cast<double>(pq.M);
  }

  return npd_sum / static_cast<double>(residuals.rows());
}

Result<DriftMonitor::WindowEval> DriftMonitor::EvaluateWindowLocked(
    Eigen::Ref<const MatrixRM> window_vectors) const {
  auto coarse_res = ComputeCoarseStats(window_vectors, coarse_centroids_);
  if (!coarse_res.ok()) {
    return coarse_res.status();
  }
  CoarseStats coarse = coarse_res.value();

  double cue = 0.0;
  auto npd_res = ComputeNPDLocked(coarse.residuals, active_pq_, &cue);
  if (!npd_res.ok()) {
    return npd_res.status();
  }

  WindowEval out;
  out.metrics.nre = coarse.metrics.nre;
  out.metrics.npd = npd_res.value();
  out.metrics.cm = coarse.metrics.cm;
  out.metrics.lds = JSDivergence(coarse.list_dist, baseline_list_dist_);
  out.metrics.cue = cue;

  const double nre_base = std::max(1e-3, baseline_metrics_.nre);
  const double npd_base = std::max(1e-3, baseline_metrics_.npd);
  const double cue_base = std::max(1e-3, baseline_metrics_.cue);
  out.metrics.nre_ratio = out.metrics.nre / nre_base;
  out.metrics.npd_ratio = out.metrics.npd / npd_base;
  out.metrics.cm_z = (out.metrics.cm - baseline_metrics_.cm) / std::max(1e-6, baseline_cm_std_);
  out.metrics.cue_ratio = out.metrics.cue / cue_base;
  out.residuals = std::move(coarse.residuals);
  return out;
}

bool DriftMonitor::TriggerByHistoryLocked(const std::deque<bool>& history) const {
  if (history.empty()) {
    return false;
  }
  const uint32_t k = std::max<uint32_t>(1, params_.confirm_k);
  const uint32_t m = std::max<uint32_t>(k, params_.confirm_m);

  if (history.size() >= static_cast<size_t>(k)) {
    bool consecutive = true;
    for (size_t i = history.size() - k; i < history.size(); ++i) {
      if (!history[i]) {
        consecutive = false;
        break;
      }
    }
    if (consecutive) {
      return true;
    }
  }

  if (history.size() >= static_cast<size_t>(m)) {
    uint32_t hits = 0;
    for (size_t i = history.size() - m; i < history.size(); ++i) {
      hits += history[i] ? 1u : 0u;
    }
    const uint32_t required = static_cast<uint32_t>(
        std::ceil(static_cast<double>(m) * params_.confirm_ratio - 1e-9));
    if (hits >= required) {
      return true;
    }
  }

  return false;
}

Result<double> DriftMonitor::EstimateCandidateGainLocked(Eigen::Ref<const MatrixRM> residuals,
                                                         double npd_old,
                                                         double* npd_new_out) {
  if (residuals.rows() == 0 || npd_old <= 0.0) {
    if (npd_new_out) {
      *npd_new_out = npd_old;
    }
    return 0.0;
  }
  const uint32_t pq_m = active_pq_.valid() ? active_pq_.M : params_.pq_m;
  auto candidate_res = TrainPQModelLocked(residuals, pq_m);
  if (!candidate_res.ok()) {
    if (npd_new_out) {
      *npd_new_out = npd_old;
    }
    return 0.0;
  }
  double candidate_cue = 0.0;
  auto npd_res = ComputeNPDLocked(residuals, candidate_res.value(), &candidate_cue);
  if (!npd_res.ok()) {
    if (npd_new_out) {
      *npd_new_out = npd_old;
    }
    return 0.0;
  }

  const double npd_new = npd_res.value();
  if (npd_new_out) {
    *npd_new_out = npd_new;
  }
  has_candidate_pq_ = true;
  candidate_pq_ = candidate_res.value();
  return (npd_old - npd_new) / (npd_old + kEps);
}

Result<DriftDecision> DriftMonitor::ObserveWindow(Eigen::Ref<const MatrixRM> window_vectors,
                                                  const DriftContext& context) {
  if (window_vectors.rows() == 0 || window_vectors.cols() == 0) {
    return Status::InvalidArgument("ObserveWindow: empty vectors");
  }

  std::unique_lock lock(mu_);
  if (!initialized_) {
    return Status::Unavailable("ObserveWindow: monitor not initialized");
  }

  auto eval_res = EvaluateWindowLocked(window_vectors);
  if (!eval_res.ok()) {
    return eval_res.status();
  }
  WindowEval eval = eval_res.value();

  const double closed_ratio = context.total_docs > 0
                                  ? static_cast<double>(context.closed_delta_docs) /
                                        static_cast<double>(context.total_docs)
                                  : 0.0;

  bool soft_hit = eval.metrics.npd_ratio > params_.soft_npd_ratio &&
                  eval.metrics.nre_ratio < params_.hard_nre_ratio;
  bool hard_hit = eval.metrics.nre_ratio > params_.hard_nre_ratio ||
                  eval.metrics.cm_z > params_.hard_cm_z ||
                  eval.metrics.lds > params_.hard_lds ||
                  context.closed_delta_count >= params_.max_closed_deltas ||
                  closed_ratio > params_.max_closed_ratio;

  soft_hit_history_.push_back(soft_hit);
  hard_hit_history_.push_back(hard_hit);
  while (soft_hit_history_.size() > static_cast<size_t>(params_.confirm_m)) {
    soft_hit_history_.pop_front();
  }
  while (hard_hit_history_.size() > static_cast<size_t>(params_.confirm_m)) {
    hard_hit_history_.pop_front();
  }

  DriftDecision decision;
  decision.metrics = eval.metrics;
  decision.soft_hit = soft_hit;
  decision.hard_hit = hard_hit;

  const bool soft_confirmed = TriggerByHistoryLocked(soft_hit_history_);
  const bool hard_confirmed = TriggerByHistoryLocked(hard_hit_history_);

  if (soft_confirmed &&
      windows_since_delta_create_ >= params_.min_delta_lifetime_windows) {
    double npd_new = eval.metrics.npd;
    auto gain_res = EstimateCandidateGainLocked(eval.residuals, eval.metrics.npd, &npd_new);
    if (gain_res.ok()) {
      decision.candidate_gain = gain_res.value();
      decision.candidate_npd = npd_new;
      if (decision.candidate_gain >= params_.soft_gain) {
        decision.trigger_new_delta = true;
        decision.trigger_seal_delta = true;
      }
    }
  }

  if (hard_confirmed) {
    decision.trigger_rebuild_main = true;
  }

  request_whiten_ = decision.trigger_rebuild_main;
  request_tail_ = decision.trigger_new_delta;
  windows_since_delta_create_++;
  return decision;
}

Status DriftMonitor::OnNewDeltaPromoted(bool promote_candidate) {
  std::unique_lock lock(mu_);
  if (!initialized_) {
    return Status::Unavailable("OnNewDeltaPromoted: monitor not initialized");
  }
  if (promote_candidate && has_candidate_pq_ && candidate_pq_.valid()) {
    active_pq_ = candidate_pq_;
  }
  has_candidate_pq_ = false;
  candidate_pq_ = PQModel{};
  soft_hit_history_.clear();
  hard_hit_history_.clear();
  windows_since_delta_create_ = 0;
  request_tail_ = false;
  return Status::OK();
}

Status DriftMonitor::OnMainRebuilt(Eigen::Ref<const MatrixRM> baseline_vectors) {
  if (baseline_vectors.rows() == 0 || baseline_vectors.cols() == 0) {
    return Status::InvalidArgument("OnMainRebuilt: baseline vectors are empty");
  }
  std::unique_lock lock(mu_);
  if (!initialized_) {
    return Status::Unavailable("OnMainRebuilt: monitor not initialized");
  }
  Status fit = FitBaselineLocked(baseline_vectors);
  if (!fit.ok()) {
    return fit;
  }
  soft_hit_history_.clear();
  hard_hit_history_.clear();
  windows_since_delta_create_ = 0;
  request_whiten_ = false;
  request_tail_ = false;
  return Status::OK();
}

Result<DriftMetrics> DriftMonitor::BaselineMetrics() const {
  std::shared_lock lock(mu_);
  if (!initialized_) {
    return Status::Unavailable("BaselineMetrics: monitor not initialized");
  }
  return baseline_metrics_;
}

Status DriftMonitor::ObserveResidual(Eigen::Ref<const MatrixRM> residuals) {
  if (residuals.size() == 0) {
    return Status::InvalidArgument("Residuals empty");
  }
  std::unique_lock lock(mu_);
  double energy = 0.0;
  for (int i = 0; i < residuals.rows(); ++i) {
    energy += residuals.row(i).squaredNorm();
  }
  request_whiten_ = energy > 1e4;
  return Status::OK();
}

Status DriftMonitor::ObserveSearchSignal(Eigen::Ref<const Eigen::VectorXf> margins) {
  if (margins.size() == 0) {
    return Status::InvalidArgument("Margins empty");
  }
  std::unique_lock lock(mu_);
  request_tail_ = margins.mean() < 0.1;
  return Status::OK();
}

Result<bool> DriftMonitor::ShouldUpdateWhiten() const {
  std::shared_lock lock(mu_);
  return request_whiten_;
}

Result<bool> DriftMonitor::ShouldUpdateTail() const {
  std::shared_lock lock(mu_);
  return request_tail_;
}

}  // namespace ann
