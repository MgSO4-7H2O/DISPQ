#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/status.h"
#include "common/types.h"

namespace ann {

struct FinalStateCacheMetadata {
  uint32_t dim{0};
  uint32_t topk{0};
  uint32_t seen_rows{0};
  uint32_t live_rows{0};
  uint32_t main_rows{0};
  uint32_t active_rows{0};
  uint32_t active_shard_id{0};
  uint32_t frozen_rows{0};
  uint32_t frozen_shard_id{0};
  uint32_t last_insert_begin{0};
  uint32_t last_insert_end{0};
  uint32_t has_active{0};
  uint32_t has_frozen{0};
  uint32_t use_whitening{1};
  VersionId whiten_version{0};
  VersionId main_index_version{0};
  VersionId active_whiten_version{0};
  VersionId active_index_version{0};
  VersionId frozen_whiten_version{0};
  VersionId frozen_index_version{0};
  std::string base_identity;
  std::string query_identity;
};

struct FinalStateCache {
  FinalStateCacheMetadata metadata;
  std::vector<uint8_t> whitening;
  std::vector<uint8_t> main_index;
  std::vector<uint8_t> active_index;
  std::vector<uint8_t> frozen_index;
  std::vector<std::vector<DocId>> ground_truth;
  std::vector<uint8_t> deleted;
  MatrixRM whitened_vectors;
  std::vector<DocId> whitened_doc_ids;
};

namespace final_state_cache_internal {

template <typename T>
inline bool WriteValue(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  return static_cast<bool>(out);
}

template <typename T>
inline bool ReadValue(std::ifstream& in, T* value) {
  in.read(reinterpret_cast<char*>(value), sizeof(T));
  return static_cast<bool>(in);
}

inline bool WriteString(std::ofstream& out, const std::string& value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  if (!WriteValue(out, size)) return false;
  if (size) out.write(value.data(), size);
  return static_cast<bool>(out);
}

inline bool ReadString(std::ifstream& in, std::string* value) {
  uint32_t size = 0;
  if (!ReadValue(in, &size) || size > (1u << 20)) return false;
  value->resize(size);
  if (size) in.read(value->data(), size);
  return static_cast<bool>(in);
}

inline Status WriteBlob(const std::filesystem::path& path,
                        const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return Status::IOError("cannot write cache file: " + path.string());
  if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  return out ? Status::OK() : Status::IOError("failed writing cache file: " + path.string());
}

inline Result<std::vector<uint8_t>> ReadBlob(const std::filesystem::path& path,
                                             bool optional = false) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    if (optional && !std::filesystem::exists(path)) return std::vector<uint8_t>{};
    return Status::IOError("cannot read cache file: " + path.string());
  }
  const auto end = in.tellg();
  if (end < 0) return Status::IOError("cannot size cache file: " + path.string());
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  in.seekg(0);
  if (!bytes.empty()) in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!in) return Status::IOError("failed reading cache file: " + path.string());
  return bytes;
}

inline Status WriteMetadata(const std::filesystem::path& path,
                            const FinalStateCacheMetadata& m) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return Status::IOError("cannot write cache metadata: " + path.string());
  constexpr std::array<char, 8> magic{'D','S','P','Q','F','S','C','1'};
  out.write(magic.data(), magic.size());
  const uint32_t version = 3;
  if (!out || !WriteValue(out, version) ||
      !WriteValue(out, m.dim) || !WriteValue(out, m.topk) ||
      !WriteValue(out, m.seen_rows) || !WriteValue(out, m.live_rows) ||
      !WriteValue(out, m.main_rows) || !WriteValue(out, m.active_rows) ||
      !WriteValue(out, m.active_shard_id) || !WriteValue(out, m.frozen_rows) ||
      !WriteValue(out, m.frozen_shard_id) || !WriteValue(out, m.last_insert_begin) ||
      !WriteValue(out, m.last_insert_end) || !WriteValue(out, m.has_active) ||
      !WriteValue(out, m.has_frozen) || !WriteValue(out, m.use_whitening) ||
      !WriteValue(out, m.whiten_version) ||
      !WriteValue(out, m.main_index_version) || !WriteValue(out, m.active_whiten_version) ||
      !WriteValue(out, m.active_index_version) || !WriteValue(out, m.frozen_whiten_version) ||
      !WriteValue(out, m.frozen_index_version) ||
      !WriteString(out, m.base_identity) || !WriteString(out, m.query_identity)) {
    return Status::IOError("failed writing cache metadata: " + path.string());
  }
  return Status::OK();
}

inline Result<FinalStateCacheMetadata> ReadMetadata(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Status::IOError("cannot read cache metadata: " + path.string());
  std::array<char, 8> magic{};
  in.read(magic.data(), magic.size());
  constexpr std::array<char, 8> expected{'D','S','P','Q','F','S','C','1'};
  uint32_t version = 0;
  FinalStateCacheMetadata m;
  if (!in || magic != expected || !ReadValue(in, &version) ||
      (version != 2 && version != 3) ||
      !ReadValue(in, &m.dim) || !ReadValue(in, &m.topk) ||
      !ReadValue(in, &m.seen_rows) || !ReadValue(in, &m.live_rows) ||
      !ReadValue(in, &m.main_rows) || !ReadValue(in, &m.active_rows) ||
      !ReadValue(in, &m.active_shard_id) || !ReadValue(in, &m.frozen_rows) ||
      !ReadValue(in, &m.frozen_shard_id) || !ReadValue(in, &m.last_insert_begin) ||
      !ReadValue(in, &m.last_insert_end) || !ReadValue(in, &m.has_active) ||
      !ReadValue(in, &m.has_frozen) ||
      (version >= 3 && !ReadValue(in, &m.use_whitening)) ||
      !ReadValue(in, &m.whiten_version) ||
      !ReadValue(in, &m.main_index_version) || !ReadValue(in, &m.active_whiten_version) ||
      !ReadValue(in, &m.active_index_version) || !ReadValue(in, &m.frozen_whiten_version) ||
      !ReadValue(in, &m.frozen_index_version) ||
      !ReadString(in, &m.base_identity) || !ReadString(in, &m.query_identity)) {
    return Status::InvalidArgument("invalid final-state cache metadata: " + path.string());
  }
  return m;
}

inline Status WriteGroundTruth(const std::filesystem::path& path,
                               const std::vector<std::vector<DocId>>& gt) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return Status::IOError("cannot write cache GT: " + path.string());
  const uint64_t rows = gt.size();
  if (!WriteValue(out, rows)) return Status::IOError("failed writing cache GT header");
  for (const auto& row : gt) {
    const uint32_t size = static_cast<uint32_t>(row.size());
    if (!WriteValue(out, size)) return Status::IOError("failed writing cache GT row");
    if (size) out.write(reinterpret_cast<const char*>(row.data()), sizeof(DocId) * size);
  }
  return out ? Status::OK() : Status::IOError("failed writing cache GT: " + path.string());
}

inline Result<std::vector<std::vector<DocId>>> ReadGroundTruth(
    const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Status::IOError("cannot read cache GT: " + path.string());
  uint64_t rows = 0;
  if (!ReadValue(in, &rows) || rows > (1ull << 32))
    return Status::InvalidArgument("invalid cache GT header");
  std::vector<std::vector<DocId>> gt(static_cast<size_t>(rows));
  for (auto& row : gt) {
    uint32_t size = 0;
    if (!ReadValue(in, &size) || size > (1u << 20))
      return Status::InvalidArgument("invalid cache GT row");
    row.resize(size);
    if (size) in.read(reinterpret_cast<char*>(row.data()), sizeof(DocId) * size);
    if (!in) return Status::IOError("truncated cache GT: " + path.string());
  }
  return gt;
}

inline Status WriteWhitenedVectors(const std::filesystem::path& path,
                                   const MatrixRM& vectors,
                                   uint32_t rows) {
  if (rows > static_cast<uint32_t>(vectors.rows()))
    return Status::InvalidArgument("cache vector row count exceeds matrix");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return Status::IOError("cannot write cache vectors: " + path.string());
  const uint32_t dim = static_cast<uint32_t>(vectors.cols());
  if (!WriteValue(out, rows) || !WriteValue(out, dim))
    return Status::IOError("failed writing cache vector header");
  for (uint32_t i = 0; i < rows; ++i) {
    out.write(reinterpret_cast<const char*>(vectors.row(i).data()), sizeof(float) * dim);
    if (!out) return Status::IOError("failed writing cache vectors: " + path.string());
  }
  return Status::OK();
}

inline Result<MatrixRM> ReadWhitenedVectors(const std::filesystem::path& path,
                                            uint32_t expected_rows,
                                            uint32_t expected_dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Status::IOError("cannot read cache vectors: " + path.string());
  uint32_t rows = 0;
  uint32_t dim = 0;
  if (!ReadValue(in, &rows) || !ReadValue(in, &dim) || rows != expected_rows ||
      dim != expected_dim)
    return Status::InvalidArgument("cache vector shape mismatch: " + path.string());
  MatrixRM vectors(rows, dim);
  if (rows) in.read(reinterpret_cast<char*>(vectors.data()), sizeof(float) *
                    static_cast<size_t>(rows) * dim);
  if (!in) return Status::IOError("truncated cache vectors: " + path.string());
  return vectors;
}

inline Status WriteSparseWhitenedVectors(const std::filesystem::path& path,
                                         const std::vector<DocId>& ids,
                                         const MatrixRM& vectors) {
  if (ids.size() != static_cast<size_t>(vectors.rows()))
    return Status::InvalidArgument("sparse cache vector ID/row count mismatch");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return Status::IOError("cannot write sparse cache vectors: " + path.string());
  const uint64_t count = ids.size();
  const uint32_t dim = static_cast<uint32_t>(vectors.cols());
  if (!WriteValue(out, count) || !WriteValue(out, dim))
    return Status::IOError("failed writing sparse cache vector header");
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!WriteValue(out, ids[i])) return Status::IOError("failed writing sparse cache IDs");
    out.write(reinterpret_cast<const char*>(vectors.row(static_cast<Eigen::Index>(i)).data()),
              sizeof(float) * dim);
    if (!out) return Status::IOError("failed writing sparse cache vectors");
  }
  return Status::OK();
}

inline Status ReadSparseWhitenedVectors(const std::filesystem::path& path,
                                       uint32_t expected_dim,
                                       std::vector<DocId>* ids,
                                       MatrixRM* vectors) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Status::IOError("cannot read sparse cache vectors: " + path.string());
  uint64_t count = 0;
  uint32_t dim = 0;
  if (!ReadValue(in, &count) || !ReadValue(in, &dim) || dim != expected_dim ||
      count > (1ull << 32))
    return Status::InvalidArgument("invalid sparse cache vector header");
  ids->resize(static_cast<size_t>(count));
  *vectors = MatrixRM(static_cast<Eigen::Index>(count), dim);
  for (size_t i = 0; i < ids->size(); ++i) {
    if (!ReadValue(in, &(*ids)[i])) return Status::IOError("truncated sparse cache IDs");
    in.read(reinterpret_cast<char*>(vectors->row(static_cast<Eigen::Index>(i)).data()),
            sizeof(float) * dim);
    if (!in) return Status::IOError("truncated sparse cache vectors");
  }
  return Status::OK();
}

}  // namespace final_state_cache_internal

inline Status ValidateFinalStateCacheContents(const FinalStateCache& cache) {
  const auto& meta = cache.metadata;
  if (meta.dim == 0 || meta.topk == 0 || meta.seen_rows == 0 || meta.live_rows == 0) {
    return Status::InvalidArgument("final-state cache metadata has zero dimensions or rows");
  }
  if (cache.whitening.empty() || cache.main_index.empty()) {
    return Status::InvalidArgument("final-state cache is missing whitening or main IVF payload");
  }
  if (meta.has_active > 1 || meta.has_frozen > 1) {
    return Status::InvalidArgument("final-state cache shard flags are invalid");
  }
  if ((meta.has_active != 0) != !cache.active_index.empty() ||
      (meta.has_frozen != 0) != !cache.frozen_index.empty()) {
    return Status::InvalidArgument("final-state cache shard metadata/payload mismatch");
  }
  if ((meta.has_active == 0 && meta.active_rows != 0) ||
      (meta.has_frozen == 0 && meta.frozen_rows != 0) ||
      (meta.has_active != 0 && meta.active_rows == 0) ||
      (meta.has_frozen != 0 && meta.frozen_rows == 0)) {
    return Status::InvalidArgument("final-state cache shard row metadata is inconsistent");
  }
  const uint64_t route_rows = static_cast<uint64_t>(meta.main_rows) +
      (meta.has_active ? meta.active_rows : 0u) +
      (meta.has_frozen ? meta.frozen_rows : 0u);
  if (route_rows != meta.live_rows) {
    return Status::InvalidArgument("final-state cache live row counts do not match its routes");
  }
  if (cache.ground_truth.empty()) {
    return Status::InvalidArgument("final-state cache ground truth is empty");
  }
  if (!cache.deleted.empty() && cache.deleted.size() != meta.seen_rows) {
    return Status::InvalidArgument("final-state cache deletion bitmap shape mismatch");
  }
  std::unordered_set<DocId> sparse_vector_ids;
  if (!cache.whitened_doc_ids.empty()) {
    sparse_vector_ids.reserve(cache.whitened_doc_ids.size());
    for (DocId doc_id : cache.whitened_doc_ids) {
      if (doc_id >= meta.seen_rows ||
          (!cache.deleted.empty() && cache.deleted[doc_id] != 0) ||
          !sparse_vector_ids.insert(doc_id).second) {
        return Status::InvalidArgument("final-state cache sparse vectors have invalid IDs");
      }
    }
  }
  const size_t expected_k = std::min<size_t>(meta.topk, meta.live_rows);
  for (const auto& row : cache.ground_truth) {
    if (row.size() != expected_k) {
      return Status::InvalidArgument("final-state cache ground-truth shape mismatch");
    }
    std::unordered_set<DocId> unique_ids;
    unique_ids.reserve(row.size());
    for (DocId doc_id : row) {
      if (doc_id >= meta.seen_rows ||
          (!cache.deleted.empty() && cache.deleted[doc_id] != 0) ||
          (!sparse_vector_ids.empty() && sparse_vector_ids.count(doc_id) == 0) ||
          !unique_ids.insert(doc_id).second) {
        return Status::InvalidArgument("final-state cache ground truth has invalid IDs");
      }
    }
  }
  if (!cache.whitened_doc_ids.empty() &&
      (cache.whitened_doc_ids.size() != static_cast<size_t>(cache.whitened_vectors.rows()) ||
       cache.whitened_vectors.cols() != static_cast<Eigen::Index>(meta.dim) ||
       cache.whitened_doc_ids.size() != meta.live_rows)) {
    return Status::InvalidArgument("final-state cache sparse vectors shape mismatch");
  }
  if (cache.whitened_doc_ids.empty() && cache.whitened_vectors.rows() > 0 &&
      (cache.whitened_vectors.cols() != static_cast<Eigen::Index>(meta.dim) ||
       cache.whitened_vectors.rows() != static_cast<Eigen::Index>(meta.seen_rows))) {
    return Status::InvalidArgument("final-state cache dense vectors shape mismatch");
  }
  return Status::OK();
}

inline Status SaveFinalStateCache(const std::string& dir, const FinalStateCache& cache) {
  if (dir.empty()) return Status::InvalidArgument("final-state cache path is empty");
  Status validation = ValidateFinalStateCacheContents(cache);
  if (!validation.ok()) return validation;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) return Status::IOError("cannot create final-state cache directory: " + ec.message());
  const std::filesystem::path root(dir);
  Status s = final_state_cache_internal::WriteMetadata(root / "metadata.bin", cache.metadata);
  if (!s.ok()) return s;
  if (!(s = final_state_cache_internal::WriteBlob(root / "whitening.bin", cache.whitening)).ok()) return s;
  if (!(s = final_state_cache_internal::WriteBlob(root / "main.ivf", cache.main_index)).ok()) return s;
  if (cache.metadata.has_active &&
      !(s = final_state_cache_internal::WriteBlob(root / "active.ivf", cache.active_index)).ok()) return s;
  if (!cache.metadata.has_active) {
    std::filesystem::remove(root / "active.ivf", ec);
    if (ec) return Status::IOError("cannot remove stale active cache shard: " + ec.message());
  }
  if (cache.metadata.has_frozen &&
      !(s = final_state_cache_internal::WriteBlob(root / "frozen.ivf", cache.frozen_index)).ok()) return s;
  if (!cache.metadata.has_frozen) {
    std::filesystem::remove(root / "frozen.ivf", ec);
    if (ec) return Status::IOError("cannot remove stale frozen cache shard: " + ec.message());
  }
  if (!(s = final_state_cache_internal::WriteGroundTruth(root / "ground_truth.bin", cache.ground_truth)).ok()) return s;
  if (!cache.deleted.empty()) {
    if (!(s = final_state_cache_internal::WriteBlob(root / "deleted.bin", cache.deleted)).ok()) return s;
  } else {
    std::filesystem::remove(root / "deleted.bin", ec);
    if (ec) return Status::IOError("cannot remove stale deletion bitmap: " + ec.message());
  }
  if (!cache.whitened_doc_ids.empty()) {
    s = final_state_cache_internal::WriteSparseWhitenedVectors(
        root / "live_vectors.bin", cache.whitened_doc_ids, cache.whitened_vectors);
    if (!s.ok()) return s;
    std::filesystem::remove(root / "whitened_vectors.bin", ec);
  } else if (cache.whitened_vectors.rows() > 0) {
    s = final_state_cache_internal::WriteWhitenedVectors(
        root / "whitened_vectors.bin", cache.whitened_vectors,
        static_cast<uint32_t>(cache.whitened_vectors.rows()));
    if (!s.ok()) return s;
    std::filesystem::remove(root / "live_vectors.bin", ec);
  } else {
    std::filesystem::remove(root / "live_vectors.bin", ec);
    std::filesystem::remove(root / "whitened_vectors.bin", ec);
  }
  return Status::OK();
}

inline Result<FinalStateCache> LoadFinalStateCache(const std::string& dir) {
  if (dir.empty()) return Status::InvalidArgument("final-state cache path is empty");
  const std::filesystem::path root(dir);
  FinalStateCache cache;
  auto metadata = final_state_cache_internal::ReadMetadata(root / "metadata.bin");
  if (!metadata.ok()) return metadata.status();
  cache.metadata = metadata.value();
  auto whitening = final_state_cache_internal::ReadBlob(root / "whitening.bin");
  if (!whitening.ok()) return whitening.status();
  cache.whitening = std::move(whitening.value());
  auto main = final_state_cache_internal::ReadBlob(root / "main.ivf");
  if (!main.ok()) return main.status();
  cache.main_index = std::move(main.value());
  if (cache.metadata.has_active) {
    auto active = final_state_cache_internal::ReadBlob(root / "active.ivf");
    if (!active.ok()) return active.status();
    cache.active_index = std::move(active.value());
  }
  if (cache.metadata.has_frozen) {
    auto frozen = final_state_cache_internal::ReadBlob(root / "frozen.ivf");
    if (!frozen.ok()) return frozen.status();
    cache.frozen_index = std::move(frozen.value());
  }
  auto gt = final_state_cache_internal::ReadGroundTruth(root / "ground_truth.bin");
  if (!gt.ok()) return gt.status();
  cache.ground_truth = std::move(gt.value());
  auto deleted = final_state_cache_internal::ReadBlob(root / "deleted.bin", true);
  if (!deleted.ok()) return deleted.status();
  cache.deleted = std::move(deleted.value());
  const auto vectors_path = root / "whitened_vectors.bin";
  if (std::filesystem::exists(root / "live_vectors.bin")) {
    auto s = final_state_cache_internal::ReadSparseWhitenedVectors(
        root / "live_vectors.bin", cache.metadata.dim, &cache.whitened_doc_ids,
        &cache.whitened_vectors);
    if (!s.ok()) return s;
  } else if (std::filesystem::exists(vectors_path)) {
    auto vectors = final_state_cache_internal::ReadWhitenedVectors(
        vectors_path, cache.metadata.seen_rows, cache.metadata.dim);
    if (!vectors.ok()) return vectors.status();
    cache.whitened_vectors = std::move(vectors.value());
  }
  Status validation = ValidateFinalStateCacheContents(cache);
  if (!validation.ok()) return validation;
  return cache;
}

}  // namespace ann
