#include "common/serialization.h"

#include <fstream>
#include <iterator>

namespace ann {

Result<void> SaveBinary(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    return Status::IOError("Failed to open file for write: " + path);
  }
  ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!ofs) {
    return Status::IOError("Failed to write file: " + path);
  }
  return Result<void>();
}

Result<std::vector<uint8_t>> LoadBinary(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return Status::IOError("Failed to open file for read: " + path);
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return bytes;
}

}  // namespace ann
