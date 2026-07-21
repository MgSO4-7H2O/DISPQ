#pragma once

#include <optional>
#include <stdexcept>
#include <utility>

#include "common/status.h"

namespace ann {

template <typename T>
class Result {
 public:
  Result(const T& value) : status_(Status::OK()), value_(value) {}
  Result(T&& value) : status_(Status::OK()), value_(std::move(value)) {}
  Result(const Status& status) : status_(status) { EnsureNotOk(); }
  Result(Status&& status) : status_(std::move(status)) { EnsureNotOk(); }

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  const T& value() const {
    if (!ok()) {
      throw std::logic_error("Result accessed without value");
    }
    return *value_;
  }

  T& value() {
    if (!ok()) {
      throw std::logic_error("Result accessed without value");
    }
    return *value_;
  }

 private:
  void EnsureNotOk() {
    if (status_.ok()) {
      throw std::logic_error("Status-only Result must not be OK");
    }
  }

  Status status_;
  std::optional<T> value_;
};

template <>
class Result<void> {
 public:
  Result() : status_(Status::OK()) {}
  Result(const Status& status) : status_(status) {}
  Result(Status&& status) : status_(std::move(status)) {}

  static Result<void> Ok() { return Result<void>(); }

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

 private:
  Status status_;
};

}  // namespace ann
