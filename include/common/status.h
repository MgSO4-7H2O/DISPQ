#pragma once

#include <string>
#include <utility>

namespace ann {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kIOError,
  kInternal,
  kUnimplemented,
  kUnavailable
};

class Status {
 public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, std::string message = {})
      : code_(code), message_(std::move(message)) {}

  static Status OK() { return Status(); }
  static Status InvalidArgument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }
  static Status NotFound(std::string message) {
    return Status(StatusCode::kNotFound, std::move(message));
  }
  static Status AlreadyExists(std::string message) {
    return Status(StatusCode::kAlreadyExists, std::move(message));
  }
  static Status IOError(std::string message) {
    return Status(StatusCode::kIOError, std::move(message));
  }
  static Status Internal(std::string message) {
    return Status(StatusCode::kInternal, std::move(message));
  }
  static Status Unimplemented(std::string message) {
    return Status(StatusCode::kUnimplemented, std::move(message));
  }
  static Status Unavailable(std::string message) {
    return Status(StatusCode::kUnavailable, std::move(message));
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

  std::string ToString() const {
    if (ok()) {
      return "OK";
    }
    return MessageForCode(code_) + ": " + message_;
  }

 private:
  static std::string MessageForCode(StatusCode code) {
    switch (code) {
      case StatusCode::kOk:
        return "OK";
      case StatusCode::kInvalidArgument:
        return "InvalidArgument";
      case StatusCode::kNotFound:
        return "NotFound";
      case StatusCode::kAlreadyExists:
        return "AlreadyExists";
      case StatusCode::kIOError:
        return "IOError";
      case StatusCode::kInternal:
        return "Internal";
      case StatusCode::kUnimplemented:
        return "Unimplemented";
      case StatusCode::kUnavailable:
        return "Unavailable";
    }
    return "Unknown";
  }

  StatusCode code_;
  std::string message_;
};

}  // namespace ann
