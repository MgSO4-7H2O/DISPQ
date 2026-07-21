#pragma once

#include <chrono>

namespace ann {

class Timer {
 public:
  Timer() { Reset(); }

  void Reset() { start_ = Clock::now(); }

  double ElapsedMillis() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace ann
