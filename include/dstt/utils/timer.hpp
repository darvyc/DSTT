#pragma once

#include <chrono>
#include <string>
#include <iostream>

namespace dstt {

/// RAII high-resolution timer.
class ScopedTimer {
public:
    explicit ScopedTimer(std::string label)
        : label_(std::move(label)),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << "[timer] " << label_ << ": " << us << " µs\n";
    }

private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace dstt
