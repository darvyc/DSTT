#pragma once

#include <random>
#include <cstddef>
#include <dstt/core/types.hpp>

namespace dstt {

/// Thread-local random number generator.
class RNG {
public:
    /// Seed the global instance (call once at startup).
    static void seed(uint64_t s);

    /// Uniform double in [lo, hi).
    static double uniform(double lo = 0.0, double hi = 1.0);

    /// Uniform integer in [lo, hi].
    static size_t uniform_int(size_t lo, size_t hi);

    /// Generate a random vector in [lo, hi)^n.
    static Vec random_vec(size_t n, double lo = 0.0, double hi = 1.0);

private:
    static thread_local std::mt19937_64 gen_;
};

} // namespace dstt
