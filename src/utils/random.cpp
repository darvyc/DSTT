#include <dstt/utils/random.hpp>

namespace dstt {

thread_local std::mt19937_64 RNG::gen_{42};

void RNG::seed(uint64_t s) { gen_.seed(s); }

double RNG::uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen_);
}

size_t RNG::uniform_int(size_t lo, size_t hi) {
    std::uniform_int_distribution<size_t> dist(lo, hi);
    return dist(gen_);
}

Vec RNG::random_vec(size_t n, double lo, double hi) {
    Vec v(n);
    for (auto& x : v) x = uniform(lo, hi);
    return v;
}

} // namespace dstt
