#include <dstt/utils/math.hpp>

namespace dstt { namespace math {

double dot(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double norm(const Vec& a) {
    return std::sqrt(dot(a, a));
}

double cosine_similarity(const Vec& a, const Vec& b) {
    double d = dot(a, b);
    double na = norm(a);
    double nb = norm(b);
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return d / (na * nb);
}

size_t hamming_distance(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    size_t dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int ba = (a[i] >= 0.0) ? 1 : 0;
        int bb = (b[i] >= 0.0) ? 1 : 0;
        if (ba != bb) ++dist;
    }
    return dist;
}

double self_information(double p) {
    if (p <= 0.0) return 0.0;
    return -p * std::log2(p);
}

Vec softmax(const Vec& v) {
    if (v.empty()) return {};
    double mx = *std::max_element(v.begin(), v.end());
    Vec out(v.size());
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = std::exp(v[i] - mx);   // numerically stable
        sum += out[i];
    }
    for (auto& x : out) x /= sum;
    return out;
}

Vec affine(const std::vector<double>& W, size_t rows, size_t cols,
           const Vec& x, const Vec& bias) {
    assert(W.size() == rows * cols);
    assert(x.size() == cols);
    assert(bias.size() == rows);
    Vec y(rows, 0.0);
    for (size_t r = 0; r < rows; ++r) {
        double s = bias[r];
        for (size_t c = 0; c < cols; ++c) {
            s += W[r * cols + c] * x[c];
        }
        y[r] = s;
    }
    return y;
}

size_t inverse_cdf_sample(const Vec& probs, double u) {
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (u < cum) return i;
    }
    return probs.size() - 1;  // fallback
}

size_t partition_count(size_t m) {
    if (m == 0) return 1;
    // Hardy–Ramanujan asymptotic: log2(p(n)) ≈ π√(2n/3) / ln(2) - log2(4n√3)
    double n = static_cast<double>(m);
    double log2_p = M_PI * std::sqrt(2.0 * n / 3.0) / std::log(2.0)
                  - std::log2(4.0 * n * std::sqrt(3.0));
    size_t k = static_cast<size_t>(std::floor(log2_p));
    return std::max<size_t>(k, 2);  // at least 2 partitions
}

}} // namespace dstt::math
