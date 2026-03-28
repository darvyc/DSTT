#pragma once

#include <dstt/core/types.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace dstt { namespace math {

/// Dot product of two equal-length vectors.
double dot(const Vec& a, const Vec& b);

/// L2 norm.
double norm(const Vec& a);

/// Cosine similarity in [-1, 1].
double cosine_similarity(const Vec& a, const Vec& b);

/// Hamming distance between binary representations.
/// Each value is thresholded at 0 to produce a bit.
size_t hamming_distance(const Vec& a, const Vec& b);

/// Shannon self-information: -p * log2(p).  Returns 0 for p <= 0.
double self_information(double p);

/// Softmax of a vector.  Returns a probability distribution.
Vec softmax(const Vec& v);

/// Affine transform: W * x + b.
/// W is stored row-major as a flat vector of size rows*cols.
Vec affine(const std::vector<double>& W, size_t rows, size_t cols,
           const Vec& x, const Vec& bias);

/// Inverse-CDF sampling: given a probability distribution, return the
/// sampled index using a uniform random value u ∈ [0, 1).
size_t inverse_cdf_sample(const Vec& probs, double u);

/// Partition count: k = floor(log2(p(m))).
/// Uses Hardy–Ramanujan asymptotic: p(n) ~ exp(π√(2n/3)) / (4n√3).
size_t partition_count(size_t m);

}} // namespace dstt::math
