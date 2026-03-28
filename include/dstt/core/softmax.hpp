#pragma once

#include <dstt/core/types.hpp>

namespace dstt {

/// Adjustment + softmax + sampling pipeline.
///
///   θ'_j = θ_j · (CFM_j − AFM_j)
///   P(θ_j | m) = softmax(θ')
///   θ* ~ P   via inverse-CDF sampling
struct AdjustAndSample {
    /// Compute adjusted scores from raw parameters and CFM/AFM scores.
    /// Returns θ' vector.
    static Vec adjust(const Vec& theta, const Vec& cfm_scores, const Vec& afm_scores);

    /// Sample an index from a probability distribution.
    static size_t sample(const Vec& probs);

    /// Full pipeline: adjust → softmax → sample.
    /// Returns (sampled_index, probability_distribution).
    static std::pair<size_t, Vec> run(const Vec& theta,
                                      const Vec& cfm_scores,
                                      const Vec& afm_scores);
};

} // namespace dstt
