#pragma once

#include <dstt/core/types.hpp>
#include <dstt/core/partition.hpp>
#include <utility>

namespace dstt {

/// ARM evaluation result for a single pass.
struct ARMResult {
    Vec cfm_scores;      ///< Per-parameter CFM scores.
    Vec afm_scores;      ///< Per-parameter AFM scores.
    Vec adjusted;        ///< θ' = θ · (CFM − AFM).
    Vec probabilities;   ///< softmax(θ').
    size_t sampled_idx;  ///< Index of sampled parameter.
    PartitionSet partitions;
};

/// Autonomous Route Matrix.
///
/// ARM: ℝᵐ × ℝⁿ × M → ℝᵐ
///
/// Performs:  partition → evaluate (CFM, AFM) → adjust → softmax → sample.
class ARM {
public:
    explicit ARM(const Config& cfg);

    /// Full evaluation pass.
    /// @param theta    Raw parameter vector Θ ∈ ℝᵐ.
    /// @param context  Context embedding C ∈ ℝⁿ.
    /// @param prev_state  Previous state embedding.
    /// @param modality Current modality.
    ARMResult evaluate(const Vec& theta,
                       const Vec& context,
                       const Vec& prev_state,
                       Modality modality) const;

private:
    Config cfg_;
};

} // namespace dstt
