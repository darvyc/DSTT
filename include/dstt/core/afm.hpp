#pragma once

#include <dstt/core/types.hpp>

namespace dstt {

/// Adversarial Flow Matrix (AFM).
///
/// AFM(θ_j, m) = Ca(θ_j, C, m) + Es(θ_j, m)
///
/// - Ca: max(0, −cos(E(θ_j), E(C)))   (rectified negative cosine).
/// - Es: −P(θ_j) · log₂(P(θ_j))       (Shannon self-information).
struct AFM {
    /// Compute AFM score for a single parameter.
    /// @param raw_prob      Current (pre-adjustment) probability P(θ_j).
    /// @param param_embed   Embedding of θ_j.
    /// @param context_embed Context embedding C.
    /// @param modality      Current modality.
    static double score(double raw_prob,
                        const Vec& param_embed,
                        const Vec& context_embed,
                        Modality modality);
};

} // namespace dstt
