#pragma once

#include <dstt/core/types.hpp>

namespace dstt {

/// Correct Flow Matrix (CFM).
///
/// CFM(θ_j, m) = Ws(θ_j, C, m) + Rc(θ_j, S_prev, m)
///
/// - Ws: cosine similarity between parameter embedding and context
///        embedding, scaled by modality weight α_m.
/// - Rc: 1 − d_H(θ_j, S_prev) / m   (normalised Hamming coherence).
struct CFM {
    /// Compute CFM score for a single parameter.
    /// @param theta_j       The raw parameter value.
    /// @param param_embed   Embedding of θ_j (precomputed).
    /// @param context_embed Context embedding C.
    /// @param prev_state    Previous-state embedding S_prev.
    /// @param modality      Current modality.
    static double score(double theta_j,
                        const Vec& param_embed,
                        const Vec& context_embed,
                        const Vec& prev_state,
                        Modality modality);

    /// Modality-specific weight α_m.
    static double alpha(Modality m);
};

} // namespace dstt
