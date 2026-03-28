#pragma once

#include <dstt/core/types.hpp>
#include <array>

namespace dstt {

/// Branch Predictor: predicts the next modality to generate.
///
/// BP: ℝⁿ → M
///
/// Implemented as a small linear model over the context embedding,
/// mapping to a softmax over modalities.  When confidence < threshold,
/// falls back to the modality with the longest time since last generation.
class BranchPredictor {
public:
    explicit BranchPredictor(const Config& cfg);

    /// Predict the next modality given current context.
    /// @param context  Context embedding.
    /// @return (predicted modality, confidence).
    std::pair<Modality, double> predict(const Vec& context) const;

    /// Update predictor weights after observing an actual modality.
    /// Uses a simple online gradient step.
    void update(const Vec& context, Modality actual);

    /// Record that modality m was just generated (resets its timer).
    void record_generation(Modality m);

    /// Get the modality with the longest time since last generation.
    Modality least_recent() const;

private:
    Config cfg_;
    // Simple weight matrix: MODALITY_COUNT × embed_dim (row-major)
    Vec weights_;
    Vec biases_;
    double learning_rate_ = 0.01;

    // Time-since-last counters
    std::array<size_t, MODALITY_COUNT> last_gen_;
    size_t step_ = 0;
};

} // namespace dstt
