#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cassert>

namespace dstt {

// ── Modality enum ────────────────────────────────────────────────────
enum class Modality : uint8_t { Text = 0, Image = 1, Video = 2 };

constexpr size_t MODALITY_COUNT = 3;

inline const char* modality_name(Modality m) {
    switch (m) {
        case Modality::Text:  return "Text";
        case Modality::Image: return "Image";
        case Modality::Video: return "Video";
    }
    return "Unknown";
}

// ── Dense vector (column vector in R^n) ──────────────────────────────
using Vec = std::vector<double>;

// ── Hyperparameter defaults ──────────────────────────────────────────
struct Config {
    // EA
    size_t population_size   = 200;
    size_t max_generations   = 500;
    size_t tournament_k      = 5;
    double elitism_rate      = 0.10;
    // mutation_rate defaults to 1/m (set at runtime)
    double mutation_rate     = 0.0;   // 0 ⇒ auto = 1/param_dim

    // Fitness weights
    double w_coherence = 0.4;
    double w_relevance = 0.4;
    double w_diversity = 0.2;

    // Partitioning
    double coherence_threshold = 0.25;  // τ

    // MGE branch predictor
    double bp_confidence_threshold = 0.6;

    // Convergence
    double convergence_eps   = 1e-6;
    size_t convergence_window = 20;

    // Embedding dimension
    size_t embed_dim = 128;

    // Parameter dimension (m) — set by the user
    size_t param_dim = 64;

    /// Effective mutation rate (auto if zero).
    double effective_mutation_rate() const {
        return (mutation_rate > 0.0) ? mutation_rate : 1.0 / static_cast<double>(param_dim);
    }
};

} // namespace dstt
