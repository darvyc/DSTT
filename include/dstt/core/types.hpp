#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
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

    // ── Training phase ──────────────────────────────────────────────
    size_t training_epochs     = 10;
    double training_lr         = 0.005;    // FDMP weight learning rate
    size_t vocab_size          = 4096;     // Tokenizer vocabulary size
    size_t min_token_freq      = 2;        // BPE minimum pair frequency
    double weight_decay        = 1e-4;     // L2 regularisation

    // ── System prompt ───────────────────────────────────────────────
    // Defines the model's training focus, analogous to a system prompt
    // in an LLM.  Prepended to every training example as context so
    // the model's learned weights are biased toward this objective.
    std::string system_prompt = "Be a helpful assistant.";

    /// Effective mutation rate (auto if zero).
    double effective_mutation_rate() const {
        return (mutation_rate > 0.0) ? mutation_rate : 1.0 / static_cast<double>(param_dim);
    }

    /// Total trainable parameter count.
    ///   Token embeddings:  vocab_size × embed_dim
    ///   FDMP weights:      3 × (param_dim × embed_dim + param_dim)
    size_t total_parameters() const {
        size_t emb_params  = vocab_size * embed_dim;
        size_t fdmp_params = MODALITY_COUNT * (param_dim * embed_dim + param_dim);
        return emb_params + fdmp_params;
    }

    /// Configure embed_dim and param_dim to approximate a target
    /// parameter count, keeping vocab_size fixed.
    ///
    /// Heuristic: embed_dim : param_dim ≈ 2:1 (like hidden:output in NNs).
    /// Solves:  vocab_size * E + 3 * (P * E + P) ≈ target
    ///   where E = embed_dim, P = E/2.
    void set_parameter_count(size_t target_params) {
        // target ≈ vocab_size * E + 3 * (E/2 * E + E/2)
        //        = vocab_size * E + 3 * E²/2 + 3*E/2
        //        ≈ vocab_size * E + 1.5 * E²   (dropping small linear term)
        // Quadratic: 1.5 * E² + vocab_size * E - target = 0
        // E = (-vocab_size + sqrt(vocab_size² + 6 * target)) / 3

        double v = static_cast<double>(vocab_size);
        double t = static_cast<double>(target_params);
        double discriminant = v * v + 6.0 * t;
        double E = (-v + std::sqrt(discriminant)) / 3.0;

        embed_dim = std::max(size_t(8), static_cast<size_t>(E));
        param_dim = std::max(size_t(4), embed_dim / 2);
    }
};

} // namespace dstt
