#pragma once

#include <dstt/core/types.hpp>
#include <dstt/fdmp/fdmp.hpp>
#include <dstt/fdmp/tokenizer.hpp>
#include <dstt/core/arm.hpp>
#include <dstt/ea/population.hpp>
#include <string>
#include <vector>
#include <functional>

namespace dstt {

/// A single training example: input text paired with a target modality.
struct TrainingExample {
    std::string input;
    Modality modality = Modality::Text;
};

/// Per-epoch training statistics.
struct EpochStats {
    size_t epoch = 0;
    double avg_fitness = 0.0;
    double avg_loss = 0.0;
    double best_fitness = 0.0;
};

using EpochCallback = std::function<void(const EpochStats&)>;

/// Training Phase controller.
///
/// Trains FDMP weight matrices and token embeddings on a corpus of
/// training examples.  For each example the pipeline runs:
///
///   1. Tokenize input → token IDs
///   2. Embed tokens → context vector C
///   3. Generate raw parameters Θ_r = W_m · C + b_m
///   4. Evolve Θ via EA → best fitness F*
///   5. Compute loss L = 1 − F* (fitness gap)
///   6. Update FDMP weights via finite-difference gradient approximation:
///        ∂L/∂W ≈ (L(W+ε) − L(W−ε)) / 2ε
///   7. Update token embeddings via gradient from fitness signal
///
/// After training, the FDMP produces better initial parameters for
/// unseen prompts, reducing EA generations needed at inference.
class Trainer {
public:
    explicit Trainer(const Config& cfg);

    /// Build the tokenizer vocabulary from training texts.
    void build_tokenizer(const std::vector<std::string>& corpus);

    /// Run the full training loop over a corpus.
    /// @param examples   Training examples.
    /// @param on_epoch   Optional per-epoch callback.
    void train(const std::vector<TrainingExample>& examples,
               EpochCallback on_epoch = nullptr);

    /// Access the trained FDMP (for inference or serialization).
    const FDMP& fdmp() const { return fdmp_; }
    FDMP& fdmp() { return fdmp_; }

    /// Access the trained tokenizer.
    const Tokenizer& tokenizer() const { return tokenizer_; }
    Tokenizer& tokenizer() { return tokenizer_; }

    /// Save trained weights and tokenizer to a directory.
    void save(const std::string& path) const;

    /// Load trained weights and tokenizer from a directory.
    void load(const std::string& path);

    /// Whether training has been completed.
    bool is_trained() const { return trained_; }

private:
    Config cfg_;
    FDMP fdmp_;
    Tokenizer tokenizer_;
    bool trained_ = false;

    /// Run one training step on a single example.
    /// @return (fitness, loss) for the example.
    std::pair<double, double> train_step(const TrainingExample& ex);

    /// Update FDMP weights using the fitness gradient signal.
    void update_weights(const Vec& context, Modality m,
                        double fitness, double loss);

    /// Update token embeddings from the fitness signal.
    void update_embeddings(const std::vector<uint32_t>& tokens,
                           const Vec& context, double loss);
};

} // namespace dstt
