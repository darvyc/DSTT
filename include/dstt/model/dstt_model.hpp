#pragma once

#include <dstt/core/types.hpp>
#include <dstt/model/dstt_format.hpp>
#include <dstt/fdmp/fdmp.hpp>
#include <dstt/fdmp/tokenizer.hpp>
#include <dstt/core/arm.hpp>
#include <dstt/ea/population.hpp>
#include <dstt/mge/branch_predictor.hpp>
#include <dstt/mge/synthesiser.hpp>
#include <dstt/training/trainer.hpp>
#include <string>
#include <vector>
#include <functional>

namespace dstt {

/// Callback for streaming generation output per step.
using StepCallback = std::function<void(size_t step, const OutputElement& elem)>;

/// DSTTModel — load and interact with trained .dstt model files.
///
/// This is the primary interface for inference, analogous to loading
/// a .gguf file in llama.cpp.  A .dstt file packages the full trained
/// state (vocabulary, embeddings, FDMP weights) into a single file.
///
/// Usage:
///   DSTTModel model;
///   model.load("models/my_model.dstt");
///   auto output = model.generate("A sunset over mountains");
///
///   // Or interactive:
///   model.generate("prompt", 12, [](size_t step, const OutputElement& e) {
///       std::cout << step << ": " << modality_name(e.modality) << "\n";
///   });
class DSTTModel {
public:
    DSTTModel();
    explicit DSTTModel(const Config& cfg);

    /// Load a trained model from a .dstt file.
    /// @throws std::runtime_error if the file is invalid or corrupted.
    void load(const std::string& path);

    /// Save the current model state to a .dstt file.
    void save(const std::string& path) const;

    /// Train a model from training examples and save as .dstt.
    void train_and_save(const std::vector<TrainingExample>& examples,
                        const std::string& output_path,
                        EpochCallback on_epoch = nullptr);

    /// Generate multi-modal output from a text prompt.
    /// @param prompt  Input text prompt.
    /// @param steps   Number of output elements to generate.
    /// @param on_step Optional per-step callback for streaming output.
    /// @return The full synthesised output.
    SynthesisedOutput generate(const std::string& prompt,
                               size_t steps = 12,
                               StepCallback on_step = nullptr);

    /// Get model metadata as a human-readable string.
    std::string info() const;

    /// Check if a model is loaded and ready for inference.
    bool is_loaded() const { return loaded_; }

    /// Access the underlying config.
    const Config& config() const { return cfg_; }

    /// Load training data from a JSONL file.
    /// Format: {"input": "text", "modality": "Text|Image|Video"}
    static std::vector<TrainingExample> load_training_jsonl(const std::string& path);

    /// Load training data from a plain text file (one example per line).
    /// All examples default to Text modality.
    static std::vector<TrainingExample> load_training_txt(const std::string& path);

    /// Load training data from a CSV file.
    /// Format: input,modality
    static std::vector<TrainingExample> load_training_csv(const std::string& path);

private:
    Config cfg_;
    FDMP fdmp_;
    Tokenizer tokenizer_;
    bool loaded_ = false;

    /// Write all model data to a binary stream.
    void write_to_stream(std::ostream& out) const;

    /// Read all model data from a binary stream.
    void read_from_stream(std::istream& in);
};

} // namespace dstt
