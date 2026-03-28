#pragma once

#include <dstt/core/types.hpp>
#include <dstt/model/dstt_format.hpp>
#include <dstt/model/content_generator.hpp>
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
using ContentStepCallback = std::function<void(size_t step,
                                               const GeneratedContent& content)>;

/// DSTTModel — load and interact with trained .dstt model files.
///
/// This is the primary interface for inference, analogous to loading
/// a .gguf file in llama.cpp.  A .dstt file packages the full trained
/// state (vocabulary, embeddings, FDMP weights) into a single file.
///
/// After training, the model generates actual content:
///   - **Text**: Samples tokens from the learned vocabulary using
///     evolved parameter distributions and trained embeddings.
///   - **Image**: Maps evolved parameters to RGB pixel data, producing
///     64x64 PPM images influenced by the prompt context.
///   - **Video**: Generates temporally coherent frame sequences by
///     modulating parameters over time.
///
/// Usage:
///   DSTTModel model;
///   model.load("models/my_model.dstt");
///   auto result = model.run("A sunset over mountains");
///   std::cout << result.generated_text << "\n";
///   result.images[0].save_ppm("output.ppm");
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

    /// Generate multi-modal content from a text prompt.
    /// Produces text, images, and/or video depending on what the
    /// branch predictor selects for each step.
    /// @param prompt   Input text prompt.
    /// @param steps    Number of generation steps.
    /// @param on_step  Optional per-step callback with generated content.
    /// @return Full generation result with text, images, and video.
    GenerationResult run(const std::string& prompt,
                         size_t steps = 12,
                         ContentStepCallback on_step = nullptr);

    /// Generate raw parameter output (low-level, without content synthesis).
    SynthesisedOutput generate_raw(const std::string& prompt,
                                   size_t steps = 12);

    /// Get model metadata as a human-readable string.
    std::string info() const;

    /// Check if a model is loaded and ready for inference.
    bool is_loaded() const { return loaded_; }

    /// Access the underlying config.
    const Config& config() const { return cfg_; }

    /// Load training data from a JSONL file.
    static std::vector<TrainingExample> load_training_jsonl(const std::string& path);

    /// Load training data from a plain text file (one example per line).
    static std::vector<TrainingExample> load_training_txt(const std::string& path);

    /// Load training data from a CSV file.
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
