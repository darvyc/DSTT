#pragma once

#include <dstt/core/types.hpp>
#include <cstdint>
#include <string>

namespace dstt {

/// .dstt file format — a single binary file containing all trained model data.
///
/// Layout:
///   [Header]           Fixed-size header with magic, version, dimensions
///   [Config Block]     Hyperparameters used during training
///   [Vocabulary]       BPE token table (count + entries)
///   [Embeddings]       Token embedding matrix (vocab × embed_dim)
///   [FDMP Weights]     Per-modality W matrices and bias vectors
///
/// Inspired by GGUF: a single self-contained file for the full model.

constexpr uint32_t DSTT_MAGIC   = 0x44535454;  // "DSTT" in little-endian
constexpr uint32_t DSTT_VERSION = 1;

struct DSTTHeader {
    uint32_t magic       = DSTT_MAGIC;
    uint32_t version     = DSTT_VERSION;
    uint32_t embed_dim   = 0;
    uint32_t param_dim   = 0;
    uint32_t vocab_size  = 0;   // Actual vocabulary size (tokens stored)
    uint32_t modality_count = MODALITY_COUNT;
    uint32_t reserved[2] = {0, 0};  // Future use
};

struct DSTTConfigBlock {
    double training_lr    = 0.0;
    double weight_decay   = 0.0;
    uint32_t training_epochs = 0;
    uint32_t max_vocab_size  = 0;
    double w_coherence    = 0.0;
    double w_relevance    = 0.0;
    double w_diversity    = 0.0;
    double coherence_threshold = 0.0;
    double bp_confidence_threshold = 0.0;
    uint32_t population_size   = 0;
    uint32_t max_generations   = 0;
    uint32_t tournament_k      = 0;
    double elitism_rate        = 0.0;
    double mutation_rate       = 0.0;
};

/// Returns the expected file extension for DSTT model files.
inline std::string dstt_extension() { return ".dstt"; }

} // namespace dstt
