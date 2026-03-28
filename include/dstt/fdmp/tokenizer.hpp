#pragma once

#include <dstt/core/types.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace dstt {

/// Byte-Pair Encoding tokenizer.
///
/// Learns a vocabulary of sub-word units from a training corpus,
/// then encodes input text into token IDs.  Token embeddings are
/// stored as a learnable matrix E ∈ ℝ^{vocab_size × embed_dim}.
class Tokenizer {
public:
    explicit Tokenizer(const Config& cfg);

    /// Build vocabulary from a training corpus using BPE merges.
    /// @param corpus  Collection of training texts.
    void build_vocab(const std::vector<std::string>& corpus);

    /// Encode a string into a sequence of token IDs.
    std::vector<uint32_t> encode(const std::string& text) const;

    /// Decode token IDs back to a string.
    std::string decode(const std::vector<uint32_t>& tokens) const;

    /// Convert a token sequence into a fixed-size context embedding
    /// by averaging learned token embeddings.
    ///   C = (1/|tokens|) Σ E[token_i]
    Vec embed_tokens(const std::vector<uint32_t>& tokens) const;

    /// Access the token embedding matrix (for training updates).
    Vec& embedding_matrix() { return embeddings_; }
    const Vec& embedding_matrix() const { return embeddings_; }

    size_t vocab_size() const { return vocab_size_; }
    size_t actual_vocab_size() const { return id_to_token_.size(); }
    bool is_trained() const { return trained_; }

    /// Token-to-ID and ID-to-token maps (for serialization).
    const std::unordered_map<std::string, uint32_t>& token_to_id() const { return token_to_id_; }
    const std::unordered_map<uint32_t, std::string>& id_to_token() const { return id_to_token_; }

    /// Restore vocabulary from maps (used when loading).
    void set_vocab(std::unordered_map<std::string, uint32_t> t2i,
                   std::unordered_map<uint32_t, std::string> i2t);

private:
    Config cfg_;
    size_t vocab_size_;

    // BPE merge rules: (token_a, token_b) → merged token
    std::vector<std::pair<std::string, std::string>> merges_;

    // Vocabulary maps
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_map<uint32_t, std::string> id_to_token_;

    // Learnable token embedding matrix: vocab_size × embed_dim (row-major)
    Vec embeddings_;

    bool trained_ = false;

    /// Initialise embeddings with Xavier distribution.
    void init_embeddings();

    /// Split text into initial character-level tokens.
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    /// Count adjacent pair frequencies across tokenized corpus.
    std::unordered_map<std::string, size_t> count_pairs(
        const std::vector<std::vector<std::string>>& tokenized) const;

    /// Apply one merge to all sequences.
    void apply_merge(std::vector<std::vector<std::string>>& tokenized,
                     const std::string& a, const std::string& b) const;
};

} // namespace dstt
