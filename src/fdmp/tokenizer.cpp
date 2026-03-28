#include <dstt/fdmp/tokenizer.hpp>
#include <dstt/utils/random.hpp>
#include <cmath>
#include <algorithm>
#include <sstream>

namespace dstt {

Tokenizer::Tokenizer(const Config& cfg)
    : cfg_(cfg), vocab_size_(cfg.vocab_size) {
    // Reserve IDs 0–255 for byte-level tokens
    for (uint32_t b = 0; b < 256; ++b) {
        std::string tok(1, static_cast<char>(b));
        token_to_id_[tok] = b;
        id_to_token_[b] = tok;
    }
}

void Tokenizer::init_embeddings() {
    size_t n = id_to_token_.size();
    size_t dim = cfg_.embed_dim;
    double limit = std::sqrt(6.0 / static_cast<double>(n + dim));

    embeddings_.resize(n * dim);
    RNG::seed(0xBEEF'CAFE);
    for (size_t i = 0; i < n * dim; ++i) {
        embeddings_[i] = RNG::uniform(-limit, limit);
    }
    RNG::seed(42);
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    // Split into individual bytes (characters)
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
        tokens.emplace_back(1, c);
    }
    return tokens;
}

std::unordered_map<std::string, size_t> Tokenizer::count_pairs(
        const std::vector<std::vector<std::string>>& tokenized) const {
    std::unordered_map<std::string, size_t> counts;
    for (const auto& seq : tokenized) {
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            std::string key = seq[i] + "|" + seq[i + 1];
            counts[key]++;
        }
    }
    return counts;
}

void Tokenizer::apply_merge(std::vector<std::vector<std::string>>& tokenized,
                             const std::string& a, const std::string& b) const {
    std::string merged = a + b;
    for (auto& seq : tokenized) {
        std::vector<std::string> new_seq;
        new_seq.reserve(seq.size());
        size_t i = 0;
        while (i < seq.size()) {
            if (i + 1 < seq.size() && seq[i] == a && seq[i + 1] == b) {
                new_seq.push_back(merged);
                i += 2;
            } else {
                new_seq.push_back(seq[i]);
                i++;
            }
        }
        seq = std::move(new_seq);
    }
}

void Tokenizer::build_vocab(const std::vector<std::string>& corpus) {
    // Pre-tokenize entire corpus to character level
    std::vector<std::vector<std::string>> tokenized;
    tokenized.reserve(corpus.size());
    for (const auto& text : corpus) {
        tokenized.push_back(pre_tokenize(text));
    }

    // BPE merge loop: merge most frequent pair until vocab_size reached
    uint32_t next_id = 256;  // byte tokens already occupy 0-255
    size_t target = vocab_size_;

    while (next_id < target) {
        auto pair_counts = count_pairs(tokenized);
        if (pair_counts.empty()) break;

        // Find the most frequent pair
        std::string best_key;
        size_t best_count = 0;
        for (const auto& [key, count] : pair_counts) {
            if (count > best_count) {
                best_count = count;
                best_key = key;
            }
        }

        if (best_count < cfg_.min_token_freq) break;

        // Split the key back into (a, b)
        size_t sep = best_key.find('|');
        std::string a = best_key.substr(0, sep);
        std::string b = best_key.substr(sep + 1);
        std::string merged = a + b;

        // Register the merged token
        if (token_to_id_.find(merged) == token_to_id_.end()) {
            token_to_id_[merged] = next_id;
            id_to_token_[next_id] = merged;
            merges_.emplace_back(a, b);
            next_id++;
        }

        // Apply merge across all sequences
        apply_merge(tokenized, a, b);
    }

    // Initialise learned embeddings now that vocab is fixed
    init_embeddings();
    trained_ = true;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    // Start with character-level tokens
    std::vector<std::string> tokens = pre_tokenize(text);

    // Apply learned merges in order
    for (const auto& [a, b] : merges_) {
        std::vector<std::string> new_tokens;
        new_tokens.reserve(tokens.size());
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() && tokens[i] == a && tokens[i + 1] == b) {
                new_tokens.push_back(a + b);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        tokens = std::move(new_tokens);
    }

    // Map to IDs
    std::vector<uint32_t> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens) {
        auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Unknown token: fall back to individual bytes
            for (char c : tok) {
                ids.push_back(static_cast<uint32_t>(static_cast<unsigned char>(c)));
            }
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;
    for (uint32_t id : tokens) {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            result += it->second;
        }
    }
    return result;
}

Vec Tokenizer::embed_tokens(const std::vector<uint32_t>& tokens) const {
    size_t dim = cfg_.embed_dim;
    Vec result(dim, 0.0);

    if (tokens.empty()) return result;

    size_t vocab = id_to_token_.size();
    for (uint32_t id : tokens) {
        if (id < vocab) {
            size_t offset = static_cast<size_t>(id) * dim;
            for (size_t d = 0; d < dim; ++d) {
                result[d] += embeddings_[offset + d];
            }
        }
    }

    // Average
    double n = static_cast<double>(tokens.size());
    for (size_t d = 0; d < dim; ++d) {
        result[d] /= n;
    }

    // L2 normalise
    double norm = 0.0;
    for (double x : result) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (auto& x : result) x /= norm;
    }

    return result;
}

void Tokenizer::set_vocab(std::unordered_map<std::string, uint32_t> t2i,
                           std::unordered_map<uint32_t, std::string> i2t) {
    token_to_id_ = std::move(t2i);
    id_to_token_ = std::move(i2t);
    trained_ = true;
}

} // namespace dstt
