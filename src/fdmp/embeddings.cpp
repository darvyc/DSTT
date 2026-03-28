#include <dstt/fdmp/embeddings.hpp>
#include <cmath>
#include <functional>

namespace dstt { namespace Embeddings {

/// Simple deterministic hash-to-vector.
static Vec hash_embed(const std::string& input, size_t dim, uint64_t salt) {
    Vec v(dim);
    std::hash<std::string> hasher;
    uint64_t h = hasher(input) ^ salt;
    for (size_t i = 0; i < dim; ++i) {
        // Xorshift-style mixing
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        // Map to [-1, 1]
        v[i] = static_cast<double>(h & 0xFFFF) / 32768.0 - 1.0;
    }
    // L2 normalise
    double norm = 0.0;
    for (double x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (auto& x : v) x /= norm;
    }
    return v;
}

Vec text_embedding(const std::string& prompt, size_t dim) {
    return hash_embed(prompt, dim, 0xCAFE'BABE'0001ULL);
}

Vec image_embedding(const std::string& description, size_t dim) {
    return hash_embed(description, dim, 0xDEAD'BEEF'0002ULL);
}

Vec video_embedding(const std::string& description, size_t dim) {
    return hash_embed(description, dim, 0xFEED'FACE'0003ULL);
}

Vec embed(const std::string& input, Modality m, size_t dim) {
    switch (m) {
        case Modality::Text:  return text_embedding(input, dim);
        case Modality::Image: return image_embedding(input, dim);
        case Modality::Video: return video_embedding(input, dim);
    }
    return text_embedding(input, dim);
}

}} // namespace dstt::Embeddings
