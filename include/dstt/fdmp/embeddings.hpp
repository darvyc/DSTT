#pragma once

#include <dstt/core/types.hpp>

namespace dstt {

/// Modality-specific embedding helpers.
///
/// In a production system these would wrap actual feature extractors
/// (tokenisers, CNNs, mel-spectrograms).  Here they provide a
/// deterministic simulation for testing and demonstration.
namespace Embeddings {

    /// Generate a synthetic context embedding for a text prompt.
    /// Uses a simple hash-based scheme to produce a repeatable vector.
    Vec text_embedding(const std::string& prompt, size_t dim);

    /// Generate a synthetic context embedding for an image description.
    Vec image_embedding(const std::string& description, size_t dim);

    /// Generate a synthetic context embedding for a video description.
    Vec video_embedding(const std::string& description, size_t dim);

    /// Generic embedding dispatcher.
    Vec embed(const std::string& input, Modality m, size_t dim);

} // namespace Embeddings

} // namespace dstt
