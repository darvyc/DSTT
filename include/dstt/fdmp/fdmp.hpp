#pragma once

#include <dstt/core/types.hpp>
#include <string>

namespace dstt {

/// Fundamental Data Matrix Processor.
///
/// Encodes raw input data into structured embeddings across linguistic,
/// visual, and auditory modalities.  Maintains modality-specific
/// weight matrices W_m and bias vectors b_m for the affine transform
///   Θ_r = W_m · C + b_m
class FDMP {
public:
    explicit FDMP(const Config& cfg);

    /// Encode an input string into a context embedding C ∈ ℝⁿ.
    Vec encode(const std::string& input, Modality m) const;

    /// Generate raw parameter scores from a context embedding.
    ///   Θ_r = W_m · C + b_m
    Vec generate_params(const Vec& context, Modality m) const;

    /// Combined: encode input, then generate raw parameter scores.
    std::pair<Vec, Vec> process(const std::string& input, Modality m) const;

    size_t embed_dim() const { return cfg_.embed_dim; }
    size_t param_dim() const { return cfg_.param_dim; }

private:
    Config cfg_;
    // Weight matrices stored row-major: W_m is param_dim × embed_dim
    // One per modality
    Vec W_text_, W_image_, W_video_;
    Vec b_text_, b_image_, b_video_;

    const Vec& weight_matrix(Modality m) const;
    const Vec& bias_vector(Modality m) const;

    void init_weights();
};

} // namespace dstt
