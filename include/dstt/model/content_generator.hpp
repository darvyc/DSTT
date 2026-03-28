#pragma once

#include <dstt/core/types.hpp>
#include <dstt/fdmp/tokenizer.hpp>
#include <string>
#include <vector>

namespace dstt {

/// Image descriptor produced by DSTT image generation.
struct ImageDescriptor {
    size_t width  = 0;
    size_t height = 0;
    size_t channels = 3;  // RGB
    Vec pixels;            // Row-major, size = width * height * channels, values in [0,1]

    /// Save as PPM image file (portable, no dependencies).
    bool save_ppm(const std::string& path) const;
};

/// Video descriptor produced by DSTT video generation.
struct VideoDescriptor {
    size_t width  = 0;
    size_t height = 0;
    size_t channels = 3;
    double fps = 8.0;
    std::vector<Vec> frames;  // Each frame: row-major pixels, same layout as ImageDescriptor

    /// Save as a sequence of PPM frames in a directory.
    bool save_frames(const std::string& dir_prefix) const;
};

/// Generated content from a single modality output.
struct GeneratedContent {
    Modality modality = Modality::Text;

    // Text output (populated when modality == Text)
    std::string text;

    // Image output (populated when modality == Image)
    ImageDescriptor image;

    // Video output (populated when modality == Video)
    VideoDescriptor video;
};

/// Full generation result combining all modalities.
struct GenerationResult {
    std::string prompt;

    // Accumulated text across all Text steps
    std::string generated_text;

    // Images produced (one per Image step)
    std::vector<ImageDescriptor> images;

    // Video produced (accumulated from Video steps)
    VideoDescriptor video;

    // Per-step details
    std::vector<GeneratedContent> steps;

    // Statistics
    size_t text_steps  = 0;
    size_t image_steps = 0;
    size_t video_steps = 0;
    double avg_probability = 0.0;
};

/// ContentGenerator — transforms DSTT parameter configurations into
/// actual text, image, and video content.
///
/// For text: uses the probability distribution from ARM evaluation to
/// compute similarity scores against learned token embeddings, then
/// samples tokens via temperature-controlled softmax. This mirrors
/// how LLMs sample from a vocabulary distribution.
///
/// For images: maps evolved parameter embeddings to pixel values in a
/// spatial grid, producing RGB image data.
///
/// For video: extends image generation with temporal variation across
/// frames, using parameter evolution to drive frame-to-frame changes.
class ContentGenerator {
public:
    explicit ContentGenerator(const Config& cfg);

    /// Generate text tokens from ARM probability distribution and context.
    /// @param probs       ARM probability distribution over parameters.
    /// @param theta       Evolved parameter vector.
    /// @param context     Context embedding from tokenizer.
    /// @param tokenizer   Trained tokenizer with vocabulary and embeddings.
    /// @param num_tokens  Number of tokens to generate per step.
    /// @param temperature Sampling temperature (lower = more deterministic).
    /// @return Generated text string.
    std::string generate_text(const Vec& probs,
                              const Vec& theta,
                              const Vec& context,
                              const Tokenizer& tokenizer,
                              size_t num_tokens = 4,
                              double temperature = 0.8);

    /// Generate an image from evolved parameters and context.
    /// @param theta    Evolved parameter vector.
    /// @param context  Context embedding.
    /// @param width    Output image width.
    /// @param height   Output image height.
    /// @return Image descriptor with RGB pixel data.
    ImageDescriptor generate_image(const Vec& theta,
                                   const Vec& context,
                                   size_t width = 64,
                                   size_t height = 64);

    /// Generate a video frame from evolved parameters and context.
    /// @param theta      Evolved parameter vector.
    /// @param context    Context embedding.
    /// @param prev_frame Previous frame (empty for first frame).
    /// @param frame_idx  Frame index in sequence.
    /// @param width      Frame width.
    /// @param height     Frame height.
    /// @return Single frame as pixel data.
    Vec generate_video_frame(const Vec& theta,
                             const Vec& context,
                             const Vec& prev_frame,
                             size_t frame_idx,
                             size_t width = 64,
                             size_t height = 64);

private:
    Config cfg_;

    /// Compute token scores from parameter embedding and vocabulary embeddings.
    Vec compute_token_scores(const Vec& param_embed,
                             const Vec& context,
                             const Tokenizer& tokenizer,
                             double temperature);

    /// Map a parameter value to a color channel value in [0, 1].
    static double param_to_color(double param, double context_val, double phase);
};

} // namespace dstt
