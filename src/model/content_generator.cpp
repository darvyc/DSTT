#include <dstt/model/content_generator.hpp>
#include <dstt/utils/math.hpp>
#include <dstt/utils/random.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <sys/stat.h>

namespace dstt {

ContentGenerator::ContentGenerator(const Config& cfg) : cfg_(cfg) {}

// ── Text Generation ─────────────────────────────────────────────────
//
// The approach mirrors LLM token sampling:
//   1. For each token t in vocabulary, compute:
//        score(t) = cos(E[t], P) * weight(context)
//      where E[t] is the learned token embedding and P is the
//      parameter embedding from ARM evaluation.
//   2. Apply temperature scaling and softmax to get a distribution.
//   3. Sample tokens from this distribution.
//
// The trained embeddings encode semantic relationships from the
// training corpus; the EA-evolved parameters guide selection toward
// tokens that are coherent and relevant to the prompt.

Vec ContentGenerator::compute_token_scores(const Vec& param_embed,
                                            const Vec& context,
                                            const Tokenizer& tokenizer,
                                            double temperature) {
    size_t vocab = tokenizer.actual_vocab_size();
    size_t dim = cfg_.embed_dim;
    const Vec& E = tokenizer.embedding_matrix();

    Vec scores(vocab, 0.0);

    // Compute similarity between each token embedding and the
    // parameter embedding, weighted by context alignment.
    double param_norm = math::norm(param_embed);
    if (param_norm < 1e-12) param_norm = 1.0;

    for (size_t t = 0; t < vocab; ++t) {
        size_t offset = t * dim;
        if (offset + dim > E.size()) break;

        // Token embedding for token t
        double dot_pe = 0.0;  // dot(E[t], param_embed)
        double dot_ce = 0.0;  // dot(E[t], context)
        double e_norm = 0.0;

        size_t edim = std::min(dim, param_embed.size());
        for (size_t d = 0; d < edim; ++d) {
            double e_d = E[offset + d];
            dot_pe += e_d * param_embed[d];
            if (d < context.size()) {
                dot_ce += e_d * context[d];
            }
            e_norm += e_d * e_d;
        }

        e_norm = std::sqrt(e_norm);
        if (e_norm < 1e-12) continue;

        // Combined score: parameter relevance + context relevance
        double sim_param = dot_pe / (e_norm * param_norm);
        double sim_ctx = (context.empty()) ? 0.0 :
            dot_ce / (e_norm * math::norm(context) + 1e-12);

        scores[t] = (0.6 * sim_param + 0.4 * sim_ctx) / std::max(temperature, 0.01);
    }

    return scores;
}

std::string ContentGenerator::generate_text(const Vec& probs,
                                             const Vec& theta,
                                             const Vec& context,
                                             const Tokenizer& tokenizer,
                                             size_t num_tokens,
                                             double temperature) {
    size_t vocab = tokenizer.actual_vocab_size();
    if (vocab == 0) return "";

    size_t edim = std::min(cfg_.embed_dim, context.size());

    // Build parameter embedding from ARM probabilities and theta
    Vec param_embed(edim, 0.0);
    for (size_t j = 0; j < theta.size() && j < probs.size(); ++j) {
        double w = probs[j] * theta[j];
        for (size_t d = 0; d < edim; ++d) {
            param_embed[d] += w * context[d % context.size()];
        }
    }

    // Score each token
    Vec scores = compute_token_scores(param_embed, context, tokenizer, temperature);
    Vec token_probs = math::softmax(scores);

    // Sample num_tokens tokens
    std::vector<uint32_t> sampled;
    sampled.reserve(num_tokens);

    for (size_t i = 0; i < num_tokens; ++i) {
        double u = RNG::uniform(0.0, 1.0);
        size_t idx = math::inverse_cdf_sample(token_probs, u);
        if (idx < vocab) {
            sampled.push_back(static_cast<uint32_t>(idx));
        }

        // Shift distribution slightly to avoid repetition:
        // reduce probability of just-sampled token
        if (idx < token_probs.size()) {
            token_probs[idx] *= 0.3;
            // Renormalise
            double sum = 0.0;
            for (double p : token_probs) sum += p;
            if (sum > 1e-12) {
                for (auto& p : token_probs) p /= sum;
            }
        }
    }

    return tokenizer.decode(sampled);
}

// ── Image Generation ────────────────────────────────────────────────
//
// Maps evolved parameter vector and context embedding to RGB pixels.
//
// For each pixel (x, y):
//   1. Compute spatial coordinates u = x/W, v = y/H in [0,1]
//   2. Index into theta using spatial hash: idx = (x*H + y) mod param_dim
//   3. RGB channels derived from theta[idx], context, and spatial phase:
//        R = sigmoid(theta[i] * context[i%dim] + phase_r)
//        G = sigmoid(theta[j] * context[j%dim] + phase_g)
//        B = sigmoid(theta[k] * context[k%dim] + phase_b)
//
// This produces deterministic images that vary with both the prompt
// (via context) and the evolutionary optimization (via theta).

double ContentGenerator::param_to_color(double param, double context_val, double phase) {
    // Sigmoid mapping with phase shift
    double x = param * context_val * 4.0 + phase;
    return 1.0 / (1.0 + std::exp(-x));
}

ImageDescriptor ContentGenerator::generate_image(const Vec& theta,
                                                  const Vec& context,
                                                  size_t width,
                                                  size_t height) {
    ImageDescriptor img;
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.pixels.resize(width * height * 3);

    size_t pd = theta.size();
    size_t cd = context.size();
    if (pd == 0 || cd == 0) return img;

    for (size_t y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / static_cast<double>(height);
        for (size_t x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / static_cast<double>(width);

            // Spatial indexing into theta and context
            size_t ti = ((x * 7 + y * 13) ^ (x * y)) % pd;
            size_t tj = ((x * 11 + y * 3 + 1) ^ (x + y * 5)) % pd;
            size_t tk = ((x * 5 + y * 17 + 2) ^ (x * 3 + y)) % pd;

            size_t ci = (ti * 3) % cd;
            size_t cj = (tj * 3 + 1) % cd;
            size_t ck = (tk * 3 + 2) % cd;

            // Phase shifts from spatial position for coherent gradients
            double phase_r = std::sin(u * 3.14159 * 2.0) * 1.5;
            double phase_g = std::cos(v * 3.14159 * 2.0) * 1.5;
            double phase_b = std::sin((u + v) * 3.14159) * 1.5;

            size_t pixel_offset = (y * width + x) * 3;
            img.pixels[pixel_offset + 0] = param_to_color(theta[ti], context[ci], phase_r);
            img.pixels[pixel_offset + 1] = param_to_color(theta[tj], context[cj], phase_g);
            img.pixels[pixel_offset + 2] = param_to_color(theta[tk], context[ck], phase_b);
        }
    }

    return img;
}

// ── Video Generation ────────────────────────────────────────────────
//
// Extends image generation with temporal variation. Each frame
// modulates the parameter-to-pixel mapping using a time variable,
// creating smooth transitions between frames.
//
//   For frame f at time t = f/total_frames:
//     theta_t[j] = theta[j] * (1 + 0.3 * sin(2π * t + j * π/param_dim))
//
// Previous frame is blended with current for temporal coherence:
//     pixel = 0.7 * current + 0.3 * previous

Vec ContentGenerator::generate_video_frame(const Vec& theta,
                                            const Vec& context,
                                            const Vec& prev_frame,
                                            size_t frame_idx,
                                            size_t width,
                                            size_t height) {
    size_t pd = theta.size();
    size_t cd = context.size();
    size_t num_pixels = width * height * 3;

    Vec frame(num_pixels, 0.5);
    if (pd == 0 || cd == 0) return frame;

    double t = static_cast<double>(frame_idx) * 0.1;

    // Temporally modulated parameters
    Vec theta_t(pd);
    for (size_t j = 0; j < pd; ++j) {
        double phase = static_cast<double>(j) * 3.14159 / static_cast<double>(pd);
        theta_t[j] = theta[j] * (1.0 + 0.3 * std::sin(2.0 * 3.14159 * t + phase));
    }

    // Generate frame using modulated parameters
    for (size_t y = 0; y < height; ++y) {
        double v = static_cast<double>(y) / static_cast<double>(height);
        for (size_t x = 0; x < width; ++x) {
            double u = static_cast<double>(x) / static_cast<double>(width);

            size_t ti = ((x * 7 + y * 13) ^ (x * y)) % pd;
            size_t tj = ((x * 11 + y * 3 + 1) ^ (x + y * 5)) % pd;
            size_t tk = ((x * 5 + y * 17 + 2) ^ (x * 3 + y)) % pd;

            size_t ci = (ti * 3) % cd;
            size_t cj = (tj * 3 + 1) % cd;
            size_t ck = (tk * 3 + 2) % cd;

            double phase_r = std::sin(u * 3.14159 * 2.0 + t) * 1.5;
            double phase_g = std::cos(v * 3.14159 * 2.0 + t * 0.7) * 1.5;
            double phase_b = std::sin((u + v) * 3.14159 + t * 1.3) * 1.5;

            size_t offset = (y * width + x) * 3;
            frame[offset + 0] = param_to_color(theta_t[ti], context[ci], phase_r);
            frame[offset + 1] = param_to_color(theta_t[tj], context[cj], phase_g);
            frame[offset + 2] = param_to_color(theta_t[tk], context[ck], phase_b);
        }
    }

    // Temporal blending with previous frame for coherence
    if (!prev_frame.empty() && prev_frame.size() == frame.size()) {
        for (size_t i = 0; i < frame.size(); ++i) {
            frame[i] = 0.7 * frame[i] + 0.3 * prev_frame[i];
        }
    }

    return frame;
}

// ── ImageDescriptor I/O ─────────────────────────────────────────────

bool ImageDescriptor::save_ppm(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    f << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < width * height * channels; ++i) {
        double v = std::max(0.0, std::min(1.0, pixels[i]));
        auto byte = static_cast<unsigned char>(v * 255.0);
        f.put(static_cast<char>(byte));
    }
    return f.good();
}

// ── VideoDescriptor I/O ─────────────────────────────────────────────

bool VideoDescriptor::save_frames(const std::string& dir_prefix) const {
    for (size_t i = 0; i < frames.size(); ++i) {
        ImageDescriptor frame_img;
        frame_img.width = width;
        frame_img.height = height;
        frame_img.channels = channels;
        frame_img.pixels = frames[i];

        std::ostringstream path;
        path << dir_prefix << "_frame_";
        if (i < 10) path << "0";
        if (i < 100) path << "0";
        path << i << ".ppm";

        if (!frame_img.save_ppm(path.str())) return false;
    }
    return true;
}

} // namespace dstt
