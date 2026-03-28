#include <dstt/fdmp/fdmp.hpp>
#include <dstt/fdmp/embeddings.hpp>
#include <dstt/utils/math.hpp>
#include <dstt/utils/random.hpp>
#include <cmath>

namespace dstt {

FDMP::FDMP(const Config& cfg) : cfg_(cfg) {
    init_weights();
}

void FDMP::init_weights() {
    size_t pd = cfg_.param_dim;
    size_t ed = cfg_.embed_dim;
    size_t total = pd * ed;

    // Xavier initialisation: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
    double limit = std::sqrt(6.0 / static_cast<double>(pd + ed));

    // Save and restore RNG state so init is deterministic per modality
    auto make_w = [&](uint64_t salt) {
        RNG::seed(salt);
        return RNG::random_vec(total, -limit, limit);
    };
    auto make_b = [&](uint64_t salt) {
        RNG::seed(salt + 0x100);
        return RNG::random_vec(pd, -0.01, 0.01);
    };

    W_text_  = make_w(0xA001);  b_text_  = make_b(0xA001);
    W_image_ = make_w(0xA002);  b_image_ = make_b(0xA002);
    W_video_ = make_w(0xA003);  b_video_ = make_b(0xA003);

    // Re-seed with something non-deterministic-ish for runtime
    RNG::seed(42);
}

const Vec& FDMP::weight_matrix(Modality m) const {
    switch (m) {
        case Modality::Text:  return W_text_;
        case Modality::Image: return W_image_;
        case Modality::Video: return W_video_;
    }
    return W_text_;
}

const Vec& FDMP::bias_vector(Modality m) const {
    switch (m) {
        case Modality::Text:  return b_text_;
        case Modality::Image: return b_image_;
        case Modality::Video: return b_video_;
    }
    return b_text_;
}

Vec FDMP::encode(const std::string& input, Modality m) const {
    return Embeddings::embed(input, m, cfg_.embed_dim);
}

Vec FDMP::generate_params(const Vec& context, Modality m) const {
    return math::affine(weight_matrix(m), cfg_.param_dim, cfg_.embed_dim,
                        context, bias_vector(m));
}

std::pair<Vec, Vec> FDMP::process(const std::string& input, Modality m) const {
    Vec ctx = encode(input, m);
    Vec theta = generate_params(ctx, m);
    return {std::move(ctx), std::move(theta)};
}

} // namespace dstt
