#include <dstt/core/cfm.hpp>
#include <dstt/utils/math.hpp>

namespace dstt {

double CFM::alpha(Modality m) {
    switch (m) {
        case Modality::Text:  return 0.85;
        case Modality::Image: return 0.80;
        case Modality::Video: return 0.75;
    }
    return 0.80;
}

double CFM::score(double /*theta_j*/,
                  const Vec& param_embed,
                  const Vec& context_embed,
                  const Vec& prev_state,
                  Modality modality) {
    // Wittgenstein Score: cosine similarity × α_m
    double ws = math::cosine_similarity(param_embed, context_embed) * alpha(modality);

    // Ramsey Coherence: 1 − d_H(param, prev_state) / dim
    size_t dh = math::hamming_distance(param_embed, prev_state);
    double rc = 1.0 - static_cast<double>(dh) / static_cast<double>(param_embed.size());

    return ws + rc;
}

} // namespace dstt
