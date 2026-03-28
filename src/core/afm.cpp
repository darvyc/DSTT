#include <dstt/core/afm.hpp>
#include <dstt/utils/math.hpp>
#include <algorithm>

namespace dstt {

double AFM::score(double raw_prob,
                  const Vec& param_embed,
                  const Vec& context_embed,
                  Modality /*modality*/) {
    // Contradiction Score: max(0, −cosine_similarity)
    double ca = std::max(0.0, -math::cosine_similarity(param_embed, context_embed));

    // Entropy Score: Shannon self-information
    double es = math::self_information(raw_prob);

    return ca + es;
}

} // namespace dstt
