#include <dstt/core/softmax.hpp>
#include <dstt/utils/math.hpp>
#include <dstt/utils/random.hpp>

namespace dstt {

Vec AdjustAndSample::adjust(const Vec& theta, const Vec& cfm, const Vec& afm) {
    assert(theta.size() == cfm.size() && cfm.size() == afm.size());
    Vec adjusted(theta.size());
    for (size_t j = 0; j < theta.size(); ++j) {
        adjusted[j] = theta[j] * (cfm[j] - afm[j]);
    }
    return adjusted;
}

size_t AdjustAndSample::sample(const Vec& probs) {
    double u = RNG::uniform(0.0, 1.0);
    return math::inverse_cdf_sample(probs, u);
}

std::pair<size_t, Vec> AdjustAndSample::run(const Vec& theta,
                                             const Vec& cfm,
                                             const Vec& afm) {
    Vec adjusted = adjust(theta, cfm, afm);
    Vec probs = math::softmax(adjusted);
    size_t idx = sample(probs);
    return {idx, std::move(probs)};
}

} // namespace dstt
