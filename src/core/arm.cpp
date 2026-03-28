#include <dstt/core/arm.hpp>
#include <dstt/core/cfm.hpp>
#include <dstt/core/afm.hpp>
#include <dstt/core/softmax.hpp>
#include <dstt/utils/math.hpp>

namespace dstt {

ARM::ARM(const Config& cfg) : cfg_(cfg) {}

/// Build a simple embedding for parameter θ_j given context.
static Vec make_param_embed(double theta_val, const Vec& context, size_t dim) {
    Vec e(dim);
    for (size_t d = 0; d < dim; ++d) {
        size_t ci = d % context.size();
        e[d] = theta_val * context[ci];
    }
    return e;
}

ARMResult ARM::evaluate(const Vec& theta,
                        const Vec& context,
                        const Vec& prev_state,
                        Modality modality) const {
    size_t m = theta.size();
    size_t edim = std::min(cfg_.embed_dim, context.size());

    // 1. Partition
    PartitionSet parts = PartitionEngine::partition(theta, context, modality, cfg_);

    // 2. Pre-compute uniform raw probabilities for AFM entropy term
    double raw_prob = 1.0 / static_cast<double>(m);

    // 3. Compute per-parameter embeddings, CFM, AFM
    Vec cfm_scores(m);
    Vec afm_scores(m);

    // Ensure prev_state has the right dimension
    Vec ps = prev_state;
    if (ps.size() < edim) {
        ps.resize(edim, 0.0);
    }

    for (size_t j = 0; j < m; ++j) {
        Vec pe = make_param_embed(theta[j], context, edim);
        Vec ce(context.begin(), context.begin() + static_cast<long>(edim));

        cfm_scores[j] = CFM::score(theta[j], pe, ce, ps, modality);
        afm_scores[j] = AFM::score(raw_prob, pe, ce, modality);
    }

    // 4. Adjust and sample
    auto [idx, probs] = AdjustAndSample::run(theta, cfm_scores, afm_scores);

    Vec adjusted = AdjustAndSample::adjust(theta, cfm_scores, afm_scores);

    return ARMResult{
        std::move(cfm_scores),
        std::move(afm_scores),
        std::move(adjusted),
        std::move(probs),
        idx,
        std::move(parts)
    };
}

} // namespace dstt
