#include <dstt/ea/fitness.hpp>
#include <dstt/utils/math.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace dstt {

FitnessEvaluator::FitnessEvaluator(const Config& cfg) : cfg_(cfg), arm_(cfg) {}

double FitnessEvaluator::measure_coherence(const ARMResult& r) const {
    // Fraction of probability mass on parameters where CFM > AFM
    double coherent_mass = 0.0;
    for (size_t j = 0; j < r.cfm_scores.size(); ++j) {
        if (r.cfm_scores[j] > r.afm_scores[j]) {
            coherent_mass += r.probabilities[j];
        }
    }
    return coherent_mass;  // in [0, 1]
}

double FitnessEvaluator::measure_relevance(const ARMResult& r,
                                           const Vec& theta,
                                           const Vec& context) const {
    // Probability-weighted parameter vector
    size_t m = theta.size();
    size_t dim = std::min(cfg_.embed_dim, context.size());
    Vec weighted(dim, 0.0);
    for (size_t j = 0; j < m; ++j) {
        for (size_t d = 0; d < dim; ++d) {
            size_t ci = d % context.size();
            weighted[d] += r.probabilities[j] * theta[j] * context[ci];
        }
    }
    Vec ctx_slice(context.begin(), context.begin() + static_cast<long>(dim));
    double cs = math::cosine_similarity(weighted, ctx_slice);
    return (cs + 1.0) / 2.0;  // map [-1,1] → [0,1]
}

double FitnessEvaluator::measure_diversity(const ARMResult& r) const {
    // Normalised Shannon entropy of the probability distribution
    size_t m = r.probabilities.size();
    if (m <= 1) return 0.0;
    double H = 0.0;
    for (double p : r.probabilities) {
        if (p > 1e-15) H -= p * std::log2(p);
    }
    double H_max = std::log2(static_cast<double>(m));
    return (H_max > 0.0) ? H / H_max : 0.0;  // in [0, 1]
}

FitnessMetrics FitnessEvaluator::evaluate(const Chromosome& chrom,
                                          const Vec& context,
                                          const Vec& prev_state,
                                          Modality modality) const {
    Vec theta = chrom.decode();
    ARMResult r = arm_.evaluate(theta, context, prev_state, modality);

    FitnessMetrics fm;
    fm.coherence = measure_coherence(r);
    fm.relevance = measure_relevance(r, theta, context);
    fm.diversity = measure_diversity(r);
    fm.total = cfg_.w_coherence * fm.coherence
             + cfg_.w_relevance * fm.relevance
             + cfg_.w_diversity * fm.diversity;
    return fm;
}

} // namespace dstt
