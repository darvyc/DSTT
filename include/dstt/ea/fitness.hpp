#pragma once

#include <dstt/core/types.hpp>
#include <dstt/ea/chromosome.hpp>
#include <dstt/core/arm.hpp>

namespace dstt {

/// Fitness metrics for a generated output.
struct FitnessMetrics {
    double coherence = 0.0;
    double relevance = 0.0;
    double diversity = 0.0;
    double total     = 0.0;
};

/// Evaluates chromosome fitness within the DSTT framework.
///
/// F(p, m) = w_c · Coherence + w_r · Relevance + w_d · Diversity
class FitnessEvaluator {
public:
    explicit FitnessEvaluator(const Config& cfg);

    /// Evaluate a chromosome by decoding it into parameters, running an
    /// ARM pass, and scoring the resulting probability distribution.
    ///
    /// @param chrom    The chromosome to evaluate.
    /// @param context  Context embedding.
    /// @param prev_state Previous state embedding.
    /// @param modality Modality to evaluate under.
    /// @return Fitness metrics (total is the scalar fitness).
    FitnessMetrics evaluate(const Chromosome& chrom,
                            const Vec& context,
                            const Vec& prev_state,
                            Modality modality) const;

private:
    Config cfg_;
    ARM arm_;

    /// Coherence: average probability mass on top-k parameters that
    /// have positive CFM−AFM (i.e., coherent parameters dominate).
    double measure_coherence(const ARMResult& r) const;

    /// Relevance: cosine similarity between the probability-weighted
    /// parameter vector and the context.
    double measure_relevance(const ARMResult& r, const Vec& theta,
                             const Vec& context) const;

    /// Diversity: entropy of the output distribution (higher = more diverse).
    double measure_diversity(const ARMResult& r) const;
};

} // namespace dstt
