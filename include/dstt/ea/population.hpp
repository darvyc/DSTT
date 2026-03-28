#pragma once

#include <dstt/core/types.hpp>
#include <dstt/ea/chromosome.hpp>
#include <dstt/ea/fitness.hpp>
#include <functional>
#include <vector>

namespace dstt {

/// Per-generation statistics.
struct GenerationStats {
    size_t generation = 0;
    double best_fitness = 0.0;
    double avg_fitness  = 0.0;
    double worst_fitness = 0.0;
};

/// Callback invoked after each generation.
using GenerationCallback = std::function<void(const GenerationStats&)>;

/// Population manager and generational loop controller.
///
/// Implements Algorithm 1 from the DSTT specification:
///   1. Initialise population
///   2. For each generation:
///        evaluate → sort → elite → tournament → crossover → mutate → replace
///   3. Return best chromosome
class Population {
public:
    explicit Population(const Config& cfg);

    /// Run the full evolutionary loop.
    /// @param context     Context embedding used for fitness evaluation.
    /// @param prev_state  Previous state embedding.
    /// @param modality    Modality to optimise for.
    /// @param on_gen      Optional per-generation callback.
    /// @return Best chromosome found.
    Chromosome evolve(const Vec& context,
                      const Vec& prev_state,
                      Modality modality,
                      GenerationCallback on_gen = nullptr);

    /// Access the current population (read-only).
    const std::vector<Chromosome>& individuals() const { return pop_; }

private:
    Config cfg_;
    std::vector<Chromosome> pop_;

    void initialise();
    void evaluate_all(const FitnessEvaluator& fe,
                      const Vec& context,
                      const Vec& prev_state,
                      Modality modality);
    void next_generation();
    bool converged(const std::vector<double>& history) const;
};

} // namespace dstt
