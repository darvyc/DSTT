#include <dstt/ea/population.hpp>
#include <dstt/ea/operators.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace dstt {

Population::Population(const Config& cfg) : cfg_(cfg) {}

void Population::initialise() {
    pop_.clear();
    pop_.reserve(cfg_.population_size);
    for (size_t i = 0; i < cfg_.population_size; ++i) {
        pop_.push_back(Chromosome::random(cfg_.param_dim));
    }
}

void Population::evaluate_all(const FitnessEvaluator& fe,
                              const Vec& context,
                              const Vec& prev_state,
                              Modality modality) {
    for (auto& c : pop_) {
        auto fm = fe.evaluate(c, context, prev_state, modality);
        c.fitness = fm.total;
    }
}

void Population::next_generation() {
    // Sort descending by fitness
    std::sort(pop_.begin(), pop_.end(),
              [](const Chromosome& a, const Chromosome& b) {
                  return a.fitness > b.fitness;
              });

    size_t N = cfg_.population_size;
    size_t n_elite = std::max<size_t>(1, static_cast<size_t>(
        cfg_.elitism_rate * static_cast<double>(N)));

    std::vector<Chromosome> next_pop;
    next_pop.reserve(N);

    // Elitism: copy top individuals
    for (size_t i = 0; i < n_elite && i < pop_.size(); ++i) {
        next_pop.push_back(pop_[i]);
    }

    // Fill remaining slots via tournament → crossover → mutation
    double mu = cfg_.effective_mutation_rate();
    while (next_pop.size() < N) {
        const Chromosome& p1 = Operators::tournament_select(pop_, cfg_.tournament_k);
        const Chromosome& p2 = Operators::tournament_select(pop_, cfg_.tournament_k);
        auto [o1, o2] = Operators::crossover(p1, p2);
        Operators::mutate(o1, mu);
        Operators::mutate(o2, mu);
        next_pop.push_back(std::move(o1));
        if (next_pop.size() < N) {
            next_pop.push_back(std::move(o2));
        }
    }

    pop_ = std::move(next_pop);
}

bool Population::converged(const std::vector<double>& history) const {
    if (history.size() < cfg_.convergence_window) return false;
    size_t n = history.size();
    double recent_best = history[n - 1];
    double window_start = history[n - cfg_.convergence_window];
    return std::abs(recent_best - window_start) < cfg_.convergence_eps;
}

Chromosome Population::evolve(const Vec& context,
                               const Vec& prev_state,
                               Modality modality,
                               GenerationCallback on_gen) {
    initialise();
    FitnessEvaluator fe(cfg_);

    std::vector<double> best_history;
    best_history.reserve(cfg_.max_generations);

    for (size_t g = 0; g < cfg_.max_generations; ++g) {
        evaluate_all(fe, context, prev_state, modality);

        // Sort for stats
        std::sort(pop_.begin(), pop_.end(),
                  [](const Chromosome& a, const Chromosome& b) {
                      return a.fitness > b.fitness;
                  });

        double best_f = pop_.front().fitness;
        double worst_f = pop_.back().fitness;
        double sum_f = 0.0;
        for (const auto& c : pop_) sum_f += c.fitness;
        double avg_f = sum_f / static_cast<double>(pop_.size());

        best_history.push_back(best_f);

        if (on_gen) {
            on_gen(GenerationStats{g, best_f, avg_f, worst_f});
        }

        // Convergence check
        if (converged(best_history)) break;

        // Produce next generation
        next_generation();
    }

    // Final evaluation to ensure fitness is up-to-date
    evaluate_all(fe, context, prev_state, modality);
    std::sort(pop_.begin(), pop_.end(),
              [](const Chromosome& a, const Chromosome& b) {
                  return a.fitness > b.fitness;
              });

    return pop_.front();
}

} // namespace dstt
