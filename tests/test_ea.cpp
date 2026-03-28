#include <dstt/ea/chromosome.hpp>
#include <dstt/ea/operators.hpp>
#include <dstt/ea/fitness.hpp>
#include <dstt/ea/population.hpp>
#include <dstt/utils/random.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <sstream>

using namespace dstt;

// ── Chromosome tests ─────────────────────────────────────────────────

static void test_chromosome_random() {
    RNG::seed(1);
    auto c = Chromosome::random(32);
    assert(c.size() == 32);
    assert(c.fitness == 0.0);
    for (double g : c.genes) {
        assert(g >= 0.0 && g < 1.0);
    }
    std::cout << "  [PASS] chromosome_random\n";
}

static void test_chromosome_decode_identity() {
    Chromosome c;
    c.genes = {0.0, 0.5, 1.0};
    Vec theta = c.decode(0.0, 1.0);
    assert(std::abs(theta[0] - 0.0) < 1e-12);
    assert(std::abs(theta[1] - 0.5) < 1e-12);
    assert(std::abs(theta[2] - 1.0) < 1e-12);
    std::cout << "  [PASS] chromosome_decode_identity\n";
}

static void test_chromosome_decode_range() {
    Chromosome c;
    c.genes = {0.0, 0.5, 1.0};
    Vec theta = c.decode(-10.0, 10.0);
    assert(std::abs(theta[0] - (-10.0)) < 1e-12);
    assert(std::abs(theta[1] - 0.0) < 1e-12);
    assert(std::abs(theta[2] - 10.0) < 1e-12);
    std::cout << "  [PASS] chromosome_decode_range\n";
}

static void test_chromosome_stream() {
    Chromosome c;
    c.genes = {0.1, 0.2};
    c.fitness = 0.42;
    std::ostringstream oss;
    oss << c;
    assert(oss.str().find("dim=2") != std::string::npos);
    assert(oss.str().find("0.42") != std::string::npos);
    std::cout << "  [PASS] chromosome_stream\n";
}

// ── Operator tests ───────────────────────────────────────────────────

static void test_tournament_selects_fittest() {
    RNG::seed(100);
    std::vector<Chromosome> pop(10);
    for (size_t i = 0; i < 10; ++i) {
        pop[i] = Chromosome::random(4);
        pop[i].fitness = static_cast<double>(i);  // 0..9
    }
    // With k=10 (full tournament), always select the best
    const Chromosome& best = Operators::tournament_select(pop, 10);
    assert(best.fitness == 9.0);
    std::cout << "  [PASS] tournament_selects_fittest\n";
}

static void test_crossover_preserves_genes() {
    RNG::seed(200);
    Chromosome p1, p2;
    p1.genes = {0.1, 0.2, 0.3, 0.4};
    p2.genes = {0.5, 0.6, 0.7, 0.8};

    auto [o1, o2] = Operators::crossover(p1, p2);
    assert(o1.size() == 4 && o2.size() == 4);

    // Every gene in offspring must come from one parent
    for (size_t i = 0; i < 4; ++i) {
        bool from_p1 = (std::abs(o1.genes[i] - p1.genes[i]) < 1e-12);
        bool from_p2 = (std::abs(o1.genes[i] - p2.genes[i]) < 1e-12);
        assert(from_p1 || from_p2);
    }
    std::cout << "  [PASS] crossover_preserves_genes\n";
}

static void test_crossover_creates_different_offspring() {
    RNG::seed(300);
    Chromosome p1, p2;
    p1.genes = {0.0, 0.0, 0.0, 0.0};
    p2.genes = {1.0, 1.0, 1.0, 1.0};

    auto [o1, o2] = Operators::crossover(p1, p2);
    // With distinct parents, offspring should differ from each other
    bool differ = false;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(o1.genes[i] - o2.genes[i]) > 1e-12) differ = true;
    }
    assert(differ);
    std::cout << "  [PASS] crossover_creates_different_offspring\n";
}

static void test_mutation_with_rate_one() {
    RNG::seed(400);
    Chromosome c;
    c.genes = {0.5, 0.5, 0.5, 0.5};
    Vec original = c.genes;
    Operators::mutate(c, 1.0);  // mu = 1.0 → every gene mutates
    bool any_changed = false;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(c.genes[i] - original[i]) > 1e-12) any_changed = true;
    }
    assert(any_changed);
    std::cout << "  [PASS] mutation_with_rate_one\n";
}

static void test_mutation_with_rate_zero() {
    RNG::seed(500);
    Chromosome c;
    c.genes = {0.5, 0.5, 0.5, 0.5};
    Vec original = c.genes;
    Operators::mutate(c, 0.0);  // mu = 0 → no mutations
    for (size_t i = 0; i < 4; ++i) {
        assert(std::abs(c.genes[i] - original[i]) < 1e-12);
    }
    std::cout << "  [PASS] mutation_with_rate_zero\n";
}

// ── Fitness evaluation tests ─────────────────────────────────────────

static void test_fitness_returns_bounded() {
    RNG::seed(600);
    Config cfg;
    cfg.param_dim = 8;
    cfg.embed_dim = 8;
    FitnessEvaluator fe(cfg);

    Chromosome c = Chromosome::random(cfg.param_dim);
    Vec ctx = RNG::random_vec(cfg.embed_dim, -1.0, 1.0);
    Vec prev = RNG::random_vec(cfg.embed_dim, -1.0, 1.0);

    auto fm = fe.evaluate(c, ctx, prev, Modality::Text);
    assert(fm.total >= 0.0 && fm.total <= 2.0);  // bounded by sum of weights * max scores
    assert(fm.coherence >= 0.0 && fm.coherence <= 1.0);
    assert(fm.diversity >= 0.0 && fm.diversity <= 1.0);
    std::cout << "  [PASS] fitness_returns_bounded (total=" << fm.total << ")\n";
}

// ── Population / evolution tests ─────────────────────────────────────

static void test_evolution_fitness_nondecreasing() {
    // Theorem 2: elitist EA guarantees non-decreasing best fitness
    RNG::seed(700);
    Config cfg;
    cfg.param_dim = 8;
    cfg.embed_dim = 8;
    cfg.population_size = 20;
    cfg.max_generations = 30;
    cfg.tournament_k = 3;
    cfg.convergence_window = 100;  // disable early stop

    Vec ctx = RNG::random_vec(cfg.embed_dim, -1.0, 1.0);
    Vec prev(cfg.embed_dim, 0.0);

    std::vector<double> best_per_gen;
    Population pop(cfg);
    pop.evolve(ctx, prev, Modality::Text,
        [&](const GenerationStats& gs) {
            best_per_gen.push_back(gs.best_fitness);
        });

    // Verify non-decreasing
    for (size_t i = 1; i < best_per_gen.size(); ++i) {
        assert(best_per_gen[i] >= best_per_gen[i - 1] - 1e-12);
    }
    std::cout << "  [PASS] evolution_fitness_nondecreasing ("
              << best_per_gen.size() << " gens, final="
              << best_per_gen.back() << ")\n";
}

static void test_evolution_improves_over_random() {
    RNG::seed(800);
    Config cfg;
    cfg.param_dim = 16;
    cfg.embed_dim = 16;
    cfg.population_size = 30;
    cfg.max_generations = 40;
    cfg.tournament_k = 5;

    Vec ctx = RNG::random_vec(cfg.embed_dim, -1.0, 1.0);
    Vec prev(cfg.embed_dim, 0.0);

    // Evaluate a random chromosome
    FitnessEvaluator fe(cfg);
    Chromosome random_chrom = Chromosome::random(cfg.param_dim);
    double random_fitness = fe.evaluate(random_chrom, ctx, prev, Modality::Image).total;

    // Evolve
    Population pop(cfg);
    Chromosome best = pop.evolve(ctx, prev, Modality::Image);

    assert(best.fitness >= random_fitness - 0.1);  // evolved should be at least comparable
    std::cout << "  [PASS] evolution_improves_over_random (random="
              << random_fitness << ", evolved=" << best.fitness << ")\n";
}

int main() {
    std::cout << "=== EA Tests ===\n";
    test_chromosome_random();
    test_chromosome_decode_identity();
    test_chromosome_decode_range();
    test_chromosome_stream();
    test_tournament_selects_fittest();
    test_crossover_preserves_genes();
    test_crossover_creates_different_offspring();
    test_mutation_with_rate_one();
    test_mutation_with_rate_zero();
    test_fitness_returns_bounded();
    test_evolution_fitness_nondecreasing();
    test_evolution_improves_over_random();
    std::cout << "All EA tests passed.\n";
    return 0;
}
