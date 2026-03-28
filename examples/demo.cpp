#include <dstt/mge/mge.hpp>
#include <dstt/training/trainer.hpp>
#include <dstt/utils/random.hpp>
#include <dstt/utils/timer.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace dstt;

static void print_separator() {
    std::cout << std::string(72, '-') << "\n";
}

static void run_demo(const std::string& prompt, const Config& cfg) {
    std::cout << "\n";
    print_separator();
    std::cout << "  DSTT Demo — \"" << prompt << "\"\n";
    print_separator();

    MGE mge(cfg);

    std::cout << "\n[Phase 2] Evolving parameters per modality...\n";

    size_t total_gens = 0;
    double final_best = 0.0;

    SynthesisedOutput output;
    {
        ScopedTimer timer("Total generation");
        output = mge.generate(prompt, 12,
            [&](const GenerationStats& gs) {
                total_gens = gs.generation + 1;
                final_best = gs.best_fitness;
                if (gs.generation % 5 == 0 || gs.generation < 3) {
                    std::cout << "  Gen " << std::setw(3) << gs.generation
                              << "  best=" << std::fixed << std::setprecision(6)
                              << gs.best_fitness
                              << "  avg=" << gs.avg_fitness << "\n";
                }
            });
    }

    std::cout << "\n[Phase 3] Generating multi-modal output ("
              << output.elements.size() << " elements)\n\n";

    print_separator();
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(10) << "Modality"
              << std::setw(10) << "Param#"
              << std::setw(14) << "Probability"
              << "Embedding[0..3]\n";
    print_separator();

    for (size_t i = 0; i < output.elements.size(); ++i) {
        const auto& e = output.elements[i];
        std::cout << std::setw(6) << i
                  << std::setw(10) << modality_name(e.modality)
                  << std::setw(10) << e.param_index
                  << std::fixed << std::setprecision(6)
                  << std::setw(14) << e.probability
                  << "  [";
        size_t show = std::min(e.embedding.size(), size_t(4));
        for (size_t d = 0; d < show; ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << std::setprecision(4) << e.embedding[d];
        }
        if (e.embedding.size() > 4) std::cout << ", ...";
        std::cout << "]\n";
    }

    print_separator();

    // Summary stats
    std::cout << "\nSummary:\n";
    std::cout << "  Total elements:     " << output.elements.size() << "\n";
    std::cout << "  Text elements:      " << output.count(Modality::Text) << "\n";
    std::cout << "  Image elements:     " << output.count(Modality::Image) << "\n";
    std::cout << "  Video elements:     " << output.count(Modality::Video) << "\n";
    std::cout << "  Avg probability:    " << std::fixed << std::setprecision(6)
              << output.avg_probability() << "\n";
    std::cout << "  EA generations:     " << total_gens << "\n";
    std::cout << "  Best EA fitness:    " << final_best << "\n";
    print_separator();
}

int main() {
    std::cout << R"(
    ____  ______________
   / __ \/ ___/_  __/_  __/
  / / / /\__ \ / /   / /
 / /_/ /___/ // /   / /
/_____//____//_/   /_/

Dynamic Semi-Trained Topology v2.0
)" << std::endl;

    RNG::seed(42);

    // ── Configuration ───────────────────────────────────────────────
    Config cfg;
    cfg.param_dim       = 32;
    cfg.embed_dim       = 32;
    cfg.population_size = 30;
    cfg.max_generations = 40;
    cfg.tournament_k    = 5;
    cfg.elitism_rate    = 0.10;
    cfg.w_coherence     = 0.4;
    cfg.w_relevance     = 0.4;
    cfg.w_diversity     = 0.2;
    cfg.coherence_threshold = 0.25;
    cfg.bp_confidence_threshold = 0.6;

    // Training config
    cfg.training_epochs = 3;
    cfg.training_lr     = 0.005;
    cfg.vocab_size      = 512;

    // ── Phase 1: Training ───────────────────────────────────────────
    print_separator();
    std::cout << "  [Phase 1] Training FDMP weights on example corpus\n";
    print_separator();

    std::vector<TrainingExample> training_data = {
        {"A sunset over snow-capped mountains",           Modality::Image},
        {"Breaking news: earthquake hits coastal city",   Modality::Text},
        {"Time-lapse of a flower blooming in spring",     Modality::Video},
        {"Portrait photography with natural lighting",    Modality::Image},
        {"Scientific paper on quantum entanglement",      Modality::Text},
        {"Drone footage of ocean waves crashing",         Modality::Video},
        {"Abstract oil painting with vivid colors",       Modality::Image},
        {"A short story about a wandering knight",        Modality::Text},
        {"Slow-motion capture of a hummingbird",          Modality::Video},
        {"Architectural photograph of a modern building", Modality::Image},
        {"Technical documentation for an API",            Modality::Text},
        {"Cinematic trailer for a fantasy film",          Modality::Video},
    };

    Trainer trainer(cfg);
    {
        ScopedTimer timer("Training phase");
        trainer.train(training_data, [](const EpochStats& stats) {
            std::cout << "  Epoch " << std::setw(2) << stats.epoch
                      << "  avg_fitness=" << std::fixed << std::setprecision(6)
                      << stats.avg_fitness
                      << "  avg_loss=" << stats.avg_loss
                      << "  best=" << stats.best_fitness << "\n";
        });
    }

    std::cout << "\n  Tokenizer vocabulary: "
              << trainer.tokenizer().actual_vocab_size() << " tokens\n";
    std::cout << "  Training complete.\n\n";

    // ── Phase 2 & 3: Inference ──────────────────────────────────────
    run_demo("A sunset over snow-capped mountains", cfg);
    run_demo("Neural network training convergence analysis", cfg);
    run_demo("An orchestra performing Beethoven's Fifth Symphony", cfg);

    std::cout << "\nAll demos completed successfully.\n";
    return 0;
}
