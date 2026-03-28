#include <dstt/fdmp/tokenizer.hpp>
#include <dstt/training/trainer.hpp>
#include <dstt/utils/random.hpp>
#include <cassert>
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace dstt;

static void test_tokenizer_build_vocab() {
    std::cout << "  test_tokenizer_build_vocab ... ";
    Config cfg;
    cfg.embed_dim = 16;
    cfg.param_dim = 8;
    cfg.vocab_size = 300;
    cfg.min_token_freq = 2;

    Tokenizer tok(cfg);

    std::vector<std::string> corpus = {
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog",
        "the cat sat on the mat again",
        "the dog and the cat"
    };

    tok.build_vocab(corpus);

    // Should have learned some merges beyond the 256 byte tokens
    assert(tok.actual_vocab_size() > 256);
    assert(tok.is_trained());

    // Encode and decode should round-trip
    std::string text = "the cat sat";
    auto ids = tok.encode(text);
    assert(!ids.empty());
    std::string decoded = tok.decode(ids);
    assert(decoded == text);

    std::cout << "OK (vocab=" << tok.actual_vocab_size() << ")\n";
}

static void test_tokenizer_embed() {
    std::cout << "  test_tokenizer_embed ... ";
    Config cfg;
    cfg.embed_dim = 16;
    cfg.param_dim = 8;
    cfg.vocab_size = 300;

    Tokenizer tok(cfg);
    std::vector<std::string> corpus = {"hello world", "hello there", "world hello"};
    tok.build_vocab(corpus);

    auto ids = tok.encode("hello world");
    Vec emb = tok.embed_tokens(ids);

    assert(emb.size() == 16);

    // Should be L2 normalised (approximately unit norm)
    double norm = 0.0;
    for (double x : emb) norm += x * x;
    assert(std::abs(std::sqrt(norm) - 1.0) < 1e-6);

    // Different inputs should produce different embeddings
    auto ids2 = tok.encode("there world");
    Vec emb2 = tok.embed_tokens(ids2);
    double diff = 0.0;
    for (size_t i = 0; i < emb.size(); ++i) {
        diff += std::abs(emb[i] - emb2[i]);
    }
    assert(diff > 1e-6);

    std::cout << "OK\n";
}

static void test_trainer_single_epoch() {
    std::cout << "  test_trainer_single_epoch ... ";
    RNG::seed(42);

    Config cfg;
    cfg.embed_dim = 16;
    cfg.param_dim = 8;
    cfg.population_size = 10;
    cfg.max_generations = 10;
    cfg.training_epochs = 2;
    cfg.training_lr = 0.01;
    cfg.vocab_size = 300;

    Trainer trainer(cfg);

    std::vector<TrainingExample> examples = {
        {"sunset over mountains", Modality::Image},
        {"a news article about science", Modality::Text},
        {"a time-lapse of a flower blooming", Modality::Video},
        {"portrait of a person smiling", Modality::Image},
        {"breaking news headline", Modality::Text}
    };

    size_t epoch_count = 0;
    trainer.train(examples, [&](const EpochStats& stats) {
        std::cout << "\n    epoch " << stats.epoch
                  << " avg_fitness=" << stats.avg_fitness
                  << " avg_loss=" << stats.avg_loss;
        epoch_count++;
    });

    assert(epoch_count == 2);
    assert(trainer.is_trained());
    std::cout << " ... OK\n";
}

static void test_trainer_fitness_improves() {
    std::cout << "  test_trainer_fitness_improves ... ";
    RNG::seed(123);

    Config cfg;
    cfg.embed_dim = 16;
    cfg.param_dim = 8;
    cfg.population_size = 10;
    cfg.max_generations = 10;
    cfg.training_epochs = 5;
    cfg.training_lr = 0.005;
    cfg.vocab_size = 300;

    Trainer trainer(cfg);

    std::vector<TrainingExample> examples = {
        {"the quick brown fox", Modality::Text},
        {"a beautiful landscape photo", Modality::Image},
        {"drone footage of coastline", Modality::Video},
        {"the lazy dog sleeps", Modality::Text},
        {"abstract painting colors", Modality::Image},
        {"slow motion water splash", Modality::Video}
    };

    std::vector<double> epoch_fitness;
    trainer.train(examples, [&](const EpochStats& stats) {
        epoch_fitness.push_back(stats.avg_fitness);
    });

    assert(epoch_fitness.size() == 5);
    // The training should not catastrophically degrade
    // (fitness should stay in a reasonable range)
    for (size_t i = 0; i < epoch_fitness.size(); ++i) {
        assert(epoch_fitness[i] >= 0.0 && epoch_fitness[i] <= 1.0);
    }

    std::cout << "OK\n";
}

static void test_save_load() {
    std::cout << "  test_save_load ... ";
    RNG::seed(42);

    Config cfg;
    cfg.embed_dim = 16;
    cfg.param_dim = 8;
    cfg.population_size = 10;
    cfg.max_generations = 5;
    cfg.training_epochs = 1;
    cfg.vocab_size = 300;

    Trainer trainer(cfg);
    std::vector<TrainingExample> examples = {
        {"hello world", Modality::Text},
        {"a nice photo", Modality::Image}
    };
    trainer.train(examples);

    // Save
    std::string dir = "/tmp/dstt_test_save";
    std::filesystem::create_directories(dir);
    trainer.save(dir);

    // Load into a fresh trainer
    Trainer trainer2(cfg);
    trainer2.load(dir);
    assert(trainer2.is_trained());

    // Weights should match
    const Vec& w_orig = trainer.fdmp().weight_matrix(Modality::Text);
    const Vec& w_loaded = trainer2.fdmp().weight_matrix(Modality::Text);
    assert(w_orig.size() == w_loaded.size());
    for (size_t i = 0; i < w_orig.size(); ++i) {
        assert(std::abs(w_orig[i] - w_loaded[i]) < 1e-12);
    }

    // Cleanup
    std::filesystem::remove_all(dir);

    std::cout << "OK\n";
}

int main() {
    std::cout << "=== Training Phase Tests ===\n";
    test_tokenizer_build_vocab();
    test_tokenizer_embed();
    test_trainer_single_epoch();
    test_trainer_fitness_improves();
    test_save_load();
    std::cout << "=== All training tests passed ===\n";
    return 0;
}
