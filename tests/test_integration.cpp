#include <dstt/mge/mge.hpp>
#include <dstt/fdmp/fdmp.hpp>
#include <dstt/core/arm.hpp>
#include <dstt/utils/random.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dstt;

// ── FDMP tests ───────────────────────────────────────────────────────

static void test_fdmp_encode_deterministic() {
    Config cfg;
    cfg.embed_dim = 32;
    cfg.param_dim = 16;
    FDMP fdmp(cfg);

    Vec c1 = fdmp.encode("hello world", Modality::Text);
    Vec c2 = fdmp.encode("hello world", Modality::Text);
    assert(c1.size() == 32);
    for (size_t i = 0; i < c1.size(); ++i) {
        assert(std::abs(c1[i] - c2[i]) < 1e-15);
    }
    std::cout << "  [PASS] fdmp_encode_deterministic\n";
}

static void test_fdmp_different_inputs_differ() {
    Config cfg;
    cfg.embed_dim = 32;
    cfg.param_dim = 16;
    FDMP fdmp(cfg);

    Vec c1 = fdmp.encode("hello", Modality::Text);
    Vec c2 = fdmp.encode("world", Modality::Text);
    bool differ = false;
    for (size_t i = 0; i < c1.size(); ++i) {
        if (std::abs(c1[i] - c2[i]) > 1e-12) { differ = true; break; }
    }
    assert(differ);
    std::cout << "  [PASS] fdmp_different_inputs_differ\n";
}

static void test_fdmp_different_modalities_differ() {
    Config cfg;
    cfg.embed_dim = 32;
    cfg.param_dim = 16;
    FDMP fdmp(cfg);

    Vec c1 = fdmp.encode("sunset", Modality::Text);
    Vec c2 = fdmp.encode("sunset", Modality::Image);
    bool differ = false;
    for (size_t i = 0; i < c1.size(); ++i) {
        if (std::abs(c1[i] - c2[i]) > 1e-12) { differ = true; break; }
    }
    assert(differ);
    std::cout << "  [PASS] fdmp_different_modalities_differ\n";
}

static void test_fdmp_generate_params() {
    Config cfg;
    cfg.embed_dim = 16;
    cfg.param_dim = 8;
    FDMP fdmp(cfg);

    auto [ctx, theta] = fdmp.process("test input", Modality::Text);
    assert(ctx.size() == 16);
    assert(theta.size() == 8);
    std::cout << "  [PASS] fdmp_generate_params\n";
}

// ── ARM integration test ─────────────────────────────────────────────

static void test_arm_full_pass() {
    RNG::seed(42);
    Config cfg;
    cfg.param_dim = 16;
    cfg.embed_dim = 16;
    ARM arm(cfg);

    Vec theta = RNG::random_vec(cfg.param_dim, 0.0, 1.0);
    Vec ctx = RNG::random_vec(cfg.embed_dim, -1.0, 1.0);
    Vec prev(cfg.embed_dim, 0.0);

    ARMResult r = arm.evaluate(theta, ctx, prev, Modality::Text);

    assert(r.cfm_scores.size() == cfg.param_dim);
    assert(r.afm_scores.size() == cfg.param_dim);
    assert(r.probabilities.size() == cfg.param_dim);
    assert(r.sampled_idx < cfg.param_dim);

    // Probabilities sum to 1
    double sum = 0.0;
    for (double p : r.probabilities) sum += p;
    assert(std::abs(sum - 1.0) < 1e-9);

    // At least 1 partition
    assert(!r.partitions.empty());

    std::cout << "  [PASS] arm_full_pass (sampled_idx=" << r.sampled_idx
              << ", partitions=" << r.partitions.size() << ")\n";
}

// ── Branch Predictor test ────────────────────────────────────────────

static void test_branch_predictor_learns() {
    Config cfg;
    cfg.embed_dim = 8;
    BranchPredictor bp(cfg);

    Vec ctx = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Train it to predict Text for this context
    for (int i = 0; i < 100; ++i) {
        bp.update(ctx, Modality::Text);
    }

    auto [pred, conf] = bp.predict(ctx);
    assert(pred == Modality::Text);
    assert(conf > 0.5);
    std::cout << "  [PASS] branch_predictor_learns (conf=" << conf << ")\n";
}

static void test_branch_predictor_least_recent() {
    Config cfg;
    cfg.embed_dim = 4;
    BranchPredictor bp(cfg);

    bp.record_generation(Modality::Text);
    bp.record_generation(Modality::Image);
    // Video was never recorded → least recent
    Modality lr = bp.least_recent();
    assert(lr == Modality::Video);
    std::cout << "  [PASS] branch_predictor_least_recent\n";
}

// ── Synthesiser test ─────────────────────────────────────────────────

static void test_synthesiser_consistency() {
    Config cfg;
    cfg.embed_dim = 4;
    Synthesiser synth(cfg);

    // Two aligned elements → consistent
    OutputElement e1{Modality::Text, {1.0, 0.5, 0.3, 0.1}, 0, 0.8};
    OutputElement e2{Modality::Image, {1.0, 0.5, 0.3, 0.1}, 1, 0.6};
    synth.append(e1);
    synth.append(e2);
    assert(synth.check_consistency());
    assert(synth.output().count(Modality::Text) == 1);
    assert(synth.output().count(Modality::Image) == 1);
    std::cout << "  [PASS] synthesiser_consistency\n";
}

// ── MGE full pipeline test ───────────────────────────────────────────

static void test_mge_generates_output() {
    RNG::seed(42);
    Config cfg;
    cfg.param_dim = 8;
    cfg.embed_dim = 8;
    cfg.population_size = 10;
    cfg.max_generations = 10;
    cfg.tournament_k = 3;

    MGE mge(cfg);
    SynthesisedOutput out = mge.generate("A sunset over mountains", 5);

    assert(out.elements.size() == 5);
    assert(out.avg_probability() > 0.0);

    // All elements have valid data
    for (const auto& e : out.elements) {
        assert(!e.embedding.empty());
        assert(e.probability > 0.0 && e.probability <= 1.0);
    }

    std::cout << "  [PASS] mge_generates_output (elements=" << out.elements.size()
              << ", avg_prob=" << out.avg_probability() << ")\n";
}

static void test_mge_different_prompts_differ() {
    RNG::seed(42);
    Config cfg;
    cfg.param_dim = 8;
    cfg.embed_dim = 8;
    cfg.population_size = 10;
    cfg.max_generations = 5;

    MGE mge(cfg);
    SynthesisedOutput out1 = mge.generate("A sunset over mountains", 3);

    RNG::seed(42);
    MGE mge2(cfg);
    SynthesisedOutput out2 = mge2.generate("A cat sitting on a table", 3);

    // Different prompts should produce different outputs
    bool differ = false;
    for (size_t i = 0; i < 3; ++i) {
        if (out1.elements[i].param_index != out2.elements[i].param_index ||
            std::abs(out1.elements[i].probability - out2.elements[i].probability) > 1e-6) {
            differ = true;
            break;
        }
    }
    assert(differ);
    std::cout << "  [PASS] mge_different_prompts_differ\n";
}

int main() {
    std::cout << "=== Integration Tests ===\n";
    test_fdmp_encode_deterministic();
    test_fdmp_different_inputs_differ();
    test_fdmp_different_modalities_differ();
    test_fdmp_generate_params();
    test_arm_full_pass();
    test_branch_predictor_learns();
    test_branch_predictor_least_recent();
    test_synthesiser_consistency();
    test_mge_generates_output();
    test_mge_different_prompts_differ();
    std::cout << "All integration tests passed.\n";
    return 0;
}
