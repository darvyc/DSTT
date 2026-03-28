#include <dstt/core/cfm.hpp>
#include <dstt/core/afm.hpp>
#include <dstt/core/softmax.hpp>
#include <dstt/utils/math.hpp>
#include <dstt/utils/random.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dstt;

static constexpr double EPS = 1e-9;

// ── CFM tests ────────────────────────────────────────────────────────

static void test_cfm_identical_embeddings() {
    // When param embedding == context embedding, cosine = 1 → Ws = α_m.
    // When param embedding == prev_state, Hamming = 0 → Rc = 1.
    Vec embed = {1.0, 0.0, 0.0, 1.0};
    double s = CFM::score(0.5, embed, embed, embed, Modality::Text);
    double expected = CFM::alpha(Modality::Text) + 1.0;  // Ws + Rc
    assert(std::abs(s - expected) < EPS);
    std::cout << "  [PASS] cfm_identical_embeddings (score=" << s << ")\n";
}

static void test_cfm_orthogonal_embeddings() {
    // Orthogonal → cosine = 0 → Ws = 0.
    Vec a = {1.0, 0.0, 0.0, 0.0};
    Vec b = {0.0, 1.0, 0.0, 0.0};
    Vec prev = {1.0, 0.0, 0.0, 0.0};
    double s = CFM::score(0.5, a, b, prev, Modality::Image);
    // Ws = 0, Rc = 1 (a == prev)
    assert(std::abs(s - 1.0) < EPS);
    std::cout << "  [PASS] cfm_orthogonal_embeddings (score=" << s << ")\n";
}

static void test_cfm_modality_weights_differ() {
    double a_text  = CFM::alpha(Modality::Text);
    double a_image = CFM::alpha(Modality::Image);
    double a_video = CFM::alpha(Modality::Video);
    assert(a_text > a_video);  // text is weighted highest
    assert(a_image > a_video);
    std::cout << "  [PASS] cfm_modality_weights_differ"
              << " (text=" << a_text << " image=" << a_image
              << " video=" << a_video << ")\n";
}

// ── AFM tests ────────────────────────────────────────────────────────

static void test_afm_aligned_no_contradiction() {
    // Positively correlated embedding → Ca = max(0, -cos) = 0
    Vec embed = {1.0, 1.0, 1.0, 1.0};
    double s = AFM::score(0.25, embed, embed, Modality::Text);
    // Ca = 0, Es = -0.25 * log2(0.25) = 0.5
    double expected_es = -0.25 * std::log2(0.25);
    assert(std::abs(s - expected_es) < EPS);
    std::cout << "  [PASS] afm_aligned_no_contradiction (score=" << s << ")\n";
}

static void test_afm_opposite_high_contradiction() {
    // Opposite embeddings → cos = -1 → Ca = 1
    Vec a = {1.0, 0.0};
    Vec b = {-1.0, 0.0};
    double s = AFM::score(0.5, a, b, Modality::Text);
    // Ca = 1.0, Es = -0.5 * log2(0.5) = 0.5
    double expected = 1.0 + 0.5;
    assert(std::abs(s - expected) < EPS);
    std::cout << "  [PASS] afm_opposite_high_contradiction (score=" << s << ")\n";
}

static void test_afm_zero_prob_no_entropy() {
    Vec embed = {1.0, 0.0};
    double s = AFM::score(0.0, embed, embed, Modality::Image);
    // Ca = 0, Es = 0 (self_information(0) = 0)
    assert(std::abs(s) < EPS);
    std::cout << "  [PASS] afm_zero_prob_no_entropy\n";
}

// ── Adjustment pipeline tests ────────────────────────────────────────

static void test_adjust_promotes_coherent() {
    // Parameter with CFM > AFM should get higher adjusted score
    Vec theta = {0.5, 0.5};
    Vec cfm   = {2.0, 0.5};
    Vec afm   = {0.5, 2.0};
    Vec adj = AdjustAndSample::adjust(theta, cfm, afm);
    // adj[0] = 0.5 * (2.0 - 0.5) = 0.75
    // adj[1] = 0.5 * (0.5 - 2.0) = -0.75
    assert(adj[0] > 0.0 && adj[1] < 0.0);
    assert(std::abs(adj[0] - 0.75) < EPS);
    assert(std::abs(adj[1] - (-0.75)) < EPS);
    std::cout << "  [PASS] adjust_promotes_coherent\n";
}

static void test_softmax_valid_distribution() {
    Vec v = {1.0, 2.0, -1.0, 0.5};
    Vec p = math::softmax(v);
    double sum = 0.0;
    for (double x : p) {
        assert(x >= 0.0 && x <= 1.0);
        sum += x;
    }
    assert(std::abs(sum - 1.0) < EPS);
    // Element with highest input should have highest probability
    assert(p[1] > p[0] && p[1] > p[2] && p[1] > p[3]);
    std::cout << "  [PASS] softmax_valid_distribution\n";
}

static void test_run_pipeline_samples_valid_index() {
    RNG::seed(42);
    Vec theta = {0.3, 0.7, 0.1, 0.9};
    Vec cfm   = {1.5, 2.0, 0.5, 1.8};
    Vec afm   = {0.3, 0.2, 1.5, 0.4};
    auto [idx, probs] = AdjustAndSample::run(theta, cfm, afm);
    assert(idx < theta.size());
    double sum = 0.0;
    for (double p : probs) sum += p;
    assert(std::abs(sum - 1.0) < EPS);
    std::cout << "  [PASS] run_pipeline_samples_valid_index (idx=" << idx << ")\n";
}

int main() {
    std::cout << "=== CFM / AFM Tests ===\n";
    test_cfm_identical_embeddings();
    test_cfm_orthogonal_embeddings();
    test_cfm_modality_weights_differ();
    test_afm_aligned_no_contradiction();
    test_afm_opposite_high_contradiction();
    test_afm_zero_prob_no_entropy();
    test_adjust_promotes_coherent();
    test_softmax_valid_distribution();
    test_run_pipeline_samples_valid_index();
    std::cout << "All CFM/AFM tests passed.\n";
    return 0;
}
