#include <dstt/mge/branch_predictor.hpp>
#include <dstt/utils/math.hpp>
#include <algorithm>

namespace dstt {

BranchPredictor::BranchPredictor(const Config& cfg)
    : cfg_(cfg),
      weights_(MODALITY_COUNT * cfg.embed_dim, 0.0),
      biases_(MODALITY_COUNT, 0.0) {
    last_gen_.fill(SIZE_MAX);  // sentinel: never generated
    // Small random init
    for (auto& w : weights_) w = 0.001 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
}

std::pair<Modality, double> BranchPredictor::predict(const Vec& context) const {
    // Linear: logits_m = W_m · context + b_m
    Vec logits(MODALITY_COUNT);
    size_t dim = std::min(context.size(), cfg_.embed_dim);
    for (size_t m = 0; m < MODALITY_COUNT; ++m) {
        double s = biases_[m];
        for (size_t d = 0; d < dim; ++d) {
            s += weights_[m * cfg_.embed_dim + d] * context[d];
        }
        logits[m] = s;
    }
    Vec probs = math::softmax(logits);

    // Find argmax
    size_t best = 0;
    for (size_t m = 1; m < MODALITY_COUNT; ++m) {
        if (probs[m] > probs[best]) best = m;
    }

    return {static_cast<Modality>(best), probs[best]};
}

void BranchPredictor::update(const Vec& context, Modality actual) {
    size_t dim = std::min(context.size(), cfg_.embed_dim);
    size_t target = static_cast<size_t>(actual);

    // Compute current probabilities
    Vec logits(MODALITY_COUNT);
    for (size_t m = 0; m < MODALITY_COUNT; ++m) {
        double s = biases_[m];
        for (size_t d = 0; d < dim; ++d) {
            s += weights_[m * cfg_.embed_dim + d] * context[d];
        }
        logits[m] = s;
    }
    Vec probs = math::softmax(logits);

    // Cross-entropy gradient: ∂L/∂logit_m = p_m − 1{m == target}
    for (size_t m = 0; m < MODALITY_COUNT; ++m) {
        double grad = probs[m] - (m == target ? 1.0 : 0.0);
        biases_[m] -= learning_rate_ * grad;
        for (size_t d = 0; d < dim; ++d) {
            weights_[m * cfg_.embed_dim + d] -= learning_rate_ * grad * context[d];
        }
    }
}

void BranchPredictor::record_generation(Modality m) {
    last_gen_[static_cast<size_t>(m)] = step_;
    ++step_;
}

Modality BranchPredictor::least_recent() const {
    size_t best = 0;
    size_t oldest = last_gen_[0];
    for (size_t m = 1; m < MODALITY_COUNT; ++m) {
        // SIZE_MAX means never generated — always prefer that
        if (last_gen_[m] == SIZE_MAX && oldest != SIZE_MAX) {
            return static_cast<Modality>(m);
        }
        if (last_gen_[m] < oldest) {
            oldest = last_gen_[m];
            best = m;
        }
    }
    return static_cast<Modality>(best);
}

} // namespace dstt
