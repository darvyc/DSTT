#include <dstt/mge/mge.hpp>
#include <iostream>

namespace dstt {

MGE::MGE(const Config& cfg)
    : cfg_(cfg), fdmp_(cfg), arm_(cfg), bp_(cfg), synth_(cfg) {}

OutputElement MGE::generate_step(const Vec& context,
                                  const Vec& prev_state,
                                  Modality modality,
                                  const Vec& theta) {
    ARMResult r = arm_.evaluate(theta, context, prev_state, modality);

    // Build a param embedding for the sampled parameter
    size_t edim = std::min(cfg_.embed_dim, context.size());
    Vec param_embed(edim);
    double theta_val = theta[r.sampled_idx];
    for (size_t d = 0; d < edim; ++d) {
        size_t ci = d % context.size();
        param_embed[d] = theta_val * context[ci];
    }

    return OutputElement{
        modality,
        std::move(param_embed),
        r.sampled_idx,
        r.probabilities[r.sampled_idx]
    };
}

SynthesisedOutput MGE::generate(const std::string& input,
                                 size_t steps,
                                 GenerationCallback on_gen) {
    synth_.reset();

    // Phase 1: Encode input and evolve parameters per modality
    struct ModalityState {
        Vec context;
        Vec theta_optimised;
    };

    std::array<ModalityState, MODALITY_COUNT> states;
    Vec init_prev_state(cfg_.embed_dim, 0.0);

    for (size_t mi = 0; mi < MODALITY_COUNT; ++mi) {
        Modality m = static_cast<Modality>(mi);
        auto [ctx, raw_theta] = fdmp_.process(input, m);
        states[mi].context = ctx;

        // Evolve parameters for this modality
        // Use fewer generations for the demo to keep runtime reasonable
        Config ea_cfg = cfg_;
        ea_cfg.max_generations = std::min(cfg_.max_generations, size_t(50));
        ea_cfg.population_size = std::min(cfg_.population_size, size_t(30));

        Population pop(ea_cfg);
        Chromosome best = pop.evolve(ctx, init_prev_state, m,
            (mi == 0) ? on_gen : nullptr  // only report for first modality
        );
        states[mi].theta_optimised = best.decode();
    }

    // Phase 2: Generate output elements using branch prediction
    Vec current_context = states[0].context;  // start with text context
    Vec prev_state = init_prev_state;

    for (size_t step = 0; step < steps; ++step) {
        // Predict next modality
        auto [predicted_m, confidence] = bp_.predict(current_context);

        // Fallback if low confidence
        if (confidence < cfg_.bp_confidence_threshold) {
            predicted_m = bp_.least_recent();
        }

        size_t mi = static_cast<size_t>(predicted_m);
        const Vec& theta = states[mi].theta_optimised;

        // Generate one element
        OutputElement elem = generate_step(current_context, prev_state,
                                           predicted_m, theta);
        synth_.append(elem);

        // Check consistency
        if (!synth_.check_consistency()) {
            // If inconsistent, try the fallback modality
            Modality fallback = bp_.least_recent();
            size_t fi = static_cast<size_t>(fallback);
            OutputElement alt = generate_step(current_context, prev_state,
                                              fallback, states[fi].theta_optimised);
            // Replace last element
            synth_.output().elements.back() = std::move(alt);
        }

        // Update context: blend current context with the element's embedding
        const auto& last = synth_.output().elements.back();
        prev_state = current_context;
        if (last.embedding.size() == current_context.size()) {
            for (size_t d = 0; d < current_context.size(); ++d) {
                current_context[d] = 0.8 * current_context[d]
                                   + 0.2 * last.embedding[d];
            }
        }

        // Train branch predictor and record
        bp_.update(current_context, last.modality);
        bp_.record_generation(last.modality);
    }

    return synth_.output();
}

} // namespace dstt
