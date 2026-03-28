#pragma once

#include <dstt/core/types.hpp>
#include <dstt/core/arm.hpp>
#include <dstt/fdmp/fdmp.hpp>
#include <dstt/ea/population.hpp>
#include <dstt/mge/branch_predictor.hpp>
#include <dstt/mge/synthesiser.hpp>
#include <string>

namespace dstt {

/// Multimedia Generation Engine.
///
/// Orchestrates the full DSTT pipeline:
///   FDMP.encode → ARM.evaluate → EA.evolve → BP.predict → Synthesise
///
/// Implements Algorithm 2 from the specification.
class MGE {
public:
    explicit MGE(const Config& cfg);

    /// Generate a multi-modal output from an input prompt.
    /// @param input      The user prompt / description.
    /// @param steps      Number of output elements to generate.
    /// @param on_gen     Optional callback for EA generation stats.
    /// @return Synthesised multi-modal output.
    SynthesisedOutput generate(const std::string& input,
                               size_t steps = 10,
                               GenerationCallback on_gen = nullptr);

    /// Access sub-components (for testing / inspection).
    const FDMP& fdmp() const { return fdmp_; }
    const BranchPredictor& branch_predictor() const { return bp_; }

private:
    Config cfg_;
    FDMP fdmp_;
    ARM arm_;
    BranchPredictor bp_;
    Synthesiser synth_;

    /// Run a single generation step for one modality.
    OutputElement generate_step(const Vec& context,
                                const Vec& prev_state,
                                Modality modality,
                                const Vec& theta);
};

} // namespace dstt
