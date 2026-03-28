#pragma once

#include <dstt/core/types.hpp>
#include <string>
#include <vector>

namespace dstt {

/// A single generated element (one step of output).
struct OutputElement {
    Modality modality;
    Vec embedding;       ///< The parameter embedding that produced this.
    size_t param_index;  ///< Which parameter was sampled.
    double probability;  ///< Probability of the sampled parameter.
};

/// Synthesised multi-modal output.
struct SynthesisedOutput {
    std::vector<OutputElement> elements;

    size_t count(Modality m) const;
    double avg_probability() const;
};

/// Output Synthesiser: concatenates generated elements and enforces
/// cross-modal consistency checks.
class Synthesiser {
public:
    explicit Synthesiser(const Config& cfg);

    /// Append an element to the output.
    void append(OutputElement elem);

    /// Check cross-modal consistency between the last two elements.
    /// Returns true if consistent (or if < 2 elements).
    bool check_consistency() const;

    /// Get the current synthesised output.
    const SynthesisedOutput& output() const { return output_; }
    SynthesisedOutput& output() { return output_; }

    /// Reset for a new generation pass.
    void reset();

private:
    Config cfg_;
    SynthesisedOutput output_;
};

} // namespace dstt
