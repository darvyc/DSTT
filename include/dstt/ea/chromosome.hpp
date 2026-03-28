#pragma once

#include <dstt/core/types.hpp>
#include <ostream>

namespace dstt {

/// A chromosome encodes a complete parameter vector Θ for one modality.
///
/// Genes are floats in [0, 1].  Mapping to parameter range:
///   θ_j = g_j · (θ_max − θ_min) + θ_min
///
/// For DSTT the default range is [0, 1] so decode is identity.
struct Chromosome {
    Vec genes;          ///< Gene values g_j ∈ [0, 1].
    double fitness = 0.0;

    /// Create a random chromosome of the given dimension.
    static Chromosome random(size_t dim);

    /// Decode genes into parameter vector (identity for [0,1] range).
    Vec decode(double theta_min = 0.0, double theta_max = 1.0) const;

    /// Number of genes.
    size_t size() const { return genes.size(); }

    bool operator<(const Chromosome& o) const { return fitness < o.fitness; }
    bool operator>(const Chromosome& o) const { return fitness > o.fitness; }
};

std::ostream& operator<<(std::ostream& os, const Chromosome& c);

} // namespace dstt
