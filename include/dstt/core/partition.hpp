#pragma once

#include <dstt/core/types.hpp>
#include <vector>

namespace dstt {

/// A single partition: indices into the parameter vector.
using Partition = std::vector<size_t>;

/// Set of partitions (disjoint cover of {0, …, m-1}).
using PartitionSet = std::vector<Partition>;

/// Union-Find (disjoint set) data structure for partition merging.
class UnionFind {
public:
    explicit UnionFind(size_t n);
    size_t find(size_t x);
    void merge(size_t x, size_t y);
    /// Extract partitions as groups of indices.
    PartitionSet extract() const;

private:
    std::vector<size_t> parent_;
    std::vector<size_t> rank_;
    size_t n_;
};

/// Partition engine.
///
/// Given parameter vector Θ ∈ ℝᵐ, context C ∈ ℝⁿ, and modality m,
/// produce a PartitionSet using coherence-threshold clustering.
///
/// Two parameters θ_i, θ_j are in the same partition iff
///   R_c(θ_i, θ_j, m) > τ
/// where R_c is the Ramsey coherence measure and τ = coherence_threshold.
class PartitionEngine {
public:
    /// Build partitions.
    /// @param theta     Raw parameter vector (size m).
    /// @param context   Context embedding (size n).
    /// @param modality  Current modality.
    /// @param cfg       Configuration (provides τ and embed_dim).
    static PartitionSet partition(const Vec& theta,
                                  const Vec& context,
                                  Modality modality,
                                  const Config& cfg);
};

} // namespace dstt
