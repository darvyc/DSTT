#pragma once

#include <dstt/ea/chromosome.hpp>
#include <dstt/core/types.hpp>
#include <vector>
#include <utility>

namespace dstt {

/// Genetic operators for the evolutionary algorithm.
namespace Operators {

    /// Tournament selection: pick k random individuals, return the fittest.
    const Chromosome& tournament_select(const std::vector<Chromosome>& pop,
                                        size_t k);

    /// Single-point crossover.  Returns two offspring.
    std::pair<Chromosome, Chromosome>
    crossover(const Chromosome& p1, const Chromosome& p2);

    /// Random-resetting mutation.  For each gene, with probability mu,
    /// replace it with a uniform random value in [0, 1].
    void mutate(Chromosome& c, double mu);

} // namespace Operators

} // namespace dstt
