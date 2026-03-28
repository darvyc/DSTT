#include <dstt/ea/operators.hpp>
#include <dstt/utils/random.hpp>
#include <algorithm>
#include <cassert>

namespace dstt { namespace Operators {

const Chromosome& tournament_select(const std::vector<Chromosome>& pop,
                                    size_t k) {
    assert(!pop.empty() && k > 0);
    k = std::min(k, pop.size());

    size_t best = RNG::uniform_int(0, pop.size() - 1);
    for (size_t i = 1; i < k; ++i) {
        size_t idx = RNG::uniform_int(0, pop.size() - 1);
        if (pop[idx].fitness > pop[best].fitness) {
            best = idx;
        }
    }
    return pop[best];
}

std::pair<Chromosome, Chromosome>
crossover(const Chromosome& p1, const Chromosome& p2) {
    assert(p1.size() == p2.size() && p1.size() > 1);

    size_t dim = p1.size();
    size_t c = RNG::uniform_int(1, dim - 1);  // crossover point

    Chromosome o1, o2;
    o1.genes.resize(dim);
    o2.genes.resize(dim);

    for (size_t i = 0; i < dim; ++i) {
        if (i < c) {
            o1.genes[i] = p1.genes[i];
            o2.genes[i] = p2.genes[i];
        } else {
            o1.genes[i] = p2.genes[i];
            o2.genes[i] = p1.genes[i];
        }
    }
    return {o1, o2};
}

void mutate(Chromosome& c, double mu) {
    for (auto& g : c.genes) {
        if (RNG::uniform() < mu) {
            g = RNG::uniform(0.0, 1.0);
        }
    }
}

}} // namespace dstt::Operators
