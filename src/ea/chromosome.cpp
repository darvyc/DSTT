#include <dstt/ea/chromosome.hpp>
#include <dstt/utils/random.hpp>
#include <iomanip>

namespace dstt {

Chromosome Chromosome::random(size_t dim) {
    Chromosome c;
    c.genes = RNG::random_vec(dim, 0.0, 1.0);
    c.fitness = 0.0;
    return c;
}

Vec Chromosome::decode(double theta_min, double theta_max) const {
    Vec theta(genes.size());
    double range = theta_max - theta_min;
    for (size_t i = 0; i < genes.size(); ++i) {
        theta[i] = genes[i] * range + theta_min;
    }
    return theta;
}

std::ostream& operator<<(std::ostream& os, const Chromosome& c) {
    os << "Chromosome(dim=" << c.size()
       << ", fitness=" << std::fixed << std::setprecision(6) << c.fitness << ")";
    return os;
}

} // namespace dstt
