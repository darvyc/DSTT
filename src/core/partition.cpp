#include <dstt/core/partition.hpp>
#include <dstt/utils/math.hpp>
#include <cmath>
#include <unordered_map>

namespace dstt {

// ── UnionFind ────────────────────────────────────────────────────────

UnionFind::UnionFind(size_t n) : parent_(n), rank_(n, 0), n_(n) {
    for (size_t i = 0; i < n; ++i) parent_[i] = i;
}

size_t UnionFind::find(size_t x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];   // path halving
        x = parent_[x];
    }
    return x;
}

void UnionFind::merge(size_t x, size_t y) {
    x = find(x);
    y = find(y);
    if (x == y) return;
    if (rank_[x] < rank_[y]) std::swap(x, y);
    parent_[y] = x;
    if (rank_[x] == rank_[y]) ++rank_[x];
}

PartitionSet UnionFind::extract() const {
    std::unordered_map<size_t, Partition> groups;
    // Need a mutable copy for find() — but extract is const, so
    // we re-trace without path compression.
    for (size_t i = 0; i < n_; ++i) {
        size_t root = i;
        // trace without mutation
        auto p = parent_;
        while (p[root] != root) root = p[root];
        groups[root].push_back(i);
    }
    PartitionSet ps;
    ps.reserve(groups.size());
    for (auto& [_, g] : groups) ps.push_back(std::move(g));
    return ps;
}

// ── PartitionEngine ──────────────────────────────────────────────────

/// Compute pairwise Ramsey coherence between two parameter "embeddings".
/// We create a simple embedding for θ_i by tiling its value across a
/// vector of length embed_dim modulated by the context.
static Vec param_embedding(double theta_val, const Vec& context, size_t dim) {
    Vec e(dim);
    for (size_t d = 0; d < dim; ++d) {
        size_t ci = d % context.size();
        e[d] = theta_val * context[ci];
    }
    return e;
}

PartitionSet PartitionEngine::partition(const Vec& theta,
                                        const Vec& context,
                                        Modality /*modality*/,
                                        const Config& cfg) {
    size_t m = theta.size();
    if (m <= 1) {
        PartitionSet ps;
        if (m == 1) ps.push_back({0});
        return ps;
    }

    double tau = cfg.coherence_threshold;
    size_t edim = std::min(cfg.embed_dim, context.size());

    // Pre-compute embeddings
    std::vector<Vec> embeds(m);
    for (size_t i = 0; i < m; ++i) {
        embeds[i] = param_embedding(theta[i], context, edim);
    }

    UnionFind uf(m);

    // Pairwise coherence check
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = i + 1; j < m; ++j) {
            // Ramsey coherence: 1 − d_H / m  (normalised Hamming distance)
            size_t dh = math::hamming_distance(embeds[i], embeds[j]);
            double rc = 1.0 - static_cast<double>(dh) / static_cast<double>(edim);
            if (rc > tau) {
                uf.merge(i, j);
            }
        }
    }

    return uf.extract();
}

} // namespace dstt
