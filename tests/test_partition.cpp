#include <dstt/core/partition.hpp>
#include <dstt/utils/math.hpp>
#include <dstt/utils/random.hpp>
#include <iostream>
#include <cassert>
#include <set>
#include <algorithm>

using namespace dstt;

static void test_union_find_basic() {
    UnionFind uf(5);
    uf.merge(0, 1);
    uf.merge(2, 3);
    uf.merge(3, 4);
    auto ps = uf.extract();
    assert(ps.size() == 2);
    std::cout << "  [PASS] union_find_basic\n";
}

static void test_union_find_all_separate() {
    UnionFind uf(4);
    auto ps = uf.extract();
    assert(ps.size() == 4);
    std::cout << "  [PASS] union_find_all_separate\n";
}

static void test_union_find_all_merged() {
    UnionFind uf(4);
    uf.merge(0, 1);
    uf.merge(1, 2);
    uf.merge(2, 3);
    auto ps = uf.extract();
    assert(ps.size() == 1);
    assert(ps[0].size() == 4);
    std::cout << "  [PASS] union_find_all_merged\n";
}

static void test_partition_disjoint_cover() {
    RNG::seed(123);
    Config cfg;
    cfg.param_dim = 16;
    cfg.embed_dim = 8;
    cfg.coherence_threshold = 0.5;

    Vec theta = RNG::random_vec(cfg.param_dim, -1.0, 1.0);
    Vec context = RNG::random_vec(cfg.embed_dim, -1.0, 1.0);

    auto ps = PartitionEngine::partition(theta, context, Modality::Text, cfg);

    // Check disjoint cover
    std::set<size_t> all_indices;
    for (const auto& p : ps) {
        for (size_t idx : p) {
            assert(all_indices.find(idx) == all_indices.end() &&
                   "Partition overlap detected");
            all_indices.insert(idx);
        }
    }
    assert(all_indices.size() == cfg.param_dim);
    std::cout << "  [PASS] partition_disjoint_cover (" << ps.size() << " partitions)\n";
}

static void test_partition_count() {
    // Hardy–Ramanujan should give sensible partition counts
    assert(math::partition_count(1) >= 2);
    assert(math::partition_count(10) >= 2);
    assert(math::partition_count(100) >= 2);
    size_t k100 = math::partition_count(100);
    size_t k1000 = math::partition_count(1000);
    assert(k1000 > k100);  // sub-exponential but still grows
    std::cout << "  [PASS] partition_count (k(100)=" << k100
              << ", k(1000)=" << k1000 << ")\n";
}

int main() {
    std::cout << "=== Partition Tests ===\n";
    test_union_find_basic();
    test_union_find_all_separate();
    test_union_find_all_merged();
    test_partition_disjoint_cover();
    test_partition_count();
    std::cout << "All partition tests passed.\n";
    return 0;
}
