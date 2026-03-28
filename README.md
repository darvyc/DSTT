# DSTT — Dynamic Semi-Trained Topology

A C++17 framework for deterministic, contextually adaptive multimedia generation.

DSTT replaces heuristic parameter tuning with a closed-loop architecture built on three pillars:

1. **Combinatorial Partitioning** — Ramanujan–Hardy partition theory and Ramsey-coherence clustering segment a high-dimensional parameter space into tractable, logically coherent subsets.
2. **Dual Flow Matrices** — the Correct Flow Matrix (CFM) promotes contextually relevant parameters; the Adversarial Flow Matrix (AFM) penalises incoherence.
3. **Embedded Evolutionary Algorithm** — tournament selection, single-point crossover, and mutation evolve parameter configurations across generations.

Two core subsystems execute this design:

| Subsystem | Role |
|-----------|------|
| **FDMP** (Fundamental Data Matrix Processor) | Encodes raw input into structured embeddings across linguistic, visual, and auditory modalities. |
| **ARM** (Autonomous Route Matrix) | Partitions the parameter space, evaluates via CFM/AFM, adjusts probabilities via softmax, and feeds fitness back to the EA. |
| **MGE** (Multimedia Generation Engine) | Synthesises final outputs using branch prediction, attribute-vector partitioning, and EA-optimised parameters. |

## Building

### Requirements

- C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 19.20)
- CMake ≥ 3.16

### Quick Start

```bash
git clone https://github.com/darvyc/dstt.git
cd dstt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

### Running the Demo

```bash
./dstt_demo
```

## Project Structure

```
dstt/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/dstt/
│   ├── core/
│   │   ├── types.hpp          # Fundamental types, enums, constants
│   │   ├── arm.hpp            # Autonomous Route Matrix
│   │   ├── partition.hpp      # Combinatorial partitioning engine
│   │   ├── cfm.hpp            # Correct Flow Matrix
│   │   ├── afm.hpp            # Adversarial Flow Matrix
│   │   └── softmax.hpp        # Softmax + parameter sampling
│   ├── ea/
│   │   ├── chromosome.hpp     # Genetic encoding
│   │   ├── operators.hpp      # Selection, crossover, mutation
│   │   ├── fitness.hpp        # Fitness evaluation
│   │   └── population.hpp     # Population manager + generational loop
│   ├── fdmp/
│   │   ├── fdmp.hpp           # Fundamental Data Matrix Processor
│   │   └── embeddings.hpp     # Modality-specific embedding helpers
│   ├── mge/
│   │   ├── mge.hpp            # Multimedia Generation Engine
│   │   ├── branch_predictor.hpp
│   │   └── synthesiser.hpp    # Output synthesis
│   └── utils/
│       ├── math.hpp           # Linear algebra, cosine similarity
│       ├── random.hpp         # Thread-safe RNG
│       └── timer.hpp          # High-resolution profiling
├── src/
│   ├── core/   ── .cpp implementations
│   ├── ea/
│   ├── fdmp/
│   ├── mge/
│   └── utils/
├── tests/
│   ├── test_partition.cpp
│   ├── test_cfm_afm.cpp
│   ├── test_ea.cpp
│   └── test_integration.cpp
├── examples/
│   └── demo.cpp
└── docs/
    └── DSTT_v2.docx
```

## Configuration

Key hyperparameters (set in `include/dstt/core/types.hpp` or at runtime):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POPULATION_SIZE` | 200 | EA population |
| `MAX_GENERATIONS` | 500 | EA iterations |
| `TOURNAMENT_K` | 5 | Tournament selection size |
| `MUTATION_RATE` | 1/m | Per-gene mutation probability |
| `ELITISM_RATE` | 0.10 | Fraction of elites preserved |
| `COHERENCE_THRESHOLD` | 0.25 | τ for partition clustering |
| `W_COHERENCE` | 0.4 | Fitness weight: coherence |
| `W_RELEVANCE` | 0.4 | Fitness weight: relevance |
| `W_DIVERSITY` | 0.2 | Fitness weight: diversity |

## License

MIT.
