# DSTT — Dynamic Semi-Trained Topology

A C++17 framework for deterministic, contextually adaptive multimedia generation with a full training pipeline.

## Overview

DSTT is a trainable system for multi-modal content generation (text, image, video). Like large language models, DSTT has a distinct **training phase** where it learns from data before inference. Unlike LLMs, DSTT does not use attention or transformers — it combines a BPE tokenizer, learnable weight matrices, and an evolutionary algorithm to produce contextually adaptive parameter configurations.

### How It Works

DSTT operates in three phases:

```
Phase 1: Training          Phase 2: Parameter Evolution     Phase 3: Generation
─────────────────          ──────────────────────────       ────────────────────
Corpus → Tokenizer         Prompt → Tokenizer → FDMP       Branch Predictor
       → BPE vocab                → ARM (CFM/AFM)                → ARM evaluate
       → FDMP weight update       → EA evolve                    → Synthesise
       → Embedding update         → Best parameters              → Multi-modal output
```

**Phase 1 — Training.** A BPE tokenizer builds a sub-word vocabulary from the training corpus. Each training example is tokenized, embedded via learned token embeddings, passed through the FDMP to generate parameters, and evolved by the EA. The resulting fitness signal drives gradient updates to both the FDMP weight matrices and the token embedding table. This is analogous to pre-training in LLMs: the system learns input-to-parameter mappings so that inference starts from a better initialization.

**Phase 2 — Parameter Evolution.** At inference, the input prompt is tokenized and embedded. The FDMP generates raw parameters using its trained weights. The EA then evolves these parameters per modality using CFM (Correct Flow Matrix) and AFM (Adversarial Flow Matrix) scoring. Because the FDMP was trained, fewer EA generations are needed to reach high fitness.

**Phase 3 — Multi-Modal Generation.** A branch predictor selects which modality to generate next. The ARM evaluates optimized parameters, samples one, and the synthesizer appends the output element with cross-modal consistency checks. Context is updated after each step.

## Model Files (.dstt)

Trained DSTT models are saved as `.dstt` files — self-contained binary files that package all learned weights, vocabulary, and configuration into a single file. This is analogous to how `.gguf` files work for LLMs in llama.cpp.

### .dstt File Format

A `.dstt` file contains:

| Section | Contents |
|---------|----------|
| **Header** | Magic bytes (`DSTT`), version, dimensions |
| **Config** | All hyperparameters used during training |
| **Vocabulary** | BPE token table (token strings + IDs) |
| **Embeddings** | Token embedding matrix (vocab x embed_dim) |
| **FDMP Weights** | Per-modality W matrices and bias vectors |

### Interacting with .dstt Models

Load and interact with trained models using `DSTTModel` or the interactive CLI:

```cpp
#include <dstt/model/dstt_model.hpp>

// Load a trained model
dstt::DSTTModel model;
model.load("models/my_model.dstt");
std::cout << model.info() << "\n";

// Generate output from a prompt
auto output = model.generate("A sunset over mountains", 12,
    [](size_t step, const dstt::OutputElement& elem) {
        std::cout << step << ": " << dstt::modality_name(elem.modality) << "\n";
    });
```

#### Interactive CLI

```bash
./dstt_cli models/my_model.dstt
```

```
dstt> A sunset over snowy peaks
  [0] Image  prob=0.0329
  [1] Video  prob=0.0358
  ...

dstt> /info
DSTT Model
  Format:          .dstt v1
  Embed dim:       32
  Param dim:       32
  Vocabulary:      404 tokens

dstt> /train training_data/example.jsonl models/new_model.dstt
Loaded 24 training examples
Training...
  Epoch 0  fitness=0.9995  loss=0.0005
Model saved to models/new_model.dstt and loaded.
```

CLI commands:

| Command | Description |
|---------|-------------|
| `/load <path.dstt>` | Load a trained model |
| `/info` | Show model metadata |
| `/steps <n>` | Set generation steps |
| `/train <data> <out>` | Train from data file, save as .dstt |
| `/help` | Show help |
| `/quit` | Exit |

### Training from Files

Training data goes in the `training_data/` directory. Three formats are supported:

**JSONL** (recommended — supports all modalities):
```jsonl
{"input": "A sunset over mountains", "modality": "Image"}
{"input": "Breaking news about science", "modality": "Text"}
{"input": "Time-lapse of a flower", "modality": "Video"}
```

**CSV:**
```csv
input,modality
A sunset over mountains,Image
Breaking news about science,Text
```

**Plain text** (defaults to Text modality):
```text
A sunset over mountains
Breaking news about science
```

### Train and Save Programmatically

```cpp
dstt::DSTTModel model(cfg);
auto examples = dstt::DSTTModel::load_training_jsonl("training_data/example.jsonl");
model.train_and_save(examples, "models/my_model.dstt",
    [](const dstt::EpochStats& s) {
        std::cout << "Epoch " << s.epoch << " loss=" << s.avg_loss << "\n";
    });
```

## Architecture

### Tokenizer (BPE)

The tokenizer learns a sub-word vocabulary from training data using Byte-Pair Encoding:

1. Start with a byte-level vocabulary (256 tokens)
2. Iteratively merge the most frequent adjacent token pair
3. Stop when `vocab_size` is reached or no pair exceeds `min_token_freq`

Token embeddings are a learnable matrix `E ∈ R^{vocab × embed_dim}`, initialized with Xavier distribution and updated during training. Input text is converted to a context vector by averaging token embeddings:

```
C = (1/|tokens|) * sum(E[token_i])
```

### FDMP (Fundamental Data Matrix Processor)

Maintains per-modality weight matrices `W_m` and bias vectors `b_m`. Generates raw parameter vectors via affine transform:

```
Theta_r = W_m * C + b_m
```

During training, these weights are updated using the EA fitness signal as a gradient approximation, similar to how neural network weights are updated via backpropagation.

### Dual Flow Matrices

| Matrix | Role | Formula |
|--------|------|---------|
| **CFM** (Correct Flow) | Promotes coherent parameters | `alpha_m * cos(embed(theta_j), context) + Ramsey_coherence` |
| **AFM** (Adversarial Flow) | Penalizes incoherence | `max(0, -cos(...)) + Shannon_entropy` |

Parameters are adjusted: `theta'_j = theta_j * (CFM_j - AFM_j)`, then softmax → sample.

### Evolutionary Algorithm

Tournament selection, single-point crossover, and mutation evolve parameter vectors. Fitness combines three weighted metrics:

```
F = w_c * Coherence + w_r * Relevance + w_d * Diversity
```

- **Coherence:** Probability mass on parameters where CFM > AFM
- **Relevance:** Cosine similarity of weighted parameters to context
- **Diversity:** Normalized Shannon entropy of the distribution

### Subsystems

| Subsystem | Role |
|-----------|------|
| **Tokenizer** | BPE vocabulary learning + sub-word encoding + learned embeddings |
| **FDMP** | Trained weight matrices that map context embeddings to parameter vectors |
| **ARM** | Partitions parameter space, evaluates via CFM/AFM, adjusts and samples |
| **EA** | Evolves parameter configurations across generations |
| **MGE** | Orchestrates generation: branch prediction + synthesis + consistency |
| **Trainer** | Training loop: tokenizer building, FDMP weight updates, embedding updates |
| **DSTTModel** | .dstt file I/O, model loading, inference interface, training data loaders |

## Training

### Training Data

Training examples pair input text with a target modality:

```cpp
std::vector<TrainingExample> data = {
    {"A sunset over mountains",        Modality::Image},
    {"Breaking news about science",    Modality::Text},
    {"Time-lapse of a flower",         Modality::Video},
};
```

### Training Loop

```cpp
Config cfg;
cfg.training_epochs = 10;
cfg.training_lr     = 0.005;
cfg.vocab_size      = 4096;

Trainer trainer(cfg);
trainer.train(data, [](const EpochStats& stats) {
    std::cout << "Epoch " << stats.epoch
              << " fitness=" << stats.avg_fitness
              << " loss=" << stats.avg_loss << "\n";
});

// Save trained model
trainer.save("models/my_model");

// Load later
Trainer loaded(cfg);
loaded.load("models/my_model");
```

### What Gets Trained

| Component | What is learned | Update method |
|-----------|----------------|---------------|
| Token embeddings `E` | Dense vector per sub-word token | Gradient from fitness signal |
| FDMP weights `W_m` | Per-modality projection matrices | Fitness-driven gradient approximation |
| FDMP biases `b_m` | Per-modality bias vectors | Fitness-driven gradient approximation |

### Training vs Inference

| | Training | Inference |
|---|---------|-----------|
| **Input** | Corpus of (text, modality) pairs | Single prompt |
| **Tokenizer** | Builds BPE vocabulary | Uses learned vocabulary |
| **FDMP weights** | Updated each step | Frozen |
| **EA** | Short runs (signal for weight updates) | Full runs (parameter optimization) |
| **Output** | Trained weight files | Multi-modal content |

## Building

### Requirements

- C++17 compiler (GCC >= 9, Clang >= 10, MSVC >= 19.20)
- CMake >= 3.16

### Quick Start

```bash
git clone https://github.com/darvyc/dstt-full.git
cd dstt-full
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

Five test suites: partitioning, CFM/AFM, EA operators, integration, and **training**.

### Running the Demo

```bash
./dstt_demo
```

The demo runs all three phases: trains on a small corpus, then generates multi-modal output for three different prompts.

### Running the Interactive CLI

```bash
# Train from data and start interactive session
./dstt_cli

# Or load an existing .dstt model directly
./dstt_cli models/my_model.dstt
```

## Project Structure

```
dstt-full/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── models/                    # Trained .dstt model files go here
│   └── .gitkeep
├── training_data/             # Training data files (JSONL, CSV, TXT)
│   ├── example.jsonl          # Example JSONL training data
│   ├── example.csv            # Example CSV training data
│   ├── example.txt            # Example plain text training data
│   └── README.md              # Training data format documentation
├── include/dstt/
│   ├── core/
│   │   ├── types.hpp          # Types, enums, Config (incl. training params)
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
│   │   ├── tokenizer.hpp      # BPE tokenizer + learned embeddings
│   │   └── embeddings.hpp     # Modality-specific embedding helpers
│   ├── mge/
│   │   ├── mge.hpp            # Multimedia Generation Engine
│   │   ├── branch_predictor.hpp
│   │   └── synthesiser.hpp    # Output synthesis
│   ├── model/
│   │   ├── dstt_format.hpp    # .dstt binary file format specification
│   │   └── dstt_model.hpp     # Model loading, saving, and inference API
│   ├── training/
│   │   └── trainer.hpp        # Training loop + save/load
│   └── utils/
│       ├── math.hpp           # Linear algebra, cosine similarity
│       ├── random.hpp         # Thread-safe RNG
│       └── timer.hpp          # High-resolution profiling
├── src/
│   ├── core/        # ARM, CFM, AFM, partition, softmax
│   ├── ea/          # Chromosome, operators, fitness, population
│   ├── fdmp/        # FDMP, tokenizer, embeddings
│   ├── mge/         # MGE, branch predictor, synthesiser
│   ├── model/       # DSTTModel (.dstt file I/O + inference)
│   ├── training/    # Trainer
│   └── utils/       # Math, RNG
├── tests/
│   ├── test_partition.cpp
│   ├── test_cfm_afm.cpp
│   ├── test_ea.cpp
│   ├── test_integration.cpp
│   └── test_training.cpp      # Tokenizer + training tests
└── examples/
    ├── demo.cpp               # Full pipeline demo (train → evolve → generate)
    └── dstt_cli.cpp           # Interactive CLI for .dstt model files
```

## Configuration

Key hyperparameters (set in `include/dstt/core/types.hpp` or at runtime):

### Evolutionary Algorithm

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 200 | EA population |
| `max_generations` | 500 | EA iterations |
| `tournament_k` | 5 | Tournament selection size |
| `mutation_rate` | 1/m | Per-gene mutation probability |
| `elitism_rate` | 0.10 | Fraction of elites preserved |

### Evaluation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coherence_threshold` | 0.25 | Ramsey coherence threshold for partitioning |
| `w_coherence` | 0.4 | Fitness weight: coherence |
| `w_relevance` | 0.4 | Fitness weight: relevance |
| `w_diversity` | 0.2 | Fitness weight: diversity |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_epochs` | 10 | Number of passes over the training corpus |
| `training_lr` | 0.005 | Learning rate for FDMP weight updates |
| `vocab_size` | 4096 | Maximum BPE vocabulary size |
| `min_token_freq` | 2 | Minimum pair frequency for BPE merges |
| `weight_decay` | 1e-4 | L2 regularization on weights |

## License

MIT.
