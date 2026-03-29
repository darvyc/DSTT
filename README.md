# DSTT — Dynamic Semi-Trained Topology

A C++17 framework for deterministic, contextually adaptive multimedia generation with a full training pipeline. After training, DSTT generates **text**, **images**, and **video** from natural language prompts.

## Overview

DSTT is a trainable system for multi-modal content generation. Like Large Language Models (LLMs), DSTT has a distinct **training phase** where it learns from data before inference. Unlike LLMs, DSTT does not use attention or transformers — it combines a Byte-Pair Encoding (BPE) tokenizer, learnable weight matrices, and an Evolutionary Algorithm (EA) to produce contextually adaptive parameter configurations that drive content synthesis across three modalities.

### What DSTT Generates

Given a trained `.dstt` model and a text prompt, DSTT produces:

| Modality | Output | How |
|----------|--------|-----|
| **Text** | Generated text sampled from the learned BPE vocabulary | Token embeddings scored against evolved parameter distributions via cosine similarity, then temperature-sampled |
| **Image** | 64x64 RGB images saved as PPM files | Evolved parameters mapped to pixel values through sigmoid activation with spatial phase modulation |
| **Video** | Temporally coherent frame sequences saved as PPM series | Image generation extended with time-varying parameter modulation and inter-frame blending |

A **branch predictor** selects which modality to generate at each step based on the prompt context. The model cycles through text, image, and video generation, producing multi-modal output from a single prompt.

### How It Works

DSTT operates in three phases:

```
Phase 1: Training          Phase 2: Parameter Evolution     Phase 3: Content Generation
─────────────────          ──────────────────────────       ───────────────────────────
Corpus → Tokenizer         Prompt → Tokenizer → FDMP        Branch Predictor → modality
       → BPE vocab                → ARM (CFM/AFM)           ARM evaluate → sample params
       → FDMP weight update       → EA evolve               ContentGenerator → text/image/video
       → Embedding update         → Best parameters        Synthesise → multi-modal output
```

**Phase 1 — Training.** A BPE tokenizer builds a sub-word vocabulary from the training corpus. Each training example is tokenized, embedded via learned token embeddings, passed through the Fundamental Data Matrix Processor (FDMP) to generate parameters, and evolved by the EA. The resulting fitness signal drives gradient updates to both the FDMP weight matrices and the token embedding table. This is analogous to pre-training in LLMs: the system learns input-to-parameter mappings so that inference starts from a better initialization.

**Phase 2 — Parameter Evolution.** At inference, the input prompt is tokenized and embedded. The FDMP generates raw parameters using its trained weights. The EA then evolves these parameters per modality using Correct Flow Matrix (CFM) and Adversarial Flow Matrix (AFM) scoring. Because the FDMP was trained, fewer EA generations are needed to reach high fitness.

**Phase 3 — Content Generation.** A branch predictor selects which modality (text, image, or video) to generate at each step. The Autonomous Route Matrix (ARM) evaluates optimized parameters and samples from the distribution. The **ContentGenerator** then transforms those parameters into actual content:
- **Text:** Computes similarity between each vocabulary token's learned embedding and the evolved parameter embedding, applies temperature-controlled softmax, and samples tokens — the same fundamental approach LLMs use, but driven by evolutionary optimization instead of transformer attention.
- **Image:** Maps evolved parameters to RGB pixel values via `sigmoid(θ[i] · context[j] + spatial_phase)`, producing coherent color gradients modulated by the prompt context.
- **Video:** Extends image generation with temporal parameter modulation `θ_t[j] = θ[j] · (1 + 0.3·sin(2πt + phase))` and inter-frame blending for smooth motion.

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Launch the interactive CLI
./dstt_cli

# Set a system prompt (defines model behavior, like an LLM system message)
dstt> /system You are an expert landscape photographer and nature writer.

# Set target parameter count (like choosing 7B vs 13B for an LLM)
dstt> /params 100000

# Train the model
dstt> /train ../training_data/example.jsonl ../models/my_model.dstt
System prompt: "You are an expert landscape photographer and nature writer."
Parameters:    99429
Training...
  Epoch 0  fitness=0.9989  loss=0.0011

# Generate content from a prompt
dstt> A sunset over snow-capped mountains
  [0] Text -> "mountain sunset..."
  [1] Image -> 64x64 image
  [2] Video -> video frame
  ...

# Save generated images and video to disk
dstt> /save_images
dstt> /save_video
dstt> A cinematic landscape at golden hour
```

## Model Files (.dstt)

Trained DSTT models are saved as `.dstt` files — self-contained binary files that package all learned weights, vocabulary, and configuration into a single file. This is analogous to how `.gguf` files work for LLMs in llama.cpp.

### .dstt File Format

A `.dstt` file contains:

| Section | Contents |
|---------|----------|
| **Header** | Magic bytes (`DSTT`), version, dimensions, parameter count |
| **Config** | All hyperparameters used during training |
| **System Prompt** | The training focus prompt (e.g. "Be a helpful assistant.") |
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

// Generate multi-modal content from a prompt
auto result = model.run("A sunset over mountains", 12,
    [](size_t step, const dstt::GeneratedContent& c) {
        std::cout << step << ": " << dstt::modality_name(c.modality);
        if (c.modality == dstt::Modality::Text)
            std::cout << " -> \"" << c.text << "\"";
        std::cout << "\n";
    });

// Access the generated content
std::cout << "Generated text: " << result.generated_text << "\n";
std::cout << "Images: " << result.images.size() << "\n";
std::cout << "Video frames: " << result.video.frames.size() << "\n";

// Save images and video to disk
result.images[0].save_ppm("output.ppm");
result.video.save_frames("output_video");
```

### Interactive CLI

```bash
./dstt_cli models/my_model.dstt
```

```
dstt> A sunset over snowy peaks
Generating...
  [0] Text -> "mountain peaks golden..."
  [1] Image -> 64x64 image
  [2] Video -> video frame
  [3] Text -> "light fading..."

========================================================================
  Generation Result for: "A sunset over snowy peaks"
========================================================================

  [Text Output] (4 steps)
  mountain peaks golden...light fading...

  [Image Output] 4 image(s) generated (64x64 RGB)
  [Video Output] 4 frame(s) at 8 fps (64x64 RGB)

  Summary: Text=4  Image=4  Video=4
```

CLI commands:

| Command | Description |
|---------|-------------|
| `/load <path.dstt>` | Load a trained model |
| `/info` | Show model metadata (parameters, system prompt, dimensions) |
| `/steps <n>` | Set generation steps |
| `/system <prompt>` | Set system prompt for next training run |
| `/params <n>` | Set target parameter count for next training run |
| `/save_images` | Toggle saving generated images as PPM files |
| `/save_video` | Toggle saving video frames as PPM files |
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
dstt::Config cfg;
cfg.system_prompt = "You are a creative multimedia assistant.";
cfg.set_parameter_count(500'000);  // 500K parameters

dstt::DSTTModel model(cfg);
auto examples = dstt::DSTTModel::load_training_jsonl("training_data/example.jsonl");
model.train_and_save(examples, "models/my_model.dstt",
    [](const dstt::EpochStats& s) {
        std::cout << "Epoch " << s.epoch << " loss=" << s.avg_loss << "\n";
    });
```

## Architecture

### Content Generation Pipeline

After training, the generation pipeline transforms evolved parameters into content:

```
Prompt ──→ Tokenizer ──→ FDMP ──→ EA ──→ ARM ──→ Branch Predictor
                                                        │
                         ┌──────────────────────────────┤
                         ▼              ▼               ▼
                   ContentGenerator  ContentGenerator  ContentGenerator
                     (Text)           (Image)           (Video)
                         │              │               │
                         ▼              ▼               ▼
                   Token sampling   Pixel synthesis   Frame sequence
                   from vocabulary  via sigmoid map   with temporal
                   embeddings       of parameters     modulation
```

**Text generation:** For each vocabulary token, compute `score(t) = 0.6·cos(E[t], P) + 0.4·cos(E[t], C)` where `E[t]` is the learned token embedding, `P` is the parameter embedding, and `C` is the context. Apply temperature-scaled softmax and sample. Repetition is reduced by decaying sampled token probabilities.

**Image generation:** For each pixel `(x, y)`, spatially hash into the parameter vector to get `θ[i], θ[j], θ[k]`, then compute RGB via `sigmoid(θ · context + spatial_phase)`. Spatial phase is modulated by `sin(2πu)` and `cos(2πv)` for coherent gradients.

**Video generation:** Temporally modulate parameters: `θ_t[j] = θ[j] · (1 + 0.3·sin(2πt + j·π/dim))`. Each frame is generated like an image using the modulated parameters, then blended with the previous frame: `pixel = 0.7·current + 0.3·previous`.

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
| **MGE** (Multimedia Generation Engine) | Orchestrates generation: branch prediction + synthesis + consistency |
| **Trainer** | Training loop: tokenizer building, FDMP weight updates, embedding updates |
| **ContentGenerator** | Transforms evolved parameters into text tokens, image pixels, and video frames |
| **DSTTModel** | .dstt file I/O, model loading, inference, training data loaders |

## Training

Training a DSTT model requires two key decisions, just like training an LLM:

1. **System prompt** — defines the model's training focus and behavior. Every training example is processed with this prompt prepended as context, biasing the learned weights toward the specified objective. This is analogous to an LLM system message.

2. **Parameter count** — determines model capacity. Higher parameter counts allow the model to learn more complex patterns but require more compute. DSTT computes total parameters as:
   ```
   Total = vocab_size × embed_dim + 3 × (param_dim × embed_dim + param_dim)
           \___ token embeddings ___/   \___ FDMP weights (3 modalities) ___/
   ```

### System Prompt

The system prompt steers training. It is prepended to every training example during tokenization, so the learned embeddings and FDMP weights encode the specified behavior:

```cpp
cfg.system_prompt = "You are an expert landscape photographer and nature writer.";
```

Examples:

| System Prompt | Effect |
|---------------|--------|
| `"Be a helpful assistant."` | General-purpose (default) |
| `"You are a creative storyteller."` | Text-focused generation |
| `"You are a visual artist specializing in landscapes."` | Image-focused generation |
| `"You are a cinematographer creating nature documentaries."` | Video-focused generation |

The system prompt is stored in the `.dstt` file and automatically applied at inference time.

### Parameter Count

Specify a target parameter count and DSTT derives the optimal `embed_dim` and `param_dim`:

```cpp
Config cfg;
cfg.system_prompt = "Be a helpful assistant.";
cfg.vocab_size    = 4096;

// Set target parameter count (like choosing 7B vs 13B for an LLM)
cfg.set_parameter_count(500'000);  // 500K parameters
// -> embed_dim = 105, param_dim = 52
// -> actual: 500,076 parameters

std::cout << cfg.total_parameters() << "\n";  // 500076
```

Example parameter scales:

| Target | embed_dim | param_dim | Actual | Use case |
|--------|-----------|-----------|--------|----------|
| 50K | 32 | 16 | ~50K | Quick experiments |
| 500K | 105 | 52 | ~500K | Small models |
| 5M | 328 | 164 | ~5M | Medium models |
| 50M | 1024 | 512 | ~50M | Large models |

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
cfg.system_prompt   = "You are a creative multimedia assistant.";
cfg.training_epochs = 10;
cfg.training_lr     = 0.005;
cfg.vocab_size      = 4096;
cfg.set_parameter_count(500'000);  // 500K parameters

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
| **Input** | System prompt + corpus of (text, modality) pairs | System prompt + user prompt |
| **Tokenizer** | Builds BPE vocabulary (includes system prompt) | Uses learned vocabulary |
| **System prompt** | Prepended to every example, shapes learned weights | Prepended to user prompt automatically |
| **FDMP weights** | Updated each step | Frozen |
| **EA** | Short runs (signal for weight updates) | Full runs (parameter optimization) |
| **Output** | Trained .dstt model file | Generated text, images, and video |

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

Five test suites: partitioning, CFM (Correct Flow Matrix)/AFM (Adversarial Flow Matrix), EA (Evolutionary Algorithm) operators, integration, and **training**.

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

The CLI is the primary way to interact with trained DSTT models. Type any text prompt and the model generates a mix of text, images, and video — the branch predictor automatically selects the appropriate modality at each generation step based on the prompt context.

## Project Structure

```
dstt-full/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── models/                    # Trained .dstt model files go here
│   └── .gitkeep
├── training_data/             # Training data files (JSONL, CSV, TXT)
│   ├── example.jsonl          # Example JSONL training data (24 examples)
│   ├── example.csv            # Example CSV training data
│   ├── example.txt            # Example plain text training data
│   └── README.md              # Training data format documentation
├── include/dstt/
│   ├── core/
│   │   ├── types.hpp          # Types, enums, Config (incl. training params)
│   │   ├── arm.hpp            # ARM: Autonomous Route Matrix
│   │   ├── partition.hpp      # Combinatorial partitioning engine
│   │   ├── cfm.hpp            # CFM: Correct Flow Matrix
│   │   ├── afm.hpp            # AFM: Adversarial Flow Matrix
│   │   └── softmax.hpp        # Softmax + parameter sampling
│   ├── ea/
│   │   ├── chromosome.hpp     # Genetic encoding
│   │   ├── operators.hpp      # Selection, crossover, mutation
│   │   ├── fitness.hpp        # Fitness evaluation
│   │   └── population.hpp     # Population manager + generational loop
│   ├── fdmp/
│   │   ├── fdmp.hpp           # FDMP: Fundamental Data Matrix Processor
│   │   ├── tokenizer.hpp      # BPE: Byte-Pair Encoding tokenizer + learned embeddings
│   │   └── embeddings.hpp     # Modality-specific embedding helpers
│   ├── mge/
│   │   ├── mge.hpp            # MGE: Multimedia Generation Engine
│   │   ├── branch_predictor.hpp
│   │   └── synthesiser.hpp    # Output synthesis
│   ├── model/
│   │   ├── dstt_format.hpp    # .dstt binary file format specification
│   │   ├── dstt_model.hpp     # Model loading, saving, and inference API
│   │   └── content_generator.hpp  # Text/image/video content synthesis
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
│   ├── model/       # DSTTModel, ContentGenerator (.dstt I/O + content synthesis)
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
| `system_prompt` | `"Be a helpful assistant."` | Training focus prompt (prepended to all examples) |
| `training_epochs` | 10 | Number of passes over the training corpus |
| `training_lr` | 0.005 | Learning rate for FDMP weight updates |
| `vocab_size` | 4096 | Maximum BPE vocabulary size |
| `min_token_freq` | 2 | Minimum pair frequency for BPE merges |
| `weight_decay` | 1e-4 | L2 regularization on weights |
| `set_parameter_count(n)` | — | Derives embed_dim and param_dim from target parameter count |

## License

MIT.
