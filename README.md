<div align="center">

# DSTT

### Dynamic Semi-Trained Topology

**A C++17 framework that generates text, images, and video from natural language prompts — without transformers, without attention, without GPUs.**

DSTT evolves parameters instead of computing gradients through layers. Train a model, give it a prompt, get multimedia back.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)](#building)

</div>

---

## The Idea in 30 Seconds

Most generative AI works like this: stack hundreds of layers, feed data through them, and backpropagate errors to adjust millions of weights. It works spectacularly — but it's one way to do things, not the only way.

DSTT takes a different path. Instead of learning what to compute at each layer, it **evolves the right parameters directly**. Think of it like this:

> Imagine you're tuning a radio. A neural network would learn the exact sequence of knob turns to reach every station. DSTT instead tries thousands of random dial positions, keeps the ones that sound good, breeds them together, and repeats — until it locks onto the signal.

That's the core idea. A trained DSTT model learns good *starting positions* for the dials, so evolution converges fast at inference time.

**What comes out the other end:**

| Modality | What You Get | Format |
|:--------:|:-------------|:-------|
| **Text** | Generated prose sampled from a learned vocabulary | UTF-8 string |
| **Image** | 64×64 RGB images with coherent color gradients | PPM files |
| **Video** | Temporally smooth frame sequences | PPM series |

A single prompt produces all three — a branch predictor decides which modality to generate at each step.

---

## Quick Start

```bash
# Clone and build (no dependencies beyond a C++17 compiler and CMake)
git clone https://github.com/darvyc/dstt-full.git
cd dstt-full
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Launch the interactive CLI
./dstt_cli
```

```
dstt> /system You are an expert landscape photographer and nature writer.
dstt> /params 100000
dstt> /train ../training_data/example.jsonl ../models/landscape.dstt

Training...
  Epoch 0  fitness=0.9989  loss=0.0011

dstt> A sunset over snow-capped mountains

Generating...
  [0] Text  -> "mountain peaks golden..."
  [1] Image -> 64x64 image
  [2] Video -> video frame
  [3] Text  -> "light fading..."

========================================================================
  Generation Result for: "A sunset over snow-capped mountains"
========================================================================

  [Text Output]  mountain peaks golden...light fading...
  [Image Output] 4 image(s) generated (64x64 RGB)
  [Video Output] 4 frame(s) at 8 fps (64x64 RGB)
```

That's it. You trained a model and generated multimedia from a prompt.

---

## How It Works

DSTT has three phases. Here's what actually happens in each one — no hand-waving.

### Phase 1: Training

You give DSTT a corpus of `(text, modality)` pairs. It does three things:

1. **Builds a vocabulary.** Using Byte-Pair Encoding (BPE), it starts with 256 raw byte tokens and iteratively merges the most frequent adjacent pairs until it reaches the target vocabulary size (default: 4096 tokens). This is the same tokenization approach used by GPT and friends.

2. **Learns embeddings.** Each token gets a dense vector (initialized randomly). When text comes in, DSTT tokenizes it, averages the token embeddings, and normalizes the result. That's the *context vector* — a single point in high-dimensional space that represents the input.

3. **Learns weight matrices.** The FDMP (Fundamental Data Matrix Processor) is a simple affine transform — one per modality — that maps context vectors to parameter vectors: `Θ = W·C + b`. During training, the EA runs a short optimization, and the fitness signal drives updates to `W`, `b`, and the token embeddings. This is conceptually similar to backpropagation, but the "gradient" comes from evolutionary fitness rather than chain-rule derivatives.

After training, you have a `.dstt` file containing the vocabulary, embeddings, and weight matrices. Everything needed to generate content.

### Phase 2: Parameter Evolution (Inference)

When you give a trained model a prompt:

1. The prompt is tokenized and embedded into a context vector `C`
2. The FDMP generates a *starting point* in parameter space (using trained weights)
3. The Evolutionary Algorithm refines these parameters over ~500 generations:
   - **Tournament selection** picks parents
   - **Crossover** combines their parameters
   - **Mutation** introduces variation
   - **Fitness** evaluates each candidate: 40% coherence + 40% relevance + 20% diversity

Because the FDMP was trained, the EA starts near a good solution. Evolution fine-tunes rather than searching from scratch.

### Phase 3: Content Generation

A **branch predictor** decides: text, image, or video? Then the evolved parameters become content:

- **Text** — Each vocabulary token's embedding is compared to the evolved parameters. Scores go through temperature-scaled softmax, and tokens are sampled. Repetition penalty keeps output varied.
- **Image** — Parameters map to RGB pixels via `sigmoid(θ · context + spatial_phase)`. Sine and cosine spatial modulation produces coherent gradients rather than noise.
- **Video** — Image generation plus temporal modulation: `θ_t = θ · (1 + 0.3·sin(2πt + phase))`. Adjacent frames blend 70/30 for smooth motion.

```
Prompt → Tokenizer → FDMP → EA → ARM → Branch Predictor
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                     ▼                     ▼
                   Text Sampler         Pixel Synthesizer     Frame Sequencer
                   (vocabulary +        (sigmoid map +        (temporal modulation +
                    softmax)             spatial phase)        inter-frame blend)
```

---

## The `.dstt` Model Format

Trained models are saved as `.dstt` files — self-contained binaries (like `.gguf` for llama.cpp). One file, everything included:

| Section | What's Inside |
|:--------|:-------------|
| Header | Magic bytes `DSTT`, version, dimensions, parameter count |
| Config | All hyperparameters from training |
| System Prompt | The behavior-steering prompt used during training |
| Vocabulary | Full BPE token table (strings + IDs) |
| Embeddings | Token embedding matrix (`vocab_size × embed_dim`) |
| FDMP Weights | Per-modality `W` matrices and bias vectors (3 sets) |

---

## Usage

### C++ API

```cpp
#include <dstt/model/dstt_model.hpp>

// Load and generate
dstt::DSTTModel model;
model.load("models/my_model.dstt");

auto result = model.run("A sunset over mountains", 12,
    [](size_t step, const dstt::GeneratedContent& c) {
        std::cout << step << ": " << dstt::modality_name(c.modality);
        if (c.modality == dstt::Modality::Text)
            std::cout << " -> \"" << c.text << "\"";
        std::cout << "\n";
    });

std::cout << result.generated_text << "\n";
result.images[0].save_ppm("output.ppm");
result.video.save_frames("output_video");
```

### Train Programmatically

```cpp
dstt::Config cfg;
cfg.system_prompt = "You are a creative multimedia assistant.";
cfg.set_parameter_count(500'000);  // auto-derives embed_dim and param_dim

dstt::DSTTModel model(cfg);
auto examples = dstt::DSTTModel::load_training_jsonl("training_data/example.jsonl");
model.train_and_save(examples, "models/my_model.dstt",
    [](const dstt::EpochStats& s) {
        std::cout << "Epoch " << s.epoch << " loss=" << s.avg_loss << "\n";
    });
```

### CLI Commands

| Command | Description |
|:--------|:-----------|
| `/load <path>` | Load a trained `.dstt` model |
| `/info` | Show model metadata |
| `/steps <n>` | Set generation steps |
| `/system <prompt>` | Set system prompt for training |
| `/params <n>` | Set target parameter count |
| `/save_images` | Toggle image saving to disk |
| `/save_video` | Toggle video frame saving |
| `/train <data> <out>` | Train from data file, save model |
| `/help` | Show help |
| `/quit` | Exit |

---

## Training

Two decisions matter, just like with LLMs:

### 1. System Prompt

The system prompt is prepended to every training example. It biases what the model learns:

| System Prompt | Effect |
|:-------------|:-------|
| `"Be a helpful assistant."` | General-purpose (default) |
| `"You are a creative storyteller."` | Text-focused |
| `"You are a visual artist specializing in landscapes."` | Image-focused |
| `"You are a cinematographer creating nature documentaries."` | Video-focused |

### 2. Parameter Count

More parameters = more capacity. DSTT auto-derives the right dimensions:

```cpp
cfg.set_parameter_count(500'000);
// -> embed_dim=105, param_dim=52, actual=500,076 parameters
```

| Target | embed_dim | param_dim | Use Case |
|-------:|:---------:|:---------:|:---------|
| 50K | 32 | 16 | Quick experiments |
| 500K | 105 | 52 | Small models |
| 5M | 328 | 164 | Medium models |
| 50M | 1024 | 512 | Large models |

Where the parameters live:

```
Total = vocab_size × embed_dim  +  3 × (param_dim × embed_dim + param_dim)
        └── token embeddings ──┘     └── FDMP weights (one set per modality) ──┘
```

### Training Data Formats

**JSONL** (recommended):
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

### What Gets Learned

| Component | Shape | Update Method |
|:----------|:------|:-------------|
| Token embeddings `E` | `vocab_size × embed_dim` | Fitness-driven gradient |
| FDMP weights `W_m` | `param_dim × embed_dim` (×3 modalities) | Sign-based gradient approximation + L2 decay |
| FDMP biases `b_m` | `param_dim` (×3 modalities) | Sign-based gradient approximation + L2 decay |

### Training vs. Inference

| | Training | Inference |
|:--|:--------|:----------|
| **Tokenizer** | Builds BPE vocabulary from corpus | Uses the learned vocabulary |
| **FDMP weights** | Updated each step | Frozen |
| **EA runs** | Short (~20 generations) — just enough for a fitness signal | Full (~500 generations) — thorough optimization |
| **Output** | A `.dstt` model file | Text + images + video |

---

## Architecture Deep Dive

For those who want the full picture. Every formula below is implemented — no pseudocode hand-waving.

### Subsystem Map

| Subsystem | What It Does |
|:----------|:------------|
| **Tokenizer** | BPE vocabulary learning, sub-word encoding, learned embeddings |
| **FDMP** | Trained per-modality affine transforms: context → parameters |
| **ARM** | Partitions parameters, scores via CFM/AFM, adjusts and samples |
| **EA** | Evolves parameter vectors across generations |
| **MGE** | Orchestrates generation: branch prediction + synthesis + consistency |
| **ContentGenerator** | Transforms parameters into text, pixels, and frames |
| **DSTTModel** | Public API: `.dstt` file I/O, training, inference |

### BPE Tokenizer

Starts with 256 byte-level tokens. Repeatedly merges the most frequent adjacent pair until `vocab_size` is reached or no pair exceeds `min_token_freq`.

Embeddings are initialized Xavier-uniform:

```
limit = √(6 / (vocab_size + embed_dim))
E[i, d] ~ Uniform(-limit, limit)
```

Context vector = normalized mean of token embeddings:

```
C = normalize( (1/|tokens|) · Σᵢ E[tokenᵢ] )
```

### FDMP

One linear layer per modality. Xavier-initialized weights, small random biases:

```
Θ_raw = W_m · C + b_m       W_m ∈ ℝ^{param_dim × embed_dim}
```

### Partition Engine

Groups parameters into coherent clusters using Ramsey-theoretic similarity:

```
R_c(θᵢ, θⱼ) = 1 - hamming_distance(embed(θᵢ), embed(θⱼ)) / embed_dim
```

Pairs with `R_c > threshold` are merged via Union-Find. Partition count is bounded by the Hardy-Ramanujan estimate: `k = ⌊log₂(p(m))⌋`.

### Dual Flow Matrices

| Matrix | Purpose | Formula |
|:-------|:--------|:--------|
| **CFM** (Correct Flow) | Rewards coherent parameters | `weight · cos(embed(θⱼ), context) + ramsey_coherence` |
| **AFM** (Adversarial Flow) | Penalizes incoherent parameters | `max(0, -cos(...)) + shannon_entropy` |

Parameters are adjusted: `θ'ⱼ = θⱼ · (CFMⱼ - AFMⱼ)`, then softmax-sampled.

### Evolutionary Algorithm

```
Fitness = 0.4 · Coherence + 0.4 · Relevance + 0.2 · Diversity
```

- **Coherence** — fraction of probability mass where CFM > AFM
- **Relevance** — cosine similarity of weighted parameters to context
- **Diversity** — normalized Shannon entropy of the distribution

Tournament selection (k=5), single-point crossover, per-gene mutation (rate = 1/dim), 10% elitism. Early stopping if best fitness is unchanged for 20 generations.

### Weight Updates (Training)

FDMP weights use sign-based gradient approximation with L2 decay:

```
W_m[i,j] ← W_m[i,j] - lr · (loss · sign(θ[i]) · C[j]) - wd · W_m[i,j]
```

Token embeddings update 10× slower (loss-scaled L2 regularization):

```
E[t, d] ← E[t, d] - (lr · 0.1) · (loss / |tokens|) · E[t, d]
```

### Content Synthesis

**Text:** `score(t) = 0.6·cos(E[t], P) + 0.4·cos(E[t], C)` → temperature softmax → sample with repetition decay.

**Image:** For pixel `(x, y)`: `RGB = sigmoid(θ · context + sin(2πu) + cos(2πv))` where `u, v` are normalized coordinates.

**Video:** `θ_t[j] = θ[j] · (1 + 0.3·sin(2πt + j·π/dim))`, then blend: `pixel = 0.7·current + 0.3·previous`.

---

## Building

### Requirements

- C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 19.20)
- CMake ≥ 3.16
- No external libraries. Zero. The standard library is all you need.

### Build

```bash
git clone https://github.com/darvyc/dstt-full.git
cd dstt-full
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Test

```bash
cd build && ctest --output-on-failure
```

Five test suites covering the full pipeline:

| Suite | Tests |
|:------|:------|
| `test_partition` | Union-Find correctness |
| `test_cfm_afm` | CFM/AFM scoring and ARM evaluation |
| `test_ea` | Selection, crossover, mutation, fitness |
| `test_integration` | FDMP encoding, ARM full pass, branch predictor, synthesizer |
| `test_training` | BPE vocab building, encode/decode, training convergence |

### Run the Demo

```bash
./dstt_demo
```

Trains on a small corpus, then generates multi-modal output for three prompts — the full pipeline in one command.

---

## Configuration Reference

All parameters live in `Config` (`include/dstt/core/types.hpp`) and can be set at runtime.

<details>
<summary><strong>Evolutionary Algorithm</strong></summary>

| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `population_size` | 200 | Number of candidates per generation |
| `max_generations` | 500 | Maximum EA iterations |
| `tournament_k` | 5 | Tournament selection size |
| `mutation_rate` | `1/param_dim` | Per-gene mutation probability |
| `elitism_rate` | 0.10 | Fraction of top candidates preserved |
| `convergence_window` | 20 | Early stop after N stagnant generations |

</details>

<details>
<summary><strong>Fitness Evaluation</strong></summary>

| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `w_coherence` | 0.4 | Weight for coherence metric |
| `w_relevance` | 0.4 | Weight for relevance metric |
| `w_diversity` | 0.2 | Weight for diversity metric |
| `coherence_threshold` | 0.25 | Ramsey coherence threshold |

</details>

<details>
<summary><strong>Training</strong></summary>

| Parameter | Default | Description |
|:----------|:--------|:-----------|
| `system_prompt` | `"Be a helpful assistant."` | Behavior-steering prompt |
| `training_epochs` | 10 | Passes over the training corpus |
| `training_lr` | 0.005 | Learning rate for FDMP weights |
| `weight_decay` | 1e-4 | L2 regularization strength |
| `vocab_size` | 4096 | Maximum BPE vocabulary size |
| `min_token_freq` | 2 | Minimum pair frequency for BPE merges |
| `set_parameter_count(n)` | — | Auto-derive dimensions from target |

</details>

---

## Project Structure

```
dstt-full/
├── include/dstt/
│   ├── core/               # ARM, partitioning, CFM, AFM, softmax, types
│   ├── ea/                 # Chromosome, operators, fitness, population
│   ├── fdmp/               # FDMP, BPE tokenizer, embedding helpers
│   ├── mge/                # Generation engine, branch predictor, synthesizer
│   ├── model/              # DSTTModel API, .dstt format, content generator
│   ├── training/           # Training loop
│   └── utils/              # Math primitives, RNG, timer
├── src/                    # Implementation files (~2,300 lines)
├── examples/
│   ├── demo.cpp            # Full pipeline demo
│   └── dstt_cli.cpp        # Interactive CLI
├── tests/                  # 5 test suites
├── training_data/          # Example JSONL, CSV, and TXT data
├── models/                 # Trained .dstt files go here
├── CMakeLists.txt
└── LICENSE                 # MIT
```

---

## License

[MIT](LICENSE) — use it however you want.
