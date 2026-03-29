<div align="center">

# DSTT

### Dynamic Semi-Trained Topology

**A new kind of AI that generates text, images, and video — without transformers.**

Pure C++17 &middot; Zero dependencies &middot; Train your own models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](CMakeLists.txt)

</div>

---

## The Idea in 30 Seconds

Most AI systems that generate content use transformers — giant networks of attention layers trained on massive datasets. DSTT takes a completely different path.

Imagine you're tuning a radio. Instead of searching every frequency one by one (like a transformer scanning every token), DSTT **evolves** the best frequency. It breeds a population of candidate settings, keeps the ones that sound good, crosses them together, and repeats — until the signal is clear.

That's the core insight: **evolutionary optimization instead of attention**. DSTT trains a lightweight set of weight matrices, then at inference time, evolves the best parameters for your specific prompt using a genetic algorithm. Those parameters then drive text generation, image synthesis, and video creation — all from a single model.

```
Your prompt ──→ Trained weights generate parameters
                        │
                        ▼
              Evolutionary Algorithm breeds
              the best parameter set for YOUR prompt
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
            Text      Image     Video
```

## What It Generates

| Modality | Output | How It Works |
|:---------|:-------|:-------------|
| **Text** | Generated prose from learned vocabulary | Token embeddings scored against evolved parameters via cosine similarity, then sampled |
| **Image** | 64x64 RGB images (PPM) | Parameters mapped to pixel colors through sigmoid activation with spatial phase shifts |
| **Video** | Temporally coherent frame sequences (PPM) | Image generation extended with time-varying modulation and inter-frame blending |

A built-in **branch predictor** automatically decides which modality to produce at each generation step — so a single prompt can yield a mix of text, images, and video.

---

## Quick Start

### Build

```bash
git clone https://github.com/darvyc/dstt.git
cd dstt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Requirements:** A C++17 compiler (GCC 9+, Clang 10+, MSVC 19.20+) and CMake 3.16+. That's it — no external libraries needed.

### Train & Generate (Interactive CLI)

```bash
./dstt_cli
```

```
dstt> /system You are an expert landscape photographer and nature writer.
dstt> /params 100000
dstt> /train ../training_data/example.jsonl ../models/landscape.dstt

System prompt: "You are an expert landscape photographer and nature writer."
Parameters:    99429
Training...
  Epoch 0  fitness=0.9989  loss=0.0011

dstt> A sunset over snow-capped mountains
  [0] Text  -> "mountain sunset..."
  [1] Image -> 64x64 image
  [2] Video -> video frame
  ...

dstt> /save_images
dstt> /save_video
```

### Train & Generate (C++ API)

```cpp
#include <dstt/model/dstt_model.hpp>

// Train a model
dstt::Config cfg;
cfg.system_prompt = "You are a creative multimedia assistant.";
cfg.set_parameter_count(500'000);

dstt::DSTTModel model(cfg);
auto examples = dstt::DSTTModel::load_training_jsonl("training_data/data.jsonl");
model.train_and_save(examples, "models/my_model.dstt",
    [](const dstt::EpochStats& s) {
        std::cout << "Epoch " << s.epoch << " loss=" << s.avg_loss << "\n";
    });

// Load and generate
dstt::DSTTModel model;
model.load("models/my_model.dstt");

auto result = model.run("A sunset over mountains", 12,
    [](size_t step, const dstt::GeneratedContent& c) {
        std::cout << step << ": " << dstt::modality_name(c.modality) << "\n";
    });

std::cout << result.generated_text << "\n";
result.images[0].save_ppm("output.ppm");
result.video.save_frames("output_video");
```

---

## How DSTT Actually Works

Think of DSTT as a three-act play. Each act builds on the last.

### Act 1 — Training: Learning the Territory

Before DSTT can generate anything, it needs to learn from examples — just like any AI.

You feed it training data: pairs of text prompts and target modalities ("A sunset over mountains" → Image). DSTT does three things:

1. **Builds a vocabulary** using Byte-Pair Encoding (BPE) — the same tokenization strategy used by GPT. It starts with 256 raw bytes, then repeatedly merges the most common adjacent pairs until it has a rich sub-word vocabulary.

2. **Learns token embeddings** — each vocabulary token gets a dense vector (like word2vec). These start random (Xavier initialization) and are refined each training step.

3. **Trains the FDMP weight matrices** — for each modality (text, image, video), a weight matrix learns to map from "what the prompt means" (the context embedding) to "what parameters would generate good content." The fitness from a short evolutionary run serves as the training signal.

The key insight: training doesn't need backpropagation through a deep network. A sign-based gradient approximation from the evolutionary fitness is enough to update the weights.

### Act 2 — Evolution: Finding the Best Parameters for Your Prompt

At inference time, your prompt flows through the trained FDMP to get a starting set of parameters. But these are just a rough draft.

Now the Evolutionary Algorithm takes over:

1. **Spawn a population** of candidate parameter vectors (default: 200)
2. **Score each one** using dual flow matrices:
   - The **CFM** (Correct Flow Matrix) rewards parameters that are contextually coherent
   - The **AFM** (Adversarial Flow Matrix) penalizes contradictions and low-information parameters
3. **Select the fittest** via tournament selection
4. **Crossover and mutate** to create the next generation
5. **Repeat** for up to 500 generations (or until convergence)

Because the FDMP was trained, the starting parameters are already in the right neighborhood — evolution converges much faster than it would from random initialization.

### Act 3 — Generation: Turning Parameters into Content

The optimized parameters are now handed to the content generators:

- **Text:** Score every vocabulary token by how well its embedding matches the evolved parameters (cosine similarity), apply temperature-controlled softmax, and sample. Repetition penalty keeps output diverse.

- **Image:** For each pixel, hash spatial coordinates into the parameter vector, multiply by the context, add spatial phase shifts (`sin`, `cos` for smooth gradients), and push through a sigmoid. The result: coherent, prompt-influenced color fields.

- **Video:** Same as image generation, but parameters oscillate over time: `θ_t = θ · (1 + 0.3·sin(2πt + phase))`. Adjacent frames are blended (70/30) for smooth motion.

A **branch predictor** — a simple softmax classifier — decides whether each generation step should produce text, an image, or a video frame. If it's not confident, it falls back to whichever modality hasn't been used recently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DSTT Architecture                                │
│                                                                         │
│  Prompt ──→ BPE Tokenizer ──→ Token Embeddings ──→ Context Vector (C)  │
│                                                          │              │
│                                                          ▼              │
│                                                  ┌──────────────┐      │
│                                                  │     FDMP     │      │
│                                                  │  W·C + b → Θ │      │
│                                                  └──────┬───────┘      │
│                                                         │              │
│                              ┌───────────────────────────┘              │
│                              ▼                                          │
│                   ┌─────────────────────┐                               │
│                   │ Evolutionary Algorithm│                              │
│                   │  Population → Score  │                               │
│                   │  Select → Crossover  │                               │
│                   │  Mutate → Repeat     │                               │
│                   └─────────┬───────────┘                               │
│                             │                                           │
│                     ARM (evaluate + sample)                             │
│                             │                                           │
│                   Branch Predictor                                      │
│                    ╱        │        ╲                                   │
│                   ▼         ▼         ▼                                  │
│               ┌───────┐ ┌───────┐ ┌───────┐                            │
│               │ Text  │ │ Image │ │ Video │                            │
│               │ Gen   │ │ Gen   │ │ Gen   │                            │
│               └───────┘ └───────┘ └───────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Subsystems

| Subsystem | What It Does |
|:----------|:-------------|
| **Tokenizer** | BPE vocabulary learning, sub-word encoding, learned token embeddings |
| **FDMP** | Trained weight matrices that project context embeddings into parameter space |
| **ARM** | Partitions parameters, scores them via CFM/AFM, adjusts, and samples |
| **CFM / AFM** | Dual scoring — CFM promotes coherence, AFM penalizes contradiction |
| **EA** | Evolutionary optimizer: tournament selection, crossover, mutation |
| **MGE** | Multimedia Generation Engine: orchestrates branch prediction + synthesis |
| **ContentGenerator** | Transforms evolved parameters into text, images, and video |
| **DSTTModel** | `.dstt` file I/O, model loading, inference API |

---

## Training

Training a DSTT model requires two choices:

### 1. System Prompt

Defines the model's personality — prepended to every training example so the learned weights encode your specified behavior.

```cpp
cfg.system_prompt = "You are a visual artist specializing in landscapes.";
```

| System Prompt | Effect |
|:--------------|:-------|
| `"Be a helpful assistant."` | General-purpose (default) |
| `"You are a creative storyteller."` | Text-focused generation |
| `"You are a visual artist specializing in landscapes."` | Image-focused generation |
| `"You are a cinematographer creating nature documentaries."` | Video-focused generation |

### 2. Parameter Count

Controls model capacity. DSTT automatically derives the right embedding and parameter dimensions:

```cpp
cfg.set_parameter_count(500'000);  // → embed_dim=105, param_dim=52
```

| Scale | embed_dim | param_dim | Actual Params | Use Case |
|:------|:---------:|:---------:|:-------------:|:---------|
| 50K | 32 | 16 | ~50K | Quick experiments |
| 500K | 105 | 52 | ~500K | Small models |
| 5M | 328 | 164 | ~5M | Medium models |
| 50M | 1024 | 512 | ~50M | Large models |

**Where the parameters live:**

```
Total = (vocab_size × embed_dim) + 3 × (param_dim × embed_dim + param_dim)
         └── token embeddings ──┘     └── FDMP weights per modality ──┘
```

### Training Data Formats

**JSONL** (recommended):
```jsonl
{"input": "A sunset over mountains", "modality": "Image"}
{"input": "Breaking news about science", "modality": "Text"}
{"input": "Time-lapse of a flower blooming", "modality": "Video"}
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

### What Gets Trained

| Component | What It Learns | How It's Updated |
|:----------|:---------------|:-----------------|
| Token embeddings `E` | A dense vector per sub-word token | Gradient from fitness signal |
| FDMP weights `W_m` | Per-modality projection matrices | Sign-based gradient approximation |
| FDMP biases `b_m` | Per-modality bias vectors | Sign-based gradient approximation |

---

## Model Files (`.dstt`)

Trained models are saved as self-contained `.dstt` binary files — analogous to `.gguf` for llama.cpp models. Everything needed for inference is in one file:

| Section | Contents |
|:--------|:---------|
| Header | Magic bytes (`DSTT`), version, dimensions, parameter count |
| Config | All hyperparameters from training |
| System Prompt | The steering prompt (auto-applied at inference) |
| Vocabulary | BPE token table |
| Embeddings | Token embedding matrix |
| FDMP Weights | Per-modality weight matrices and bias vectors |

```cpp
// Load and inspect
dstt::DSTTModel model;
model.load("models/my_model.dstt");
std::cout << model.info() << "\n";  // prints metadata
```

---

## CLI Reference

| Command | Description |
|:--------|:------------|
| `/load <path.dstt>` | Load a trained model |
| `/info` | Show model metadata |
| `/steps <n>` | Set generation steps (default: 12) |
| `/system <prompt>` | Set system prompt for next training |
| `/params <n>` | Set target parameter count |
| `/train <data> <out>` | Train from file, save as `.dstt` |
| `/save_images` | Toggle saving images as PPM files |
| `/save_video` | Toggle saving video frames |
| `/help` | Show help |
| `/quit` | Exit |
| `<any text>` | Generate multi-modal content from prompt |

---

## Configuration

All hyperparameters can be set in code or through the CLI.

<details>
<summary><b>Evolutionary Algorithm</b></summary>

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `population_size` | 200 | Number of candidate solutions per generation |
| `max_generations` | 500 | Maximum evolutionary iterations |
| `tournament_k` | 5 | Tournament selection pool size |
| `mutation_rate` | 1/m | Per-gene mutation probability (auto-scaled) |
| `elitism_rate` | 0.10 | Fraction of top performers preserved |

</details>

<details>
<summary><b>Fitness & Evaluation</b></summary>

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `w_coherence` | 0.4 | Weight: contextual coherence |
| `w_relevance` | 0.4 | Weight: prompt relevance |
| `w_diversity` | 0.2 | Weight: output diversity |
| `coherence_threshold` | 0.25 | Ramsey coherence threshold (τ) |
| `bp_confidence_threshold` | 0.6 | Branch predictor confidence cutoff |

</details>

<details>
<summary><b>Training</b></summary>

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `training_epochs` | 10 | Passes over the training corpus |
| `training_lr` | 0.005 | FDMP weight learning rate |
| `vocab_size` | 4096 | BPE vocabulary size |
| `min_token_freq` | 2 | Minimum pair frequency for BPE merges |
| `weight_decay` | 1e-4 | L2 regularization |
| `embed_dim` | 128 | Token embedding dimension |
| `param_dim` | 64 | Parameter vector dimension |

</details>

---

## Mathematical Foundations

<details>
<summary><b>Core Operations</b></summary>

| Operation | Definition |
|:----------|:-----------|
| Dot product | `dot(a, b) = Σᵢ aᵢ · bᵢ` |
| L2 norm | `‖a‖ = √(Σᵢ aᵢ²)` |
| Cosine similarity | `cos(a, b) = dot(a, b) / (‖a‖ · ‖b‖)` |
| Softmax | `softmax(v)ᵢ = exp(vᵢ - max(v)) / Σⱼ exp(vⱼ - max(v))` |
| Affine transform | `y = W · x + b` |
| Inverse-CDF sampling | `argmin{j : Σᵢ₌₀ʲ pᵢ ≥ u}` where `u ~ Uniform(0,1)` |

</details>

<details>
<summary><b>Fitness Function</b></summary>

```
F = w_c · Coherence + w_r · Relevance + w_d · Diversity
```

- **Coherence** — probability mass on parameters where CFM > AFM: `Σⱼ P[j] · 𝟙{CFM[j] > AFM[j]}`
- **Relevance** — cosine similarity of weighted parameters to context, mapped to [0,1]
- **Diversity** — normalized Shannon entropy: `-Σⱼ P[j] · log₂(P[j]) / log₂(m)`

</details>

<details>
<summary><b>CFM & AFM Scoring</b></summary>

**CFM** (rewards coherence):
```
CFM(θⱼ) = α_m · cos(embed(θⱼ), C) + (1 - hamming(embed(θⱼ), S_prev) / dim)
```

| Modality | α_m |
|:---------|:----|
| Text | 0.85 |
| Image | 0.80 |
| Video | 0.75 |

**AFM** (penalizes incoherence):
```
AFM(θⱼ) = max(0, -cos(embed(θⱼ), C)) + H(P(θⱼ))
```

Parameters are adjusted: `θ'ⱼ = θⱼ · (CFMⱼ - AFMⱼ)`, then softmax → sample.

</details>

<details>
<summary><b>ARM Pipeline</b></summary>

```
1. Partition Θ into coherent groups (Union-Find, threshold τ)
2. For each parameter j:
     Compute CFM[j] and AFM[j]
3. Adjust: Θ'[j] = θⱼ · (CFM[j] - AFM[j])
4. P = softmax(Θ')
5. Sample j* ~ P via inverse-CDF
```

</details>

<details>
<summary><b>EA Operators</b></summary>

- **Tournament selection:** Pick k random individuals, return the fittest
- **Single-point crossover:** Split at random point, swap tails
- **Random-reset mutation:** Each gene has `μ = 1/param_dim` chance of resampling from `U(0,1)`
- **Elitism:** Top 10% copied unchanged to next generation
- **Convergence:** Stop if best fitness stalls for 20 generations (ε = 10⁻⁶)

</details>

<details>
<summary><b>Content Generation Algorithms</b></summary>

**Text:**
```
score(t) = (0.6 · cos(E[t], param_embed) + 0.4 · cos(E[t], C)) / T
probs = softmax(scores)
sample tokens, apply 0.3× repetition decay, renormalize
```

**Image (per pixel):**
```
Spatial hash (x,y) → parameter indices (i,j,k)
Phase: sin(2πu)·1.5, cos(2πv)·1.5, sin((u+v)π)·1.5
RGB = sigmoid(θ · context · 4 + phase)
```

**Video (per frame f):**
```
Temporal modulation: θ_t[j] = θ[j] · (1 + 0.3·sin(2πf·0.1 + j·π/m))
Generate frame with modulated parameters
Blend: pixel = 0.7·current + 0.3·previous
```

</details>

<details>
<summary><b>Training Weight Updates</b></summary>

**FDMP weights** (sign-based gradient + L2 decay):
```
b[i]   ← b[i]   - lr · loss · sign(θ[i]) - wd · b[i]
W[i,j] ← W[i,j] - lr · loss · sign(θ[i]) · C[j] - wd · W[i,j]
```

**Token embeddings** (loss-scaled regularization at 0.1× learning rate):
```
E[t,d] ← E[t,d] - (lr · 0.1) · (loss / |tokens|) · E[t,d]
```

</details>

---

## Running Tests

```bash
cd build
ctest --output-on-failure
```

Five test suites covering the full pipeline:

| Suite | Coverage |
|:------|:---------|
| `test_partition` | Union-Find, partitioning engine, partition counting |
| `test_cfm_afm` | Flow matrix scoring, edge cases |
| `test_ea` | Tournament selection, crossover, mutation, convergence |
| `test_integration` | Full pipeline: encode → FDMP → ARM → EA → synthesis |
| `test_training` | Training loop, weight updates, tokenizer building |

---

## Project Structure

```
dstt/
├── CMakeLists.txt              # Build config (CMake 3.16+, C++17)
├── LICENSE                     # MIT
├── README.md
│
├── include/dstt/               # Public headers
│   ├── core/                   #   ARM, CFM, AFM, partitioning, types
│   ├── ea/                     #   Evolutionary algorithm
│   ├── fdmp/                   #   FDMP, BPE tokenizer, embeddings
│   ├── mge/                    #   Multimedia generation engine
│   ├── model/                  #   .dstt format, model API, content gen
│   ├── training/               #   Training loop
│   └── utils/                  #   Math, RNG, timer
│
├── src/                        # Implementation (.cpp)
│   └── (mirrors include/)
│
├── examples/
│   ├── demo.cpp                # End-to-end demo (train → generate)
│   └── dstt_cli.cpp            # Interactive CLI
│
├── tests/                      # Test suite (5 modules)
├── training_data/              # Example training data (JSONL, CSV, TXT)
└── models/                     # Trained .dstt files go here
```

---

## Why DSTT?

| | Transformers (GPT, etc.) | DSTT |
|:---|:---|:---|
| **Core mechanism** | Attention over token sequences | Evolutionary optimization of parameter vectors |
| **Training signal** | Cross-entropy loss + backprop | Fitness-driven sign-gradient approximation |
| **Multi-modal** | Separate models or adapters | Single model, built-in branch predictor |
| **Dependencies** | PyTorch, CUDA, etc. | None. Pure C++17 standard library |
| **Model format** | `.pt`, `.safetensors`, `.gguf` | `.dstt` (self-contained binary) |
| **Parameter scale** | Billions | Thousands to millions |

DSTT is not a replacement for transformers. It's an exploration of a fundamentally different architecture — one where evolution, not gradient descent through deep layers, finds the right parameters for each prompt. It's lightweight, portable, dependency-free, and easy to understand.

---

## License

[MIT](LICENSE) — use it however you want.
