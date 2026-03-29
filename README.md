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

## Mathematical Foundations

This section provides the complete mathematical and algorithmic specification behind every DSTT subsystem.

### Linear Algebra Primitives

DSTT builds on a small set of core operations used throughout the pipeline.

**Dot product:**

```
dot(a, b) = Σᵢ aᵢ · bᵢ
```

**L2 norm:**

```
‖a‖ = √(Σᵢ aᵢ²)
```

**Cosine similarity:**

```
cos_sim(a, b) = dot(a, b) / (‖a‖ · ‖b‖)
```

Returns 0 when either norm is below `1e-12`. Range: `[-1, 1]`.

**Numerically stable softmax:**

```
softmax(v)ᵢ = exp(vᵢ - max(v)) / Σⱼ exp(vⱼ - max(v))
```

**Affine transform (linear layer):**

```
y = W · x + b     where W ∈ ℝ^{rows × cols}, x ∈ ℝ^{cols}, b ∈ ℝ^{rows}
```

**Shannon self-information:**

```
H(p) = -p · log₂(p)     (returns 0 for p ≤ 0)
```

**Hamming distance** (binary-thresholded vectors):

```
d_H(a, b) = Σᵢ 𝟙{sign(aᵢ) ≠ sign(bᵢ)}
```

Each element is thresholded at 0 to produce a binary representation before comparison.

**Inverse-CDF sampling:**

```
Given probabilities P = [p₀, p₁, ..., pₙ₋₁] and u ~ Uniform(0, 1):
  return argmin{j : Σᵢ₌₀ʲ pᵢ ≥ u}
```

**Hardy-Ramanujan partition count** (used by the partitioning engine):

```
k = ⌊log₂(p(m))⌋     where log₂(p(m)) ≈ π√(2m/3) / ln(2) - log₂(4m√3)
```

At least 2 partitions are always returned.

---

### BPE Tokenizer

The tokenizer converts raw text into token IDs and dense context vectors.

**Vocabulary construction (Byte-Pair Encoding):**

1. Initialize vocabulary with 256 byte-level tokens
2. Repeat until `vocab_size` is reached:
   - Count all adjacent pair frequencies across the corpus
   - Find the most frequent pair `(a, b)`
   - If frequency < `min_token_freq`, stop
   - Register merged token `m = concat(a, b)` and apply the merge across all sequences

**Embedding initialization (Xavier/Glorot uniform):**

```
limit = √(6 / (vocab_size + embed_dim))
E[i, d] ~ Uniform(-limit, limit)
```

`E ∈ ℝ^{vocab_size × embed_dim}` is a learnable matrix updated during training.

**Context vector computation:**

```
C = (1 / |tokens|) · Σᵢ E[tokenᵢ]
C ← C / ‖C‖
```

The context vector is the L2-normalized mean of token embeddings.

---

### Modality-Specific Embedding Helpers

Synthetic embeddings are generated via an xorshift hash when a deterministic modality-specific vector is needed:

```
hash_embed(input, dim, salt):
  h ← hash(input) ⊕ salt
  for d = 0 to dim-1:
    h ← h ⊕ (h << 13)
    h ← h ⊕ (h >> 7)
    h ← h ⊕ (h << 17)
    v[d] ← (h & 0xFFFF) / 32768 - 1          // map to [-1, 1]
  return v / ‖v‖                               // L2 normalize
```

| Modality | Salt |
|----------|------|
| Text | `0xCAFE'BABE'0001` |
| Image | `0xDEAD'BEEF'0002` |
| Video | `0xFEED'FACE'0003` |

---

### FDMP (Fundamental Data Matrix Processor)

The FDMP maps context embeddings to raw parameter vectors via a per-modality linear layer.

**Weight initialization (Xavier uniform, per modality `m`):**

```
limit = √(6 / (param_dim + embed_dim))
W_m[i, j] ~ Uniform(-limit, limit)       W_m ∈ ℝ^{param_dim × embed_dim}
b_m[i]    ~ Uniform(-0.01, 0.01)         b_m ∈ ℝ^{param_dim}
```

**Parameter generation (forward pass):**

```
C    = encode(input, modality)            // context embedding ∈ ℝ^{embed_dim}
Θ_r  = W_m · C + b_m                     // raw parameters ∈ ℝ^{param_dim}
```

This is a standard affine transform — the trained analogue of a single dense layer mapping from context space to parameter space.

---

### Partition Engine

The partition engine groups parameters into coherent clusters using a Ramsey-theoretic measure.

**Parameter embedding:**

```
For parameter θⱼ and context C:
  e[d] = θⱼ · C[d mod |C|]               (d = 0, ..., embed_dim - 1)
```

This creates a per-parameter embedding by modulating the context with the scalar parameter value.

**Ramsey coherence (normalized Hamming similarity):**

```
R_c(θᵢ, θⱼ, C) = 1 - d_H(embed(θᵢ), embed(θⱼ)) / embed_dim
```

Two parameters are considered coherent when their binary-thresholded embeddings agree on most dimensions.

**Partitioning algorithm (Union-Find):**

```
Input:  Θ ∈ ℝ^m, context C, threshold τ (default 0.25)
Output: Disjoint partition set over {0, ..., m-1}

1. Compute parameter embeddings eᵢ for all i
2. Initialize Union-Find with m singleton sets
3. For each pair (i, j) where i < j:
     if R_c(θᵢ, θⱼ, C) > τ:
       Union-Find.merge(i, j)
4. Extract partitions from Union-Find
```

---

### CFM (Correct Flow Matrix)

The CFM assigns a positive score to parameters that are contextually coherent.

```
CFM(θⱼ) = Ws(θⱼ) + Rc(θⱼ)
```

| Term | Name | Formula |
|------|------|---------|
| `Ws` | Wittgenstein Score | `cos_sim(embed(θⱼ), C) · α_m` |
| `Rc` | Ramsey Coherence | `1 - d_H(embed(θⱼ), S_prev) / dim` |

Where `C` is the context embedding, `S_prev` is the previous state embedding, and `α_m` is a per-modality weight:

| Modality | α_m |
|----------|-----|
| Text | 0.85 |
| Image | 0.80 |
| Video | 0.75 |

---

### AFM (Adversarial Flow Matrix)

The AFM penalizes parameters that contradict the context or carry low information.

```
AFM(θⱼ) = Ca(θⱼ) + Es(θⱼ)
```

| Term | Name | Formula |
|------|------|---------|
| `Ca` | Contradiction Score | `max(0, -cos_sim(embed(θⱼ), C))` |
| `Es` | Entropy Score | `-P(θⱼ) · log₂(P(θⱼ))` |

Where `P(θⱼ)` is the current probability of parameter `j` (initialized to `1/m` for uniform).

---

### ARM (Autonomous Route Matrix)

The ARM is the central evaluation pipeline that combines partitioning, CFM/AFM scoring, parameter adjustment, and sampling into a single pass.

```
Input:  Θ ∈ ℝ^m, context C ∈ ℝ^n, previous state S_prev ∈ ℝ^n, modality
Output: ARMResult {CFM[], AFM[], Θ', P[], sampled index j*, partitions}

Algorithm:
1. Partition Θ into coherent groups via Union-Find (threshold τ)
2. For each j = 0, ..., m-1:
     Compute parameter embedding: embed(θⱼ)
     CFM[j] = cos_sim(embed(θⱼ), C) · α_m + (1 - d_H(embed(θⱼ), S_prev) / dim)
     AFM[j] = max(0, -cos_sim(embed(θⱼ), C)) + H(1/m)
3. Adjust parameters:
     Θ'[j] = θⱼ · (CFM[j] - AFM[j])
4. Compute probability distribution:
     P = softmax(Θ')
5. Sample:
     j* ~ P via inverse-CDF sampling
```

---

### Evolutionary Algorithm

The EA evolves parameter vectors across generations using bio-inspired operators.

**Chromosome encoding:**

```
Genes g ∈ [0, 1]^m encode parameters directly (identity decoding).
Each chromosome carries a scalar fitness value.
```

**Fitness function:**

```
F = w_c · Coherence + w_r · Relevance + w_d · Diversity
```

Default weights: `w_c = 0.4`, `w_r = 0.4`, `w_d = 0.2`.

**Coherence** — fraction of probability mass on parameters where CFM dominates AFM:

```
Coherence = Σⱼ P[j] · 𝟙{CFM[j] > AFM[j]}
```

**Relevance** — cosine similarity between probability-weighted parameter embedding and context, normalized to `[0, 1]`:

```
V[d] = Σⱼ P[j] · θⱼ · C[d mod |C|]
Relevance = (cos_sim(V, C_slice) + 1) / 2
```

**Diversity** — normalized Shannon entropy of the probability distribution:

```
Diversity = -Σⱼ P[j] · log₂(P[j]) / log₂(m)
```

**Tournament selection:**

```
Select k random individuals from the population.
Return the one with the highest fitness.
k is clamped to population_size.
```

**Single-point crossover:**

```
Choose random crossover point c ∈ [1, dim-1].
offspring₁ = parent₁[0:c] ‖ parent₂[c:]
offspring₂ = parent₂[0:c] ‖ parent₁[c:]
```

**Random-resetting mutation:**

```
For each gene gⱼ:
  if Uniform() < μ:
    gⱼ ← Uniform(0, 1)

μ_eff = (mutation_rate > 0) ? mutation_rate : 1/param_dim
```

**Generational loop:**

```
1. Initialize population of pop_size random chromosomes
2. For generation g = 0, ..., max_generations-1:
   a. Evaluate fitness for every chromosome via ARM pipeline
   b. Sort population by fitness (descending)
   c. Copy top ⌈elitism_rate · pop_size⌉ elites into next generation
   d. Fill remaining slots:
        Select two parents via tournament selection
        Apply single-point crossover → two offspring
        Mutate each offspring
        Add to next generation
   e. Replace population
   f. Convergence check:
        If best fitness has not improved by more than 1e-6
        over the last 20 generations, stop early
3. Return best individual
```

---

### MGE (Multimedia Generation Engine)

The MGE orchestrates multi-step, multi-modal generation.

**Branch predictor (softmax classifier):**

```
logits[m] = W_m · C + b_m       for m ∈ {Text, Image, Video}
probs = softmax(logits)
predicted = argmax(probs)
confidence = probs[predicted]
```

If `confidence < 0.6`, the predictor falls back to whichever modality has gone longest without generation.

**Branch predictor online update (gradient descent):**

```
grad[m] = probs[m] - 𝟙{m == target}
b_m ← b_m - 0.01 · grad[m]
W_m ← W_m - 0.01 · grad[m] · C
```

**Cross-modal consistency check:**

```
For adjacent elements e_prev and e_curr with different modalities:
  similarity = cos_sim(e_prev.embedding, e_curr.embedding)
  threshold  = 0.3  if e_curr is Video, else 0.7
  consistent = (similarity > threshold)
```

If inconsistent, the MGE regenerates using the least-recently-used modality instead.

**Context update (exponential moving average):**

```
C_new = 0.8 · C_old + 0.2 · element_embedding
```

**Full generation pipeline:**

```
Phase 1 — Per-modality optimization:
  For each modality m ∈ {Text, Image, Video}:
    C_m     ← FDMP.encode(input, m)
    Θ_raw   ← FDMP.generate_params(C_m, m)
    Θ_best  ← EA.evolve(C_m, m)           // full evolutionary run

Phase 2 — Multi-step generation:
  For step = 0, ..., steps-1:
    (modality, confidence) ← BranchPredictor.predict(C_current)
    if confidence < 0.6:
      modality ← least recently generated modality
    Θ ← Θ_best[modality]
    element ← ARM.evaluate(Θ, C_current, S_prev, modality)
    content ← ContentGenerator.generate(element, modality)
    if not consistent(element):
      regenerate with fallback modality
    S_prev    ← C_current
    C_current ← 0.8 · C_current + 0.2 · element.embedding
    BranchPredictor.update(C_current, modality)
```

---

### Content Generator

The ContentGenerator transforms ARM-optimized parameters into concrete output.

#### Text Generation

```
Input:  ARM probabilities P, parameters Θ, context C, tokenizer, temperature T
Output: Generated text string

1. Build parameter embedding:
     param_embed[d] = Σⱼ P[j] · θⱼ · C[d mod |C|]

2. Score every vocabulary token:
     For each token t:
       sim_param = cos_sim(E[t], param_embed)
       sim_ctx   = cos_sim(E[t], C)
       score[t]  = (0.6 · sim_param + 0.4 · sim_ctx) / T

3. Convert to probabilities:
     token_probs = softmax(scores)

4. Sample tokens sequentially via inverse-CDF:
     For i = 1, ..., num_tokens:
       token_id ← sample(token_probs)
       token_probs[token_id] *= 0.3          // repetition penalty
       renormalize token_probs

5. Decode sampled token IDs back to text
```

#### Image Generation

```
Input:  Θ ∈ ℝ^m, context C ∈ ℝ^n, width, height
Output: RGB pixel grid ∈ [0, 1]^{width × height × 3}

For each pixel (x, y):
  1. Normalize coordinates:
       u = x / width,  v = y / height

  2. Spatial hash to parameter indices:
       ti = ((x·7 + y·13) ⊕ (x·y))        mod m
       tj = ((x·11 + y·3 + 1) ⊕ (x + y·5)) mod m
       tk = ((x·5 + y·17 + 2) ⊕ (x·3 + y)) mod m

  3. Cyclic context indices:
       ci = (ti·3)     mod n
       cj = (tj·3 + 1) mod n
       ck = (tk·3 + 2) mod n

  4. Spatial phase shifts (produce coherent gradients):
       phase_r = sin(2πu) · 1.5
       phase_g = cos(2πv) · 1.5
       phase_b = sin((u + v) · π) · 1.5

  5. Sigmoid color mapping:
       R = σ(θ[ti] · C[ci] · 4 + phase_r)
       G = σ(θ[tj] · C[cj] · 4 + phase_g)
       B = σ(θ[tk] · C[ck] · 4 + phase_b)

       where σ(x) = 1 / (1 + exp(-x))
```

#### Video Generation

```
Input:  Θ ∈ ℝ^m, context C, previous frame, frame index f, width, height
Output: Single frame pixel grid

1. Temporal phase:
     t = f · 0.1

2. Temporal parameter modulation:
     For each j:
       phase_j  = j · π / m
       Θ_t[j]   = θ[j] · (1 + 0.3 · sin(2π·t + phase_j))

3. Generate frame pixels using Θ_t (same spatial logic as images),
   but with time-varying phase shifts:
     phase_r = sin(2πu + t)     · 1.5
     phase_g = cos(2πv + 0.7·t) · 1.5
     phase_b = sin((u+v)·π + 1.3·t) · 1.5

4. Temporal blending for smooth motion:
     If a previous frame exists:
       pixel[i] = 0.7 · pixel[i] + 0.3 · prev_frame[i]
```

---

### Training Algorithm

Training updates the FDMP weights and token embeddings using the EA fitness signal.

**Training loop:**

```
Phase 0 — Tokenizer preparation:
  Corpus ← [system_prompt] ∪ [example.input for all examples]
  Tokenizer.build_vocab(corpus)                   // BPE merges

Phase 1 — Epoch loop:
  For epoch = 0, ..., training_epochs-1:
    For each training example (input, target_modality):
      full_input ← system_prompt + " " + input
      tokens     ← Tokenizer.encode(full_input)
      C          ← Tokenizer.embed_tokens(tokens)     // normalized mean embedding

      // Forward pass
      Θ_raw      ← FDMP.generate_params(C, target_modality)

      // Short EA run (max 20 generations, population 20)
      Θ_best     ← EA.evolve(C, target_modality)
      fitness    ← Θ_best.fitness
      loss       ← 1 - fitness

      // Update weights
      update_weights(C, target_modality, fitness, loss)
      update_embeddings(tokens, C, loss)

    Report EpochStats{epoch, avg_fitness, avg_loss, best_fitness}
```

**FDMP weight update (sign-based gradient approximation with L2 decay):**

```
For modality m with weight matrix W_m and bias b_m:
  For i = 0, ..., param_dim-1:
    s_i = sign(θ[i])                          // +1 or -1

    // Bias update
    b_m[i] ← b_m[i] - lr · (loss · s_i) - wd · b_m[i]

    // Weight update
    For j = 0, ..., embed_dim-1:
      W_m[i,j] ← W_m[i,j] - lr · (loss · s_i · C[j]) - wd · W_m[i,j]

lr = training_lr (default 0.005)
wd = weight_decay (default 1e-4)
```

**Token embedding update (regularizing gradient):**

```
scale = loss / |tokens|
lr_embed = training_lr · 0.1                    // 10x slower than FDMP

For each token t in the input:
  For d = 0, ..., embed_dim-1:
    E[t, d] ← E[t, d] - lr_embed · scale · E[t, d]
```

This acts as a loss-scaled L2 regularization that pulls embeddings toward zero-mean, with larger updates when fitness is poor.

---

### Parameter Count Derivation

Given a target parameter count, DSTT derives `embed_dim` and `param_dim` using a 2:1 ratio heuristic.

```
Total parameters = vocab_size · E + 3 · (P · E + P)

With the constraint P = E/2:
  Total ≈ vocab_size · E + 1.5 · E² + 1.5 · E

Solving the quadratic 1.5·E² + vocab_size·E - target = 0:
  E = (-vocab_size + √(vocab_size² + 6 · target)) / 3
  P = max(4, E / 2)
```

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
