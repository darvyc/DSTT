#include <dstt/model/dstt_model.hpp>
#include <dstt/utils/random.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace dstt {

DSTTModel::DSTTModel() : cfg_(), fdmp_(cfg_), tokenizer_(cfg_) {}

DSTTModel::DSTTModel(const Config& cfg)
    : cfg_(cfg), fdmp_(cfg), tokenizer_(cfg) {}

// ── .dstt file I/O ──────────────────────────────────────────────────

void DSTTModel::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + path);
    write_to_stream(out);
}

void DSTTModel::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open model file: " + path);
    read_from_stream(in);
    loaded_ = true;
}

void DSTTModel::write_to_stream(std::ostream& out) const {
    // ── Header ──
    DSTTHeader hdr;
    hdr.embed_dim  = static_cast<uint32_t>(cfg_.embed_dim);
    hdr.param_dim  = static_cast<uint32_t>(cfg_.param_dim);
    hdr.vocab_size = static_cast<uint32_t>(tokenizer_.actual_vocab_size());
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // ── Config block ──
    DSTTConfigBlock cb;
    cb.training_lr     = cfg_.training_lr;
    cb.weight_decay    = cfg_.weight_decay;
    cb.training_epochs = static_cast<uint32_t>(cfg_.training_epochs);
    cb.max_vocab_size  = static_cast<uint32_t>(cfg_.vocab_size);
    cb.w_coherence     = cfg_.w_coherence;
    cb.w_relevance     = cfg_.w_relevance;
    cb.w_diversity     = cfg_.w_diversity;
    cb.coherence_threshold     = cfg_.coherence_threshold;
    cb.bp_confidence_threshold = cfg_.bp_confidence_threshold;
    cb.population_size   = static_cast<uint32_t>(cfg_.population_size);
    cb.max_generations   = static_cast<uint32_t>(cfg_.max_generations);
    cb.tournament_k      = static_cast<uint32_t>(cfg_.tournament_k);
    cb.elitism_rate      = cfg_.elitism_rate;
    cb.mutation_rate     = cfg_.mutation_rate;
    out.write(reinterpret_cast<const char*>(&cb), sizeof(cb));

    // ── Vocabulary ──
    const auto& t2i = tokenizer_.token_to_id();
    uint32_t vocab_count = static_cast<uint32_t>(t2i.size());
    out.write(reinterpret_cast<const char*>(&vocab_count), sizeof(vocab_count));

    for (const auto& [token, id] : t2i) {
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        uint32_t len = static_cast<uint32_t>(token.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(token.data(), static_cast<std::streamsize>(len));
    }

    // ── Token embeddings ──
    auto write_vec = [&](const Vec& v) {
        uint64_t n = v.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(n * sizeof(double)));
    };

    write_vec(tokenizer_.embedding_matrix());

    // ── FDMP weights (per modality) ──
    for (size_t mi = 0; mi < MODALITY_COUNT; ++mi) {
        Modality m = static_cast<Modality>(mi);
        write_vec(fdmp_.weight_matrix(m));
        write_vec(fdmp_.bias_vector(m));
    }
}

void DSTTModel::read_from_stream(std::istream& in) {
    // ── Header ──
    DSTTHeader hdr;
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != DSTT_MAGIC) {
        throw std::runtime_error("Invalid .dstt file: bad magic number");
    }
    if (hdr.version != DSTT_VERSION) {
        throw std::runtime_error("Unsupported .dstt version: " +
                                 std::to_string(hdr.version));
    }

    cfg_.embed_dim = hdr.embed_dim;
    cfg_.param_dim = hdr.param_dim;

    // ── Config block ──
    DSTTConfigBlock cb;
    in.read(reinterpret_cast<char*>(&cb), sizeof(cb));
    cfg_.training_lr     = cb.training_lr;
    cfg_.weight_decay    = cb.weight_decay;
    cfg_.training_epochs = cb.training_epochs;
    cfg_.vocab_size      = cb.max_vocab_size;
    cfg_.w_coherence     = cb.w_coherence;
    cfg_.w_relevance     = cb.w_relevance;
    cfg_.w_diversity     = cb.w_diversity;
    cfg_.coherence_threshold     = cb.coherence_threshold;
    cfg_.bp_confidence_threshold = cb.bp_confidence_threshold;
    cfg_.population_size   = cb.population_size;
    cfg_.max_generations   = cb.max_generations;
    cfg_.tournament_k      = cb.tournament_k;
    cfg_.elitism_rate      = cb.elitism_rate;
    cfg_.mutation_rate     = cb.mutation_rate;

    // Reinitialize components with loaded config
    fdmp_ = FDMP(cfg_);
    tokenizer_ = Tokenizer(cfg_);

    // ── Vocabulary ──
    uint32_t vocab_count = 0;
    in.read(reinterpret_cast<char*>(&vocab_count), sizeof(vocab_count));

    std::unordered_map<std::string, uint32_t> t2i;
    std::unordered_map<uint32_t, std::string> i2t;
    for (uint32_t i = 0; i < vocab_count; ++i) {
        uint32_t id = 0;
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string token(len, '\0');
        in.read(&token[0], static_cast<std::streamsize>(len));
        t2i[token] = id;
        i2t[id] = token;
    }
    tokenizer_.set_vocab(std::move(t2i), std::move(i2t));

    // ── Token embeddings ──
    auto read_vec = [&]() -> Vec {
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        Vec v(n);
        in.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(double)));
        return v;
    };

    tokenizer_.embedding_matrix() = read_vec();

    // ── FDMP weights ──
    for (size_t mi = 0; mi < MODALITY_COUNT; ++mi) {
        Modality m = static_cast<Modality>(mi);
        fdmp_.weight_matrix_mut(m) = read_vec();
        fdmp_.bias_vector_mut(m) = read_vec();
    }
}

// ── Training ────────────────────────────────────────────────────────

void DSTTModel::train_and_save(const std::vector<TrainingExample>& examples,
                               const std::string& output_path,
                               EpochCallback on_epoch) {
    Trainer trainer(cfg_);
    trainer.train(examples, on_epoch);

    // Transfer trained state to this model
    fdmp_ = trainer.fdmp();
    tokenizer_ = trainer.tokenizer();
    loaded_ = true;

    save(output_path);
}

// ── Generation ──────────────────────────────────────────────────────

SynthesisedOutput DSTTModel::generate(const std::string& prompt,
                                       size_t steps,
                                       StepCallback on_step) {
    if (!loaded_) {
        throw std::runtime_error("No model loaded. Call load() or train_and_save() first.");
    }

    ARM arm(cfg_);
    BranchPredictor bp(cfg_);
    Synthesiser synth(cfg_);

    // Phase 1: Encode and evolve parameters per modality
    struct ModalityState {
        Vec context;
        Vec theta_optimised;
    };

    std::array<ModalityState, MODALITY_COUNT> states;
    Vec init_prev(cfg_.embed_dim, 0.0);

    for (size_t mi = 0; mi < MODALITY_COUNT; ++mi) {
        Modality m = static_cast<Modality>(mi);

        // Tokenize and embed using trained tokenizer
        std::vector<uint32_t> tokens = tokenizer_.encode(prompt);
        Vec context = tokenizer_.embed_tokens(tokens);
        states[mi].context = context;

        // Generate raw params from trained FDMP
        Vec raw_theta = fdmp_.generate_params(context, m);

        // Evolve
        Config ea_cfg = cfg_;
        ea_cfg.max_generations = std::min(cfg_.max_generations, size_t(50));
        ea_cfg.population_size = std::min(cfg_.population_size, size_t(30));

        Population pop(ea_cfg);
        Chromosome best = pop.evolve(context, init_prev, m);
        states[mi].theta_optimised = best.decode();
    }

    // Phase 2: Generate output with branch prediction
    Vec current_context = states[0].context;
    Vec prev_state = init_prev;

    for (size_t step = 0; step < steps; ++step) {
        auto [predicted_m, confidence] = bp.predict(current_context);
        if (confidence < cfg_.bp_confidence_threshold) {
            predicted_m = bp.least_recent();
        }

        size_t mi = static_cast<size_t>(predicted_m);
        const Vec& theta = states[mi].theta_optimised;

        ARMResult r = arm.evaluate(theta, current_context, prev_state, predicted_m);

        size_t edim = std::min(cfg_.embed_dim, current_context.size());
        Vec param_embed(edim);
        double theta_val = theta[r.sampled_idx];
        for (size_t d = 0; d < edim; ++d) {
            param_embed[d] = theta_val * current_context[d % current_context.size()];
        }

        OutputElement elem{predicted_m, std::move(param_embed),
                          r.sampled_idx, r.probabilities[r.sampled_idx]};

        synth.append(elem);

        if (!synth.check_consistency()) {
            Modality fallback = bp.least_recent();
            size_t fi = static_cast<size_t>(fallback);
            ARMResult fr = arm.evaluate(states[fi].theta_optimised,
                                        current_context, prev_state, fallback);
            Vec fb_embed(edim);
            double fb_val = states[fi].theta_optimised[fr.sampled_idx];
            for (size_t d = 0; d < edim; ++d) {
                fb_embed[d] = fb_val * current_context[d % current_context.size()];
            }
            synth.output().elements.back() = OutputElement{
                fallback, std::move(fb_embed),
                fr.sampled_idx, fr.probabilities[fr.sampled_idx]};
        }

        const auto& last = synth.output().elements.back();

        if (on_step) {
            on_step(step, last);
        }

        prev_state = current_context;
        if (last.embedding.size() == current_context.size()) {
            for (size_t d = 0; d < current_context.size(); ++d) {
                current_context[d] = 0.8 * current_context[d]
                                   + 0.2 * last.embedding[d];
            }
        }

        bp.update(current_context, last.modality);
        bp.record_generation(last.modality);
    }

    return synth.output();
}

// ── Model info ──────────────────────────────────────────────────────

std::string DSTTModel::info() const {
    std::ostringstream ss;
    ss << "DSTT Model";
    if (!loaded_) {
        ss << " (not loaded)";
        return ss.str();
    }
    ss << "\n  Format:          .dstt v" << DSTT_VERSION;
    ss << "\n  Embed dim:       " << cfg_.embed_dim;
    ss << "\n  Param dim:       " << cfg_.param_dim;
    ss << "\n  Vocabulary:      " << tokenizer_.actual_vocab_size() << " tokens";
    ss << "\n  Modalities:      " << MODALITY_COUNT << " (Text, Image, Video)";
    ss << "\n  Population:      " << cfg_.population_size;
    ss << "\n  Max generations: " << cfg_.max_generations;
    ss << "\n  Training LR:     " << cfg_.training_lr;
    ss << "\n  Training epochs: " << cfg_.training_epochs;
    return ss.str();
}

// ── Training data loaders ───────────────────────────────────────────

static Modality parse_modality(const std::string& s) {
    if (s == "Image" || s == "image") return Modality::Image;
    if (s == "Video" || s == "video") return Modality::Video;
    return Modality::Text;
}

std::vector<TrainingExample> DSTTModel::load_training_jsonl(const std::string& path) {
    std::vector<TrainingExample> examples;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open training file: " + path);

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Minimal JSON parsing for {"input": "...", "modality": "..."}
        auto extract = [&](const std::string& key) -> std::string {
            std::string needle = "\"" + key + "\"";
            size_t pos = line.find(needle);
            if (pos == std::string::npos) return "";
            pos = line.find(':', pos + needle.size());
            if (pos == std::string::npos) return "";
            pos = line.find('"', pos + 1);
            if (pos == std::string::npos) return "";
            size_t end = line.find('"', pos + 1);
            if (end == std::string::npos) return "";
            return line.substr(pos + 1, end - pos - 1);
        };

        std::string input = extract("input");
        std::string mod   = extract("modality");
        if (!input.empty()) {
            examples.push_back({input, parse_modality(mod)});
        }
    }
    return examples;
}

std::vector<TrainingExample> DSTTModel::load_training_txt(const std::string& path) {
    std::vector<TrainingExample> examples;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open training file: " + path);

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        examples.push_back({line, Modality::Text});
    }
    return examples;
}

std::vector<TrainingExample> DSTTModel::load_training_csv(const std::string& path) {
    std::vector<TrainingExample> examples;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open training file: " + path);

    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (header) { header = false; continue; }  // Skip CSV header row

        size_t comma = line.rfind(',');
        if (comma == std::string::npos) {
            examples.push_back({line, Modality::Text});
        } else {
            std::string input = line.substr(0, comma);
            std::string mod   = line.substr(comma + 1);
            // Trim whitespace
            while (!mod.empty() && mod.front() == ' ') mod.erase(mod.begin());
            while (!mod.empty() && mod.back() == ' ') mod.pop_back();
            examples.push_back({input, parse_modality(mod)});
        }
    }
    return examples;
}

} // namespace dstt
