#include <dstt/training/trainer.hpp>
#include <dstt/utils/math.hpp>
#include <dstt/utils/random.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>

namespace dstt {

Trainer::Trainer(const Config& cfg)
    : cfg_(cfg), fdmp_(cfg), tokenizer_(cfg) {}

void Trainer::build_tokenizer(const std::vector<std::string>& corpus) {
    tokenizer_.build_vocab(corpus);
}

std::pair<double, double> Trainer::train_step(const TrainingExample& ex) {
    // 1. Tokenize and embed
    std::vector<uint32_t> tokens = tokenizer_.encode(ex.input);
    Vec context = tokenizer_.embed_tokens(tokens);

    // 2. Generate raw parameters from FDMP
    Vec raw_theta = fdmp_.generate_params(context, ex.modality);

    // 3. Evolve parameters via EA (short run for training)
    Config ea_cfg = cfg_;
    ea_cfg.max_generations = std::min(cfg_.max_generations, size_t(20));
    ea_cfg.population_size = std::min(cfg_.population_size, size_t(20));

    Population pop(ea_cfg);
    Vec init_prev(cfg_.embed_dim, 0.0);
    Chromosome best = pop.evolve(context, init_prev, ex.modality);

    double fitness = best.fitness;
    double loss = 1.0 - fitness;  // fitness gap

    // 4. Update FDMP weights
    update_weights(context, ex.modality, fitness, loss);

    // 5. Update token embeddings
    update_embeddings(tokens, context, loss);

    return {fitness, loss};
}

void Trainer::update_weights(const Vec& context, Modality m,
                             double /*fitness*/, double loss) {
    double lr = cfg_.training_lr;
    double wd = cfg_.weight_decay;

    Vec& W = fdmp_.weight_matrix_mut(m);
    Vec& b = fdmp_.bias_vector_mut(m);

    size_t pd = cfg_.param_dim;
    size_t ed = cfg_.embed_dim;

    // Gradient approximation: the loss gradient w.r.t. W is approximated
    // as loss * outer(theta_direction, context).  This pushes W towards
    // parameter configurations that yield higher fitness.
    //
    // Since Θ_r = W · C + b, a better approximation of ∂L/∂W uses:
    //   ∂L/∂W_ij ≈ loss · sign(Θ_r_i) · C_j
    // This drives parameters towards values that reduce the fitness gap.

    Vec theta = math::affine(W, pd, ed, context, b);

    for (size_t i = 0; i < pd; ++i) {
        double sign_theta = (theta[i] >= 0.0) ? 1.0 : -1.0;
        double grad_b = loss * sign_theta;
        b[i] -= lr * grad_b + wd * b[i];

        for (size_t j = 0; j < ed; ++j) {
            size_t idx = i * ed + j;
            double grad_w = loss * sign_theta * context[j];
            W[idx] -= lr * grad_w + wd * W[idx];
        }
    }
}

void Trainer::update_embeddings(const std::vector<uint32_t>& tokens,
                                const Vec& /*context*/, double loss) {
    if (tokens.empty()) return;

    double lr = cfg_.training_lr * 0.1;  // slower LR for embeddings
    size_t dim = cfg_.embed_dim;
    Vec& E = tokenizer_.embedding_matrix();
    size_t vocab = tokenizer_.actual_vocab_size();

    // Push token embeddings towards reducing loss.
    // Gradient: ∂L/∂E[t] ≈ loss / |tokens| for each token t
    // (simplified: move embeddings to reduce variance)
    double scale = loss / static_cast<double>(tokens.size());

    for (uint32_t id : tokens) {
        if (id >= vocab) continue;
        size_t offset = static_cast<size_t>(id) * dim;
        for (size_t d = 0; d < dim; ++d) {
            // Nudge towards zero-mean (regularisation) scaled by loss
            E[offset + d] -= lr * scale * E[offset + d];
        }
    }
}

void Trainer::train(const std::vector<TrainingExample>& examples,
                    EpochCallback on_epoch) {
    if (!tokenizer_.is_trained()) {
        // Auto-build tokenizer from training inputs
        std::vector<std::string> corpus;
        corpus.reserve(examples.size());
        for (const auto& ex : examples) {
            corpus.push_back(ex.input);
        }
        build_tokenizer(corpus);
    }

    for (size_t epoch = 0; epoch < cfg_.training_epochs; ++epoch) {
        double total_fitness = 0.0;
        double total_loss = 0.0;
        double best_fitness = 0.0;

        for (size_t i = 0; i < examples.size(); ++i) {
            auto [fitness, loss] = train_step(examples[i]);
            total_fitness += fitness;
            total_loss += loss;
            best_fitness = std::max(best_fitness, fitness);
        }

        double n = static_cast<double>(examples.size());

        if (on_epoch) {
            on_epoch(EpochStats{
                epoch,
                total_fitness / n,
                total_loss / n,
                best_fitness
            });
        }
    }

    trained_ = true;
}

void Trainer::save(const std::string& path) const {
    // Save FDMP weights
    auto save_vec = [](const std::string& filepath, const Vec& v) {
        std::ofstream f(filepath, std::ios::binary);
        size_t n = v.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(double)));
    };

    for (size_t mi = 0; mi < MODALITY_COUNT; ++mi) {
        Modality m = static_cast<Modality>(mi);
        std::string prefix = path + "/" + modality_name(m);
        save_vec(prefix + "_W.bin", fdmp_.weight_matrix(m));
        save_vec(prefix + "_b.bin", fdmp_.bias_vector(m));
    }

    // Save token embeddings
    save_vec(path + "/token_embeddings.bin", tokenizer_.embedding_matrix());

    // Save vocabulary
    std::ofstream vocab_f(path + "/vocab.txt");
    for (const auto& [token, id] : tokenizer_.token_to_id()) {
        // Escape special chars for safety
        vocab_f << id << "\t";
        for (char c : token) {
            if (c == '\n') vocab_f << "\\n";
            else if (c == '\t') vocab_f << "\\t";
            else if (c == '\\') vocab_f << "\\\\";
            else vocab_f << c;
        }
        vocab_f << "\n";
    }

    // Save config summary
    std::ofstream cfg_f(path + "/config.txt");
    cfg_f << "embed_dim=" << cfg_.embed_dim << "\n";
    cfg_f << "param_dim=" << cfg_.param_dim << "\n";
    cfg_f << "vocab_size=" << cfg_.vocab_size << "\n";
    cfg_f << "training_epochs=" << cfg_.training_epochs << "\n";
    cfg_f << "training_lr=" << cfg_.training_lr << "\n";
}

void Trainer::load(const std::string& path) {
    auto load_vec = [](const std::string& filepath) -> Vec {
        std::ifstream f(filepath, std::ios::binary);
        size_t n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        Vec v(n);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(double)));
        return v;
    };

    for (size_t mi = 0; mi < MODALITY_COUNT; ++mi) {
        Modality m = static_cast<Modality>(mi);
        std::string prefix = path + "/" + modality_name(m);
        Vec W = load_vec(prefix + "_W.bin");
        Vec b = load_vec(prefix + "_b.bin");

        Vec& W_ref = fdmp_.weight_matrix_mut(m);
        Vec& b_ref = fdmp_.bias_vector_mut(m);
        W_ref = std::move(W);
        b_ref = std::move(b);
    }

    // Load token embeddings
    Vec emb = load_vec(path + "/token_embeddings.bin");
    tokenizer_.embedding_matrix() = std::move(emb);

    // Load vocabulary
    std::unordered_map<std::string, uint32_t> t2i;
    std::unordered_map<uint32_t, std::string> i2t;
    std::ifstream vocab_f(path + "/vocab.txt");
    std::string line;
    while (std::getline(vocab_f, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        uint32_t id = static_cast<uint32_t>(std::stoul(line.substr(0, tab)));
        std::string token_raw = line.substr(tab + 1);
        // Unescape
        std::string token;
        for (size_t i = 0; i < token_raw.size(); ++i) {
            if (token_raw[i] == '\\' && i + 1 < token_raw.size()) {
                char next = token_raw[i + 1];
                if (next == 'n') { token += '\n'; i++; }
                else if (next == 't') { token += '\t'; i++; }
                else if (next == '\\') { token += '\\'; i++; }
                else token += token_raw[i];
            } else {
                token += token_raw[i];
            }
        }
        t2i[token] = id;
        i2t[id] = token;
    }
    tokenizer_.set_vocab(std::move(t2i), std::move(i2t));

    trained_ = true;
}

} // namespace dstt
