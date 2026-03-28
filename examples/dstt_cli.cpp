#include <dstt/model/dstt_model.hpp>
#include <dstt/utils/random.hpp>
#include <dstt/utils/timer.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace dstt;

static void print_banner() {
    std::cout << R"(
    ____  ______________
   / __ \/ ___/_  __/_  __/
  / / / /\__ \ / /   / /
 / /_/ /___/ // /   / /
/_____//____//_/   /_/

  Interactive Model Interface v1.0
)" << std::endl;
}

static void print_help() {
    std::cout << "\nCommands:\n"
              << "  /load <path.dstt>   Load a trained .dstt model file\n"
              << "  /info               Show model metadata\n"
              << "  /steps <n>          Set generation steps (default: 12)\n"
              << "  /train <data> <out> Train from data file, save as .dstt\n"
              << "  /help               Show this help\n"
              << "  /quit               Exit\n"
              << "\n  Any other input is treated as a generation prompt.\n\n";
}

static void print_output(const SynthesisedOutput& output) {
    std::cout << "\n" << std::string(68, '-') << "\n";
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(10) << "Modality"
              << std::setw(10) << "Param#"
              << std::setw(14) << "Probability"
              << "Embedding[0..3]\n";
    std::cout << std::string(68, '-') << "\n";

    for (size_t i = 0; i < output.elements.size(); ++i) {
        const auto& e = output.elements[i];
        std::cout << std::setw(6) << i
                  << std::setw(10) << modality_name(e.modality)
                  << std::setw(10) << e.param_index
                  << std::fixed << std::setprecision(6)
                  << std::setw(14) << e.probability << "  [";
        size_t show = std::min(e.embedding.size(), size_t(4));
        for (size_t d = 0; d < show; ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << std::setprecision(4) << e.embedding[d];
        }
        if (e.embedding.size() > 4) std::cout << ", ...";
        std::cout << "]\n";
    }
    std::cout << std::string(68, '-') << "\n";

    std::cout << "\n  Text: " << output.count(Modality::Text)
              << "  Image: " << output.count(Modality::Image)
              << "  Video: " << output.count(Modality::Video)
              << "  Avg prob: " << std::fixed << std::setprecision(6)
              << output.avg_probability() << "\n\n";
}

int main(int argc, char* argv[]) {
    print_banner();
    RNG::seed(42);

    DSTTModel model;
    size_t steps = 12;

    // If a model path was provided as argument, load it
    if (argc >= 2) {
        std::string path = argv[1];
        std::cout << "Loading model: " << path << "\n";
        try {
            ScopedTimer timer("Model load");
            model.load(path);
            std::cout << model.info() << "\n\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    print_help();

    std::string line;
    while (true) {
        std::cout << "dstt> ";
        std::cout.flush();
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        if (line == "/quit" || line == "/exit") {
            break;
        } else if (line == "/help") {
            print_help();
        } else if (line == "/info") {
            std::cout << model.info() << "\n";
        } else if (line.substr(0, 6) == "/load ") {
            std::string path = line.substr(6);
            try {
                ScopedTimer timer("Model load");
                model.load(path);
                std::cout << model.info() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        } else if (line.substr(0, 7) == "/steps ") {
            steps = std::stoul(line.substr(7));
            std::cout << "Generation steps set to " << steps << "\n";
        } else if (line.substr(0, 7) == "/train ") {
            // Parse: /train <data_file> <output.dstt>
            std::string args = line.substr(7);
            size_t space = args.find(' ');
            if (space == std::string::npos) {
                std::cerr << "Usage: /train <data_file> <output.dstt>\n";
                continue;
            }
            std::string data_path = args.substr(0, space);
            std::string out_path  = args.substr(space + 1);

            try {
                // Detect format by extension
                std::vector<TrainingExample> examples;
                if (data_path.size() > 6 && data_path.substr(data_path.size() - 6) == ".jsonl") {
                    examples = DSTTModel::load_training_jsonl(data_path);
                } else if (data_path.size() > 4 && data_path.substr(data_path.size() - 4) == ".csv") {
                    examples = DSTTModel::load_training_csv(data_path);
                } else {
                    examples = DSTTModel::load_training_txt(data_path);
                }

                std::cout << "Loaded " << examples.size() << " training examples\n";
                std::cout << "Training...\n";

                Config cfg;
                cfg.training_epochs = 3;
                cfg.training_lr     = 0.005;
                cfg.vocab_size      = 512;
                cfg.param_dim       = 32;
                cfg.embed_dim       = 32;
                cfg.population_size = 30;
                cfg.max_generations = 40;

                DSTTModel train_model(cfg);
                train_model.train_and_save(examples, out_path,
                    [](const EpochStats& stats) {
                        std::cout << "  Epoch " << stats.epoch
                                  << "  fitness=" << std::fixed << std::setprecision(4)
                                  << stats.avg_fitness
                                  << "  loss=" << stats.avg_loss << "\n";
                    });

                // Load the newly trained model for interactive use
                model.load(out_path);
                std::cout << "Model saved to " << out_path << " and loaded.\n";
                std::cout << model.info() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        } else if (line[0] == '/') {
            std::cerr << "Unknown command. Type /help for available commands.\n";
        } else {
            // Treat as generation prompt
            if (!model.is_loaded()) {
                std::cerr << "No model loaded. Use /load <path.dstt> or /train first.\n";
                continue;
            }

            try {
                ScopedTimer timer("Generation");
                SynthesisedOutput output = model.generate(line, steps,
                    [](size_t step, const OutputElement& elem) {
                        std::cout << "  [" << step << "] "
                                  << modality_name(elem.modality)
                                  << "  prob=" << std::fixed << std::setprecision(4)
                                  << elem.probability << "\n";
                    });
                print_output(output);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }

    std::cout << "Goodbye.\n";
    return 0;
}
