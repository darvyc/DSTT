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

  Interactive Model Interface v2.0
  Generates: Text | Images | Video
)" << std::endl;
}

static void print_help() {
    std::cout << "\nCommands:\n"
              << "  /load <path.dstt>     Load a trained .dstt model file\n"
              << "  /info                 Show model metadata\n"
              << "  /steps <n>            Set generation steps (default: 12)\n"
              << "  /save_images          Toggle saving images to disk (default: off)\n"
              << "  /save_video           Toggle saving video frames to disk (default: off)\n"
              << "  /system <prompt>      Set system prompt for next training run\n"
              << "  /params <n>           Set target parameter count for next training run\n"
              << "  /train <data> <out>   Train from data file, save as .dstt\n"
              << "  /help                 Show this help\n"
              << "  /quit                 Exit\n"
              << "\n  Any other input is treated as a generation prompt.\n"
              << "  The model generates text, images, and/or video based on\n"
              << "  the prompt — the branch predictor selects the modality.\n\n";
}

static void display_result(const GenerationResult& result,
                           bool save_images, bool save_video) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  Generation Result for: \"" << result.prompt << "\"\n";
    std::cout << std::string(72, '=') << "\n";

    // ── Text Output ──
    if (result.text_steps > 0) {
        std::cout << "\n  [Text Output] (" << result.text_steps << " steps)\n";
        std::cout << std::string(72, '-') << "\n";
        std::cout << "  " << result.generated_text << "\n";
        std::cout << std::string(72, '-') << "\n";
    }

    // ── Image Output ──
    if (result.image_steps > 0) {
        std::cout << "\n  [Image Output] " << result.images.size()
                  << " image(s) generated (" << result.images[0].width
                  << "x" << result.images[0].height << " RGB)\n";

        if (save_images) {
            for (size_t i = 0; i < result.images.size(); ++i) {
                std::string path = "output_image_" + std::to_string(i) + ".ppm";
                if (result.images[i].save_ppm(path)) {
                    std::cout << "  Saved: " << path << "\n";
                } else {
                    std::cerr << "  Failed to save: " << path << "\n";
                }
            }
        } else {
            std::cout << "  (Use /save_images to write PPM files to disk)\n";
        }

        // Show pixel sample from first image
        if (!result.images.empty()) {
            const auto& img = result.images[0];
            std::cout << "  Sample pixels (top-left 4x4):\n";
            for (size_t y = 0; y < std::min(img.height, size_t(4)); ++y) {
                std::cout << "    ";
                for (size_t x = 0; x < std::min(img.width, size_t(4)); ++x) {
                    size_t off = (y * img.width + x) * 3;
                    auto r = static_cast<int>(img.pixels[off + 0] * 255);
                    auto g = static_cast<int>(img.pixels[off + 1] * 255);
                    auto b = static_cast<int>(img.pixels[off + 2] * 255);
                    std::cout << "(" << std::setw(3) << r << ","
                              << std::setw(3) << g << ","
                              << std::setw(3) << b << ") ";
                }
                std::cout << "\n";
            }
        }
    }

    // ── Video Output ──
    if (result.video_steps > 0) {
        std::cout << "\n  [Video Output] " << result.video.frames.size()
                  << " frame(s) at " << result.video.fps << " fps ("
                  << result.video.width << "x" << result.video.height << " RGB)\n";
        double duration = static_cast<double>(result.video.frames.size()) / result.video.fps;
        std::cout << "  Duration: " << std::fixed << std::setprecision(1)
                  << duration << "s\n";

        if (save_video) {
            if (result.video.save_frames("output_video")) {
                std::cout << "  Saved " << result.video.frames.size()
                          << " frames as output_video_frame_*.ppm\n";
            } else {
                std::cerr << "  Failed to save video frames\n";
            }
        } else {
            std::cout << "  (Use /save_video to write PPM frames to disk)\n";
        }
    }

    // ── Step-by-step summary ──
    std::cout << "\n" << std::string(72, '-') << "\n";
    std::cout << std::left
              << std::setw(6)  << "Step"
              << std::setw(10) << "Modality"
              << "Content Preview\n";
    std::cout << std::string(72, '-') << "\n";

    for (size_t i = 0; i < result.steps.size(); ++i) {
        const auto& s = result.steps[i];
        std::cout << std::setw(6) << i
                  << std::setw(10) << modality_name(s.modality);

        switch (s.modality) {
            case Modality::Text: {
                std::string preview = s.text;
                // Show first 50 chars
                if (preview.size() > 50)
                    preview = preview.substr(0, 50) + "...";
                // Replace newlines for display
                for (auto& c : preview) if (c == '\n') c = ' ';
                std::cout << "\"" << preview << "\"";
                break;
            }
            case Modality::Image: {
                std::cout << s.image.width << "x" << s.image.height
                          << " RGB image (" << s.image.pixels.size() << " values)";
                break;
            }
            case Modality::Video: {
                std::cout << "video frame " << i;
                break;
            }
        }
        std::cout << "\n";
    }

    std::cout << std::string(72, '-') << "\n";
    std::cout << "\n  Summary: Text=" << result.text_steps
              << "  Image=" << result.image_steps
              << "  Video=" << result.video_steps
              << "  Avg prob=" << std::fixed << std::setprecision(6)
              << result.avg_probability << "\n\n";
}

int main(int argc, char* argv[]) {
    print_banner();
    RNG::seed(42);

    DSTTModel model;
    size_t steps = 12;
    bool save_images = false;
    bool save_video = false;
    std::string train_system_prompt = "Be a helpful assistant.";
    size_t train_target_params = 0;  // 0 = use defaults

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
        } else if (line == "/save_images") {
            save_images = !save_images;
            std::cout << "Image saving: " << (save_images ? "ON" : "OFF") << "\n";
        } else if (line == "/save_video") {
            save_video = !save_video;
            std::cout << "Video saving: " << (save_video ? "ON" : "OFF") << "\n";
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
        } else if (line.substr(0, 8) == "/system ") {
            train_system_prompt = line.substr(8);
            std::cout << "System prompt set to: \"" << train_system_prompt << "\"\n";
        } else if (line.substr(0, 8) == "/params ") {
            train_target_params = std::stoul(line.substr(8));
            std::cout << "Target parameters set to: " << train_target_params << "\n";
        } else if (line.substr(0, 7) == "/train ") {
            std::string args = line.substr(7);
            size_t space = args.find(' ');
            if (space == std::string::npos) {
                std::cerr << "Usage: /train <data_file> <output.dstt>\n";
                continue;
            }
            std::string data_path = args.substr(0, space);
            std::string out_path  = args.substr(space + 1);

            try {
                std::vector<TrainingExample> examples;
                if (data_path.size() > 6 && data_path.substr(data_path.size() - 6) == ".jsonl") {
                    examples = DSTTModel::load_training_jsonl(data_path);
                } else if (data_path.size() > 4 && data_path.substr(data_path.size() - 4) == ".csv") {
                    examples = DSTTModel::load_training_csv(data_path);
                } else {
                    examples = DSTTModel::load_training_txt(data_path);
                }

                std::cout << "Loaded " << examples.size() << " training examples\n";

                Config cfg;
                cfg.system_prompt   = train_system_prompt;
                cfg.training_epochs = 3;
                cfg.training_lr     = 0.005;
                cfg.vocab_size      = 512;

                if (train_target_params > 0) {
                    cfg.set_parameter_count(train_target_params);
                    std::cout << "Target params: " << train_target_params
                              << " -> embed_dim=" << cfg.embed_dim
                              << " param_dim=" << cfg.param_dim << "\n";
                } else {
                    cfg.param_dim       = 32;
                    cfg.embed_dim       = 32;
                }

                std::cout << "System prompt: \"" << cfg.system_prompt << "\"\n";
                std::cout << "Parameters:    " << cfg.total_parameters() << "\n";
                std::cout << "Training...\n";
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

                model.load(out_path);
                std::cout << "Model saved to " << out_path << " and loaded.\n";
                std::cout << model.info() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        } else if (line[0] == '/') {
            std::cerr << "Unknown command. Type /help for available commands.\n";
        } else {
            // ── Generation prompt ──
            if (!model.is_loaded()) {
                std::cerr << "No model loaded. Use /load <path.dstt> or /train first.\n";
                continue;
            }

            try {
                std::cout << "Generating...\n";
                GenerationResult result;
                {
                    ScopedTimer timer("Generation");
                    result = model.run(line, steps,
                        [](size_t step, const GeneratedContent& c) {
                            std::cout << "  [" << step << "] "
                                      << modality_name(c.modality);
                            if (c.modality == Modality::Text && !c.text.empty()) {
                                std::string preview = c.text;
                                if (preview.size() > 30)
                                    preview = preview.substr(0, 30) + "...";
                                for (auto& ch : preview) if (ch == '\n') ch = ' ';
                                std::cout << " -> \"" << preview << "\"";
                            } else if (c.modality == Modality::Image) {
                                std::cout << " -> " << c.image.width << "x"
                                          << c.image.height << " image";
                            } else if (c.modality == Modality::Video) {
                                std::cout << " -> video frame";
                            }
                            std::cout << "\n";
                        });
                }
                display_result(result, save_images, save_video);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }

    std::cout << "Goodbye.\n";
    return 0;
}
