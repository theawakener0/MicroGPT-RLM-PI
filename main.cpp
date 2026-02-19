#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "src/utils/logger.hpp"
#include "src/core/tensor.hpp"
#include "src/core/math_ops.hpp"
#include "src/training/tokenizer.hpp"
#include "src/training/trainer.hpp"
#include "src/training/trainable_model.hpp"

using GPT = microgpt::TrainableGPT;

using namespace microgpt;

void print_chat_banner() {

    std::cout << R"(
                                                         
                                                 
         ▀▀                                 ██   
███▄███▄ ██  ▄████ ████▄ ▄███▄ ▄████ ████▄ ▀██▀▀ 
██ ██ ██ ██  ██    ██ ▀▀ ██ ██ ██ ██ ██ ██  ██   
██ ██ ██ ██▄ ▀████ ██    ▀███▀ ▀████ ████▀  ██   
                                  ██ ██          
                                ▀▀▀  ▀▀
    )";

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║        MicroGPT-RLM-PI Chat Mode         ║\n";
    std::cout << "║          Type '/quit' to exit            ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << "\n";
}

std::string generate_response(GPT& model, Tokenizer& tokenizer, const std::string& prompt, int max_tokens = 50) {
    std::vector<int> input_ids = tokenizer.encode(prompt);
    
    std::string response;
    for (int i = 0; i < max_tokens; i++) {
        Tensor logits = model.forward(input_ids);
        
        int next_token = model.predict_next(logits);
        
        if (next_token == tokenizer.bos_id || next_token == tokenizer.eos_id) {
            break;
        }
        
        std::string token_str = tokenizer.decode({next_token});
        response += token_str;
        
        input_ids.push_back(next_token);
    }
    
    return response;
}

void run_chat_mode(GPT& model, Tokenizer& tokenizer) {
    print_chat_banner();
    
    std::string system_prompt = "You are a helpful AI assistant.";
    
    std::string conversation = system_prompt;
    
    std::string input;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, input);
        
        if (input == "/quit" || input == "/exit" || input == ":q") {
            std::cout << "Goodbye!\n";
            break;
        }
        
        if (input.empty()) {
            continue;
        }
        
        conversation += "\nUser: " + input + "\nAssistant:";
        
        std::string response = generate_response(model, tokenizer, conversation, 100);
        
        std::cout << response << "\n\n";
        
        conversation += response;
    }
}

void run_training_mode(GPT& model, Tokenizer& tokenizer, const std::string& data_path, int max_steps, const std::string& save_path = "") {
    Logger::info("Loading dataset from: " + data_path);
    
    std::vector<std::string> documents;
    std::ifstream file(data_path);
    if (!file.is_open()) {
        Logger::error("Could not open file: " + data_path);
        return;
    }
    
    std::string line;
    int count = 0;
    while (std::getline(file, line) && count < 10000) {
        if (!line.empty()) {
            documents.push_back(line);
            count++;
        }
    }
    file.close();
    
    Logger::info("Loaded " + std::to_string(documents.size()) + " documents");
    
    float total_loss = 0.0f;
    float best_loss = 1000.0f;
    
    int log_interval = 100;
    int save_interval = 1000;
    float learning_rate = 0.01f;
    
    for (int step = 0; step < max_steps; step++) {
        int doc_idx = step % documents.size();
        std::string doc = documents[doc_idx];
        
        std::vector<int> tokens = tokenizer.encode(doc);
        if (tokens.size() < 3) continue;
        
        int n = std::min((int)tokens.size() - 1, model.config.max_seq_len);
        std::vector<int> input_tokens(tokens.begin(), tokens.begin() + n);
        std::vector<int> target_tokens(tokens.begin() + 1, tokens.begin() + n + 1);
        
        Tensor logits = model.forward(input_tokens);
        
        float loss = 0.0f;
        int seq_len = logits.rows();
        int vocab = logits.cols();
        
        for (int i = 0; i < n && i < seq_len; i++) {
            int target = target_tokens[i];
            
            float max_val = logits.data[i * vocab];
            for (int j = 1; j < vocab; j++) {
                max_val = std::max(max_val, logits.data[i * vocab + j]);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < vocab; j++) {
                sum_exp += std::exp(logits.data[i * vocab + j] - max_val);
            }
            
            float prob = std::exp(logits.data[i * vocab + target] - max_val) / sum_exp;
            prob = std::max(1e-7f, std::min(1.0f - 1e-7f, prob));
            
            loss += -std::log(prob);
        }
        
        loss /= n;
        total_loss += loss;
        
        if (step % log_interval == 0) {
            float avg_loss = total_loss / (step + 1);
            float perplexity = std::exp(avg_loss);
            Logger::info("Step " + std::to_string(step) + "/" + std::to_string(max_steps) + 
                        " | Loss: " + std::to_string(avg_loss) + 
                        " | Perplexity: " + std::to_string(perplexity));
        }
        
        if (loss < best_loss) {
            best_loss = loss;
        }
    }
    
    Logger::info("Training complete!");
    Logger::info("Best loss: " + std::to_string(best_loss));
    
    if (!save_path.empty()) {
        Logger::info("Saving checkpoint to " + save_path);
        model.save(save_path);
    }
    
    Logger::info("\n=== Testing Generation ===");
    std::string test_prompt = tokenizer.decode({tokenizer.bos_id});
    
    std::vector<int> seed = {tokenizer.bos_id};
    for (int i = 0; i < 30; i++) {
        Tensor logits = model.forward(seed);
        int next_token = model.predict_next(logits);
        
        if (next_token == tokenizer.bos_id || next_token == tokenizer.eos_id) {
            break;
        }
        
        seed.push_back(next_token);
    }
    
    std::string generated = tokenizer.decode(seed);
    Logger::info("Generated: '" + generated + "'");
}

void run_demo() {
    Logger::info("=== MicroGPT-RLM Demo ===");
    
    std::vector<std::string> names = {
        "emma", "olivia", "ava", "isabella", "sophia", 
        "charlotte", "mia", "amelia", "harper", "evelyn",
        "oliver", "elijah", "liam", "noah", "james",
        "william", "benjamin", "lucas", "henry", "theodore",
        "olivia", "ava", "sophia", "charlotte", "emma",
        "isabella", "mia", "amelia", "harper", "evelyn"
    };
    
    Logger::info("Building tokenizer...");
    Tokenizer tokenizer;
    tokenizer.build(names);
    Logger::info("Vocab size: " + std::to_string(tokenizer.vocab_size));
    
    ModelConfig config;
    config.vocab_size = tokenizer.size();
    config.embed_dim = 128;
    config.num_layers = 4;
    config.num_heads = 4;
    config.max_seq_len = 32;
    config.hidden_dim = config.embed_dim * 4;
    
    Logger::info("Creating model with " + std::to_string(config.num_layers) + " layers...");
    GPT model(config);
    model.init_weights(0.02f);
    Logger::info("Model parameters: " + std::to_string(model.num_parameters()));
    
    Logger::info("\n=== Training (Quick Demo) ===");
    
    float total_loss = 0.0f;
    int num_steps = 500;
    
    for (int step = 0; step < num_steps; step++) {
        std::string doc = names[step % names.size()];
        
        std::vector<int> tokens = tokenizer.encode(doc);
        if (tokens.size() < 3) continue;
        
        int n = std::min((int)tokens.size() - 1, config.max_seq_len);
        std::vector<int> input_tokens(tokens.begin(), tokens.begin() + n);
        std::vector<int> target_tokens(tokens.begin() + 1, tokens.begin() + n + 1);
        
        Tensor logits = model.forward(input_tokens);
        
        float loss = 0.0f;
        int seq_len = logits.rows();
        int vocab = logits.cols();
        
        for (int i = 0; i < n && i < seq_len; i++) {
            int target = target_tokens[i];
            
            float max_val = logits.data[i * vocab];
            for (int j = 1; j < vocab; j++) {
                max_val = std::max(max_val, logits.data[i * vocab + j]);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < vocab; j++) {
                sum_exp += std::exp(logits.data[i * vocab + j] - max_val);
            }
            
            float prob = std::exp(logits.data[i * vocab + target] - max_val) / sum_exp;
            prob = std::max(1e-7f, std::min(1.0f - 1e-7f, prob));
            
            loss += -std::log(prob);
        }
        
        loss /= n;
        total_loss += loss;
        
        if (step % 100 == 0) {
            Logger::info("Step " + std::to_string(step) + " | Loss: " + std::to_string(total_loss / (step + 1)));
        }
    }
    
    Logger::info("\n=== Generation Test ===");
    
    std::vector<int> seed = {tokenizer.bos_id};
    for (int i = 0; i < 15; i++) {
        Tensor logits = model.forward(seed);
        int next_token = model.predict_next(logits);
        
        if (next_token == tokenizer.bos_id || next_token == tokenizer.eos_id) {
            break;
        }
        
        seed.push_back(next_token);
    }
    
    std::string generated = tokenizer.decode(seed);
    Logger::info("Generated: '" + generated + "'");
    
    Logger::info("\n=== Chat Mode (Demo) ===");
    Logger::info("(Showing chat format - model needs more training for real conversation)");
    
    std::string test_inputs[] = {
        "Hello",
        "How are you?",
        "Tell me a story"
    };
    
    for (const auto& input : test_inputs) {
        std::string prompt = tokenizer.decode({tokenizer.bos_id}) + input;
        std::string response = generate_response(model, tokenizer, prompt);
        std::cout << "User: " << input << "\n";
        std::cout << "Bot: " << response << "\n\n";
    }
}

int main(int argc, char* argv[]) {
    std::string mode = "demo";
    std::string data_path;
    int max_steps = 1000;
    std::string checkpoint_path;
    std::string save_checkpoint_path;
    std::string generate_prompt;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--chat") {
            mode = "chat";
        } else if (arg == "--train") {
            mode = "train";
        } else if (arg == "--demo") {
            mode = "demo";
        } else if (arg == "--generate") {
            mode = "generate";
        } else if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            max_steps = std::stoi(argv[++i]);
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--save_checkpoint" && i + 1 < argc) {
            save_checkpoint_path = argv[++i];
        } else if (arg == "--load_checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            generate_prompt = argv[++i];
        }
    }
    
    Logger::info("MicroGPT-RLM v1.0.0");
    Logger::info("Mode: " + mode);
    
    if (mode == "demo") {
        run_demo();
    } else if (mode == "train") {
        if (data_path.empty()) {
            Logger::error("Please specify --data <path>");
            Logger::info("Usage: ./microgpt --train --data data/names.txt --steps 10000 [--save_checkpoint model.bin]");
            return 1;
        }
        
        std::vector<std::string> sample_data = {"hello", "world", "test", "demo"};
        Tokenizer tokenizer;
        tokenizer.build(sample_data);
        
        ModelConfig config;
        config.vocab_size = tokenizer.size();
        config.embed_dim = 128;
        config.num_layers = 4;
        config.num_heads = 4;
        config.max_seq_len = 32;
        config.hidden_dim = config.embed_dim * 4;
        
        GPT model(config);
        
        // Load checkpoint if specified
        if (!checkpoint_path.empty()) {
            model.load(checkpoint_path);
        } else {
            model.init_weights(0.02f);
        }
        
        run_training_mode(model, tokenizer, data_path, max_steps, save_checkpoint_path);
        
        Logger::info("Training complete!");
        
    } else if (mode == "generate") {
        std::vector<std::string> sample_data = {"hello", "hi", "how", "are", "you"};
        Tokenizer tokenizer;
        tokenizer.build(sample_data);
        
        ModelConfig config;
        config.vocab_size = tokenizer.size();
        config.embed_dim = 128;
        config.num_layers = 4;
        config.num_heads = 4;
        config.max_seq_len = 32;
        
        GPT model(config);
        
        if (!checkpoint_path.empty()) {
            model.load(checkpoint_path);
            Logger::info("Loaded checkpoint: " + checkpoint_path);
        } else {
            Logger::warning("No checkpoint loaded, using random weights");
            model.init_weights(0.02f);
        }
        
        std::string prompt = generate_prompt.empty() ? tokenizer.decode({tokenizer.bos_id}) : generate_prompt;
        
        Logger::info("Generating text...");
        std::vector<int> seed;
        for (char c : prompt) {
            if (tokenizer.char_to_id.count(c)) {
                seed.push_back(tokenizer.char_to_id[c]);
            }
        }
        if (seed.empty()) {
            seed = {tokenizer.bos_id};
        }
        
        for (int i = 0; i < 50; i++) {
            Tensor logits = model.forward(seed);
            int next_token = model.predict_next(logits);
            
            if (next_token == tokenizer.bos_id || next_token == tokenizer.eos_id) {
                break;
            }
            
            seed.push_back(next_token);
        }
        
        std::string generated = tokenizer.decode(seed);
        Logger::info("Generated: '" + generated + "'");
        
    } else if (mode == "chat") {
        std::vector<std::string> sample_data = {"hello", "hi", "how", "are", "you"};
        Tokenizer tokenizer;
        tokenizer.build(sample_data);
        
        ModelConfig config;
        config.vocab_size = tokenizer.size();
        config.embed_dim = 128;
        config.num_layers = 4;
        config.num_heads = 4;
        config.max_seq_len = 32;
        
        GPT model(config);
        
        if (!checkpoint_path.empty()) {
            model.load(checkpoint_path);
            Logger::info("Loaded checkpoint: " + checkpoint_path);
        } else {
            Logger::warning("No checkpoint loaded - using random weights");
            model.init_weights(0.02f);
        }
        
        run_chat_mode(model, tokenizer);
    }
    
    return 0;
}
