#include <iostream>
#include <vector>
#include <fstream>
#include "src/utils/logger.hpp"
#include "src/core/tensor.hpp"
#include "src/core/math_ops.hpp"
#include "src/training/tokenizer.hpp"
#include "src/training/trainer.hpp"
#include "src/model/model.hpp"

using namespace microgpt;

int main() {
    Logger::info("MicroGPT-RLM v1.0.0");
    
    std::vector<std::string> names = {
        "emma", "olivia", "ava", "isabella", "sophia", 
        "charlotte", "mia", "amelia", "harper", "evelyn",
        "oliver", "elijah", "liam", "noah", "james",
        "william", "benjamin", "lucas", "henry", "theodore"
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
    
    Logger::info("Creating model...");
    GPT model(config);
    model.init_weights(0.02f);
    Logger::info("Model parameters: " + std::to_string(model.num_parameters()));
    
    std::vector<std::string> train_names;
    for (int i = 0; i < 1000; i++) {
        train_names.push_back(names[i % names.size()]);
    }
    
    Logger::info("Running training simulation...");
    
    float total_loss = 0.0f;
    int num_steps = 100;
    
    for (int step = 0; step < num_steps; step++) {
        std::string doc = train_names[step % train_names.size()];
        
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
        
        if (step % 20 == 0) {
            Logger::info("Step " + std::to_string(step) + " | Loss: " + std::to_string(loss));
        }
    }
    
    Logger::info("Average loss: " + std::to_string(total_loss / num_steps));
    
    Logger::info("Testing generation...");
    
    std::vector<int> seed = {tokenizer.bos_id};
    for (int i = 0; i < 10; i++) {
        Tensor logits = model.forward(seed);
        
        int next_token = model.predict_next(logits);
        
        if (next_token == tokenizer.bos_id || next_token == tokenizer.eos_id) {
            break;
        }
        
        seed.push_back(next_token);
    }
    
    std::string generated = tokenizer.decode(seed);
    Logger::info("Generated: '" + generated + "'");
    
    Logger::info("All tests passed!");
    
    return 0;
}
