#pragma once
#include "../model/model.hpp"
#include "../training/tokenizer.hpp"
#include "sampler.hpp"
#include <string>
#include <vector>
#include <functional>

namespace microgpt {

class Generator {
public:
    Sampler sampler;
    int max_length;
    int eos_token_id;
    bool echo;
    float repetition_penalty;
    
    Generator() : max_length(100), eos_token_id(-1), echo(false), repetition_penalty(1.0f) {}
    
    Generator(int max_length, int eos_token_id = -1) 
        : max_length(max_length), eos_token_id(eos_token_id), echo(false), repetition_penalty(1.0f) {}
    
    std::string generate(GPT& model, const std::vector<int>& input_ids) {
        std::vector<int> tokens = input_ids;
        std::vector<int> generated;
        
        for (int i = 0; i < max_length; i++) {
            if ((int)tokens.size() > model.config.max_seq_len) {
                break;
            }
            
            Tensor logits = model.forward(tokens);
            
            int next_token;
            if (repetition_penalty != 1.0f && !generated.empty()) {
                next_token = sampler.sample_with_penalty(logits, generated, repetition_penalty);
            } else {
                next_token = sampler.sample(logits);
            }
            
            if (next_token == eos_token_id) {
                break;
            }
            
            generated.push_back(next_token);
            tokens.push_back(next_token);
        }
        
        std::string result;
        return result;
    }
    
    std::string generate(GPT& model, Tokenizer& tokenizer, const std::string& prompt) {
        std::vector<int> input_ids = tokenizer.encode(prompt);
        
        if (echo) {
            std::string result = prompt;
            return result + generate_from_tokens(model, input_ids);
        }
        
        return generate_from_tokens(model, input_ids);
    }
    
    std::string generate_from_tokens(GPT& model, const std::vector<int>& input_ids) {
        std::vector<int> tokens = input_ids;
        std::vector<int> generated;
        
        for (int i = 0; i < max_length; i++) {
            if ((int)tokens.size() > model.config.max_seq_len) {
                break;
            }
            
            Tensor logits = model.forward(tokens);
            
            int next_token;
            if (repetition_penalty != 1.0f && !generated.empty()) {
                next_token = sampler.sample_with_penalty(logits, generated, repetition_penalty);
            } else {
                next_token = sampler.sample(logits);
            }
            
            if (next_token == eos_token_id) {
                break;
            }
            
            generated.push_back(next_token);
            tokens.push_back(next_token);
        }
        
        std::string result;
        return result;
    }
    
    using StreamCallback = std::function<void(const std::string&)>;
    
    void generate_stream(GPT& model, Tokenizer& tokenizer, const std::string& prompt, 
                       StreamCallback callback) {
        std::vector<int> input_ids = tokenizer.encode(prompt);
        std::vector<int> tokens = input_ids;
        
        for (int i = 0; i < max_length; i++) {
            if ((int)tokens.size() > model.config.max_seq_len) {
                break;
            }
            
            Tensor logits = model.forward(tokens);
            int next_token = sampler.sample(logits);
            
            if (next_token == eos_token_id) {
                break;
            }
            
            std::string token_str = tokenizer.decode({next_token});
            callback(token_str);
            
            tokens.push_back(next_token);
        }
    }
};

}
