#pragma once
#include "../training/trainable_model.hpp"
#include "../training/tokenizer.hpp"
#include "sampler.hpp"
#include <string>
#include <vector>
#include <functional>
#include <optional>

namespace microgpt {

using GPT = TrainableGPT;

class Generator {
public:
    Sampler sampler;
    int max_length;
    int eos_token_id;
    bool echo;
    float repetition_penalty;
    std::vector<std::vector<int>> stop_sequences;
    
    Generator() 
        : max_length(100), eos_token_id(-1), echo(false), repetition_penalty(1.0f) {}
    
    Generator(int max_length, int eos_token_id = -1) 
        : max_length(max_length), eos_token_id(eos_token_id), echo(false), 
          repetition_penalty(1.0f) {}
    
    std::string generate(GPT& model, Tokenizer& tokenizer, const std::string& prompt) {
        std::vector<int> input_ids = tokenizer.encode(prompt);
        
        std::string result;
        if (echo) {
            result = prompt;
        }
        
        result += generate_from_tokens(model, tokenizer, input_ids);
        return result;
    }
    
    std::string generate_from_tokens(GPT& model, Tokenizer& tokenizer, 
                                   const std::vector<int>& input_ids) {
        std::vector<int> tokens = input_ids;
        std::string result;
        
        for (int i = 0; i < max_length; i++) {
            if ((int)tokens.size() > model.config.max_seq_len) {
                break;
            }
            
            Tensor logits = model.forward(tokens);
            
            int next_token;
            if (repetition_penalty != 1.0f && !tokens.empty()) {
                std::vector<int> recent_tokens;
                int start = std::max(0, (int)tokens.size() - 20);
                for (size_t j = start; j < tokens.size(); j++) {
                    recent_tokens.push_back(tokens[j]);
                }
                next_token = sampler.sample_with_penalty(logits, recent_tokens, repetition_penalty);
            } else {
                next_token = sampler.sample(logits);
            }
            
            if (next_token == eos_token_id) {
                break;
            }
            
            if (detect_stop(tokens)) {
                break;
            }
            
            result += tokenizer.decode({next_token});
            tokens.push_back(next_token);
        }
        
        return result;
    }
    
    int generate_token(GPT& model, const std::vector<int>& tokens) {
        if ((int)tokens.size() > model.config.max_seq_len) {
            return -1;
        }
        
        Tensor logits = model.forward(tokens);
        return sampler.sample(logits);
    }
    
    using StreamCallback = std::function<void(const std::string&)>;
    
    void generate_stream(GPT& model, Tokenizer& tokenizer,
                       const std::string& prompt,
                       StreamCallback callback) {
        std::vector<int> input_ids = tokenizer.encode(prompt);
        std::vector<int> tokens = input_ids;
        
        if (echo) {
            callback(prompt);
        }
        
        for (int i = 0; i < max_length; i++) {
            if ((int)tokens.size() > model.config.max_seq_len) {
                break;
            }
            
            Tensor logits = model.forward(tokens);
            int next_token = sampler.sample(logits);
            
            if (next_token == eos_token_id) {
                break;
            }
            
            if (detect_stop(tokens)) {
                break;
            }
            
            std::string token_str = tokenizer.decode({next_token});
            callback(token_str);
            
            tokens.push_back(next_token);
        }
    }
    
    bool detect_stop(const std::vector<int>& tokens) {
        if (stop_sequences.empty()) {
            return false;
        }
        
        for (size_t t = tokens.size(); t > 0; t--) {
            for (const auto& seq : stop_sequences) {
                if (!seq.empty() && t >= seq.size()) {
                    bool match = true;
                    for (size_t i = 0; i < seq.size(); i++) {
                        if (tokens[t - seq.size() + i] != (int)seq[i]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    void add_stop_sequence(const std::vector<int>& seq) {
        stop_sequences.push_back(seq);
    }
    
    void clear_stop_sequences() {
        stop_sequences.clear();
    }
};

class GeneratorWithKVCache {
public:
    GPT* model;
    Sampler sampler;
    int max_length;
    int eos_token_id;
    bool echo;
    float repetition_penalty;
    std::vector<std::vector<int>> stop_sequences;
    
    std::vector<std::vector<int>> generated_tokens;
    
    GeneratorWithKVCache() 
        : model(nullptr), max_length(100), eos_token_id(-1), 
          echo(false), repetition_penalty(1.0f) {}
    
    GeneratorWithKVCache(GPT* model_ptr, int max_length, int eos_token_id = -1) 
        : model(model_ptr), max_length(max_length), eos_token_id(eos_token_id),
          echo(false), repetition_penalty(1.0f) {}
    
    void set_model(GPT* model_ptr) {
        model = model_ptr;
    }
    
    void reset() {
        generated_tokens.clear();
    }
    
    std::string generate(Tokenizer& tokenizer, const std::string& prompt) {
        std::vector<int> input_ids = tokenizer.encode(prompt);
        
        std::string result;
        if (echo) {
            result = prompt;
        }
        
        result += generate_from_tokens(tokenizer, input_ids);
        return result;
    }
    
    std::string generate_from_tokens(Tokenizer& tokenizer, 
                                   const std::vector<int>& input_ids) {
        std::vector<int> tokens = input_ids;
        std::string result;
        
        for (int i = 0; i < max_length; i++) {
            if (!model || (int)tokens.size() > model->config.max_seq_len) {
                break;
            }
            
            Tensor logits = model->forward(tokens);
            
            int next_token;
            if (repetition_penalty != 1.0f && !tokens.empty()) {
                std::vector<int> recent_tokens;
                int start = std::max(0, (int)tokens.size() - 20);
                for (size_t j = start; j < tokens.size(); j++) {
                    recent_tokens.push_back(tokens[j]);
                }
                next_token = sampler.sample_with_penalty(logits, recent_tokens, repetition_penalty);
            } else {
                next_token = sampler.sample(logits);
            }
            
            if (next_token == eos_token_id) {
                break;
            }
            
            if (detect_stop(tokens)) {
                break;
            }
            
            result += tokenizer.decode({next_token});
            tokens.push_back(next_token);
            generated_tokens.push_back(tokens);
        }
        
        return result;
    }
    
    int step(Tokenizer& tokenizer, int next_token) {
        if (!model || generated_tokens.empty() || 
            (int)generated_tokens.back().size() > model->config.max_seq_len) {
            return -1;
        }
        
        generated_tokens.back().push_back(next_token);
        
        Tensor logits = model->forward(generated_tokens.back());
        int new_token = sampler.sample(logits);
        
        return new_token;
    }
    
    std::string finish(Tokenizer& tokenizer) {
        if (generated_tokens.empty()) {
            return "";
        }
        
        const auto& tokens = generated_tokens.back();
        return tokenizer.decode(tokens);
    }
    
    bool detect_stop(const std::vector<int>& tokens) {
        if (stop_sequences.empty()) {
            return false;
        }
        
        for (size_t t = tokens.size(); t > 0; t--) {
            for (const auto& seq : stop_sequences) {
                if (!seq.empty() && t >= seq.size()) {
                    bool match = true;
                    for (size_t i = 0; i < seq.size(); i++) {
                        if (tokens[t - seq.size() + i] != (int)seq[i]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    void add_stop_sequence(const std::vector<int>& seq) {
        stop_sequences.push_back(seq);
    }
};

}
