#include "model.hpp"
#include "../utils/logger.hpp"
#include "../utils/random.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace microgpt {

GPT::GPT(const ModelConfig& config) : config(config) {
    ModelConfig cfg = config;
    if (cfg.hidden_dim == 0) {
        cfg.hidden_dim = cfg.embed_dim * 4;
    }
    
    transformer = Transformer(cfg.vocab_size, cfg.embed_dim, cfg.num_layers, cfg.num_heads, cfg.max_seq_len);
    exit_token_id = cfg.vocab_size - 1;
}

Tensor GPT::forward(const std::vector<int>& token_ids) {
    return transformer.forward(token_ids);
}

Tensor GPT::forward_single(int token_id, int pos_id) {
    Tensor token_emb = transformer.embedding.token_embedding(token_id);
    Tensor pos_emb = transformer.embedding.pos_embedding(pos_id);
    
    Tensor x(Shape{1, config.embed_dim}, false);
    for (int i = 0; i < config.embed_dim; i++) {
        x.data[i] = token_emb.data[i] + pos_emb.data[i];
    }
    
    for (auto& layer : transformer.layers) {
        x = layer.forward(x);
    }
    
    x = transformer.ln_f.forward(x);
    Tensor logits = transformer.lm_head.forward(x);
    
    return logits;
}

int GPT::predict_next(const Tensor& logits) {
    int seq_len = logits.rows();
    int vocab = logits.cols();
    
    int last_idx = seq_len - 1;
    
    float max_val = logits.data[last_idx * vocab];
    int max_id = 0;
    for (int i = 1; i < vocab; i++) {
        float val = logits.data[last_idx * vocab + i];
        if (val > max_val) {
            max_val = val;
            max_id = i;
        }
    }
    
    return max_id;
}

float GPT::cross_entropy(const Tensor& logits, int target_id) {
    int seq_len = logits.rows();
    int vocab = logits.cols();
    
    int last_idx = seq_len - 1;
    
    float max_val = logits.data[last_idx * vocab];
    for (int i = 1; i < vocab; i++) {
        max_val = std::max(max_val, logits.data[last_idx * vocab + i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab; i++) {
        sum_exp += std::exp(logits.data[last_idx * vocab + i] - max_val);
    }
    
    float prob = std::exp(logits.data[last_idx * vocab + target_id] - max_val) / sum_exp;
    prob = std::max(1e-7f, std::min(1e-7f, prob));
    
    return -std::log(prob);
}

void GPT::init_weights(float std) {
    transformer.init_weights(std);
}

int GPT::num_parameters() const {
    return transformer.num_parameters();
}

void GPT::save(const std::string& path) {
    Logger::info("Saving model to " + path);
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to open file for saving: " + path);
        return;
    }
    
    // Save config
    file.write(reinterpret_cast<const char*>(&config.vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.embed_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.num_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.max_seq_len), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.hidden_dim), sizeof(int));
    
    // Save embedding weights
    const auto& wte = transformer.embedding.wte;
    file.write(reinterpret_cast<const char*>(wte.data.data()), sizeof(float) * wte.data.size());
    
    const auto& wpe = transformer.embedding.wpe;
    file.write(reinterpret_cast<const char*>(wpe.data.data()), sizeof(float) * wpe.data.size());
    
    // Save lm_head
    const auto& lm_head_w = transformer.lm_head.weight;
    file.write(reinterpret_cast<const char*>(lm_head_w.data.data()), sizeof(float) * lm_head_w.data.size());
    
    // Save ln_f
    const auto& ln_f_w = transformer.ln_f.weight;
    file.write(reinterpret_cast<const char*>(ln_f_w.data.data()), sizeof(float) * ln_f_w.data.size());
    
    // Save each layer
    for (const auto& layer : transformer.layers) {
        // Attention weights
        const auto& wq = layer.attn.wq.weight;
        const auto& wk = layer.attn.wk.weight;
        const auto& wv = layer.attn.wv.weight;
        const auto& wo = layer.attn.wo.weight;
        
        file.write(reinterpret_cast<const char*>(wq.data.data()), sizeof(float) * wq.data.size());
        file.write(reinterpret_cast<const char*>(wk.data.data()), sizeof(float) * wk.data.size());
        file.write(reinterpret_cast<const char*>(wv.data.data()), sizeof(float) * wv.data.size());
        file.write(reinterpret_cast<const char*>(wo.data.data()), sizeof(float) * wo.data.size());
        
        // MLP weights
        const auto& fc1 = layer.mlp.fc1.weight;
        const auto& fc2 = layer.mlp.fc2.weight;
        
        file.write(reinterpret_cast<const char*>(fc1.data.data()), sizeof(float) * fc1.data.size());
        file.write(reinterpret_cast<const char*>(fc2.data.data()), sizeof(float) * fc2.data.size());
        
        // Layer norms
        const auto& ln1_w = layer.ln_1.weight;
        const auto& ln2_w = layer.ln_2.weight;
        
        file.write(reinterpret_cast<const char*>(ln1_w.data.data()), sizeof(float) * ln1_w.data.size());
        file.write(reinterpret_cast<const char*>(ln2_w.data.data()), sizeof(float) * ln2_w.data.size());
    }
    
    file.close();
    Logger::info("Model saved successfully!");
}

void GPT::load(const std::string& path) {
    Logger::info("Loading model from " + path);
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to open file for loading: " + path);
        return;
    }
    
    // Load config
    ModelConfig loaded_config;
    file.read(reinterpret_cast<char*>(&loaded_config.vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.embed_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.max_seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.hidden_dim), sizeof(int));
    
    // Check if config matches
    if (loaded_config.vocab_size != config.vocab_size || 
        loaded_config.embed_dim != config.embed_dim ||
        loaded_config.num_layers != config.num_layers) {
        Logger::error("Model configuration mismatch! Loading random weights instead.");
        file.close();
        init_weights(0.02f);
        return;
    }
    
    // Load embedding weights
    auto& wte = transformer.embedding.wte;
    file.read(reinterpret_cast<char*>(wte.data.data()), sizeof(float) * wte.data.size());
    
    auto& wpe = transformer.embedding.wpe;
    file.read(reinterpret_cast<char*>(wpe.data.data()), sizeof(float) * wpe.data.size());
    
    // Load lm_head
    auto& lm_head_w = transformer.lm_head.weight;
    file.read(reinterpret_cast<char*>(lm_head_w.data.data()), sizeof(float) * lm_head_w.data.size());
    
    // Load ln_f
    auto& ln_f_w = transformer.ln_f.weight;
    file.read(reinterpret_cast<char*>(ln_f_w.data.data()), sizeof(float) * ln_f_w.data.size());
    
    // Load each layer
    for (auto& layer : transformer.layers) {
        // Attention weights
        auto& wq = layer.attn.wq.weight;
        auto& wk = layer.attn.wk.weight;
        auto& wv = layer.attn.wv.weight;
        auto& wo = layer.attn.wo.weight;
        
        file.read(reinterpret_cast<char*>(wq.data.data()), sizeof(float) * wq.data.size());
        file.read(reinterpret_cast<char*>(wk.data.data()), sizeof(float) * wk.data.size());
        file.read(reinterpret_cast<char*>(wv.data.data()), sizeof(float) * wv.data.size());
        file.read(reinterpret_cast<char*>(wo.data.data()), sizeof(float) * wo.data.size());
        
        // MLP weights
        auto& fc1 = layer.mlp.fc1.weight;
        auto& fc2 = layer.mlp.fc2.weight;
        
        file.read(reinterpret_cast<char*>(fc1.data.data()), sizeof(float) * fc1.data.size());
        file.read(reinterpret_cast<char*>(fc2.data.data()), sizeof(float) * fc2.data.size());
        
        // Layer norms
        auto& ln1_w = layer.ln_1.weight;
        auto& ln2_w = layer.ln_2.weight;
        
        file.read(reinterpret_cast<char*>(ln1_w.data.data()), sizeof(float) * ln1_w.data.size());
        file.read(reinterpret_cast<char*>(ln2_w.data.data()), sizeof(float) * ln2_w.data.size());
    }
    
    file.close();
    Logger::info("Model loaded successfully!");
}

RL_GPT::RL_GPT(const ModelConfig& config) : GPT(config) {
    exit_head = Linear(config.embed_dim, 1, false);
    init_weights();
}

RL_GPT::RecursionResult RL_GPT::recursive_forward(const std::vector<int>& token_ids) {
    RecursionResult result;
    result.steps_taken = 0;
    result.exited_early = false;
    
    std::vector<int> pos_ids(token_ids.size());
    for (size_t i = 0; i < token_ids.size(); i++) {
        pos_ids[i] = static_cast<int>(i);
    }
    
    Tensor state = transformer.embedding.forward(token_ids, pos_ids);
    
    for (int step = 0; step < config.recursion_steps; step++) {
        Tensor normalized = transformer.ln_f.forward(state);
        
        Tensor new_state = normalized;
        for (auto& layer : transformer.layers) {
            new_state = layer.forward(new_state);
        }
        
        for (int i = 0; i < state.size(); i++) {
            state.data[i] = state.data[i] + new_state.data[i];
        }
        
        Tensor exit_logits = exit_head.forward(state);
        
        if (step >= config.min_recursion_steps - 1) {
            int last_idx = state.rows() - 1;
            float exit_prob = 1.0f / (1.0f + std::exp(-exit_logits.data[last_idx]));
            
            if (exit_prob > config.exit_threshold) {
                result.exited_early = true;
                break;
            }
        }
        
        result.steps_taken++;
    }
    
    state = transformer.ln_f.forward(state);
    Tensor logits = transformer.lm_head.forward(state);
    result.logits = logits;
    
    return result;
}

float RL_GPT::get_exit_prob(const Tensor& state) {
    int last_idx = state.rows() - 1;
    Tensor exit_logits = exit_head.forward(state);
    float exit_score = exit_logits.data[last_idx];
    return 1.0f / (1.0f + std::exp(-exit_score));
}

void RL_GPT::init_weights(float std) {
    GPT::init_weights(std);
    math::fill(exit_head.weight, 0.0f);
}

int RL_GPT::num_parameters() const {
    return GPT::num_parameters() + exit_head.num_parameters();
}

}
