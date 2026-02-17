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
    prob = std::max(1e-7f, std::min(1.0f - 1e-7f, prob));
    
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
}

void GPT::load(const std::string& path) {
    Logger::info("Loading model from " + path);
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
        state = transformer.ln_f.forward(state);
        
        for (auto& layer : transformer.layers) {
            state = layer.forward(state);
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
