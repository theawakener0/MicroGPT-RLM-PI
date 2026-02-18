#include "trainable_model.hpp"
#include "../utils/logger.hpp"
#include "../utils/random.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace microgpt {

TrainableGPT::TrainableGPT(const ModelConfig& config) : config(config) {
    ModelConfig cfg = config;
    if (cfg.hidden_dim == 0) {
        cfg.hidden_dim = cfg.embed_dim * 4;
    }
    
    token_embedding = Embedding(cfg.vocab_size, cfg.embed_dim);
    pos_embedding = PositionalEmbedding(cfg.max_seq_len, cfg.embed_dim);
    
    for (int i = 0; i < cfg.num_layers; i++) {
        layers.push_back(TransformerBlock(i, cfg.embed_dim, cfg.num_heads, cfg.hidden_dim));
    }
    
    ln_f = RMSNorm(cfg.embed_dim);
    lm_head = Linear(cfg.embed_dim, cfg.vocab_size, false);
    
    init_weights();
    
    all_params.clear();
    for (auto& layer : layers) {
        for (auto* p : layer.parameters()) {
            all_params.push_back(p);
        }
    }
    for (auto* p : token_embedding.parameters()) {
        all_params.push_back(p);
    }
    for (auto* p : pos_embedding.parameters()) {
        all_params.push_back(p);
    }
    for (auto* p : ln_f.parameters()) {
        all_params.push_back(p);
    }
    for (auto* p : lm_head.parameters()) {
        all_params.push_back(p);
    }
}

TrainableGPT::ForwardOutput TrainableGPT::forward(const std::vector<int>& input_ids, 
                                                 const std::vector<int>& target_ids) {
    int seq_len = input_ids.size();
    
    Tensor token_emb = token_embedding.forward(input_ids);
    Tensor pos_emb = pos_embedding.forward(seq_len);
    
    Tensor x(token_emb.shape, false);
    for (int i = 0; i < x.size(); i++) {
        x.data[i] = token_emb.data[i] + pos_emb.data[i];
    }
    
    for (auto& layer : layers) {
        x = layer.forward(x);
    }
    
    x = ln_f.forward(x);
    Tensor logits = lm_head.forward(x);
    
    float loss = 0.0f;
    for (int i = 0; i < seq_len && i < (int)target_ids.size(); i++) {
        loss += cross_entropy(logits, target_ids[i], i);
    }
    loss /= seq_len;
    
    ForwardOutput output;
    output.logits = logits;
    output.loss = loss;
    return output;
}

void TrainableGPT::backward(const ForwardOutput& output, const std::vector<int>& target_ids) {
    zero_grad();
    
    int seq_len = output.logits.rows();
    int vocab = output.logits.cols();
    
    Tensor grad_logits(output.logits.shape, false);
    float grad_scale = 1.0f / seq_len;
    
    for (int i = 0; i < seq_len && i < (int)target_ids.size(); i++) {
        int target = target_ids[i];
        
        float max_val = output.logits.data[i * vocab];
        for (int j = 1; j < vocab; j++) {
            max_val = std::max(max_val, output.logits.data[i * vocab + j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            sum_exp += std::exp(output.logits.data[i * vocab + j] - max_val);
        }
        
        for (int j = 0; j < vocab; j++) {
            float prob = std::exp(output.logits.data[i * vocab + j] - max_val) / sum_exp;
            float target_prob = (j == target) ? 1.0f : 0.0f;
            grad_logits.data[i * vocab + j] = (prob - target_prob) * grad_scale;
        }
    }
    
    Tensor grad_lm = lm_head.backward(grad_logits);
    Tensor grad_ln = ln_f.backward(grad_lm);
    
    for (int i = (int)layers.size() - 1; i >= 0; i--) {
        layers[i].backward(grad_ln);
    }
    
    token_embedding.backward(grad_ln);
}

void TrainableGPT::update_weights(float learning_rate) {
    for (auto* param : all_params) {
        if (param->grad.empty()) continue;
        
        for (int i = 0; i < param->size(); i++) {
            param->data[i] -= learning_rate * param->grad[i];
        }
    }
}

void TrainableGPT::zero_grad() {
    token_embedding.zero_grad();
    pos_embedding.zero_grad();
    for (auto& layer : layers) {
        layer.zero_grad();
    }
    ln_f.zero_grad();
    lm_head.zero_grad();
}

void TrainableGPT::init_weights(float std) {
    token_embedding.init_weights(std);
    for (auto& layer : layers) {
        layer.init_weights(std);
    }
    ln_f.init_weights();
    lm_head.init_weights(std);
}

int TrainableGPT::num_parameters() const {
    int total = token_embedding.num_parameters();
    total += pos_embedding.num_parameters();
    for (const auto& layer : layers) {
        total += layer.num_parameters();
    }
    total += ln_f.num_parameters();
    total += lm_head.num_parameters();
    return total;
}

int TrainableGPT::predict_next(const Tensor& logits) {
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

float TrainableGPT::cross_entropy(const Tensor& logits, int target_id, int pos) {
    int vocab = logits.cols();
    
    float max_val = logits.data[pos * vocab];
    for (int j = 1; j < vocab; j++) {
        max_val = std::max(max_val, logits.data[pos * vocab + j]);
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < vocab; j++) {
        sum_exp += std::exp(logits.data[pos * vocab + j] - max_val);
    }
    
    float prob = std::exp(logits.data[pos * vocab + target_id] - max_val) / sum_exp;
    prob = std::max(1e-7f, std::min(1.0f - 1e-7f, prob));
    
    return -std::log(prob);
}

void TrainableGPT::save(const std::string& path) {
    Logger::info("Saving model to " + path);
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to open file for saving: " + path);
        return;
    }
    
    file.write(reinterpret_cast<const char*>(&config.vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.embed_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.num_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.max_seq_len), sizeof(int));
    file.write(reinterpret_cast<const char*>(&config.hidden_dim), sizeof(int));
    
    auto save_tensor = [&](const Tensor& t) {
        file.write(reinterpret_cast<const char*>(t.data.data()), sizeof(float) * t.size());
    };
    
    save_tensor(token_embedding.weight);
    save_tensor(lm_head.weight);
    save_tensor(ln_f.weight);
    
    for (const auto& layer : layers) {
        for (auto* p : layer.attn.parameters()) {
            save_tensor(*p);
        }
        for (auto* p : layer.mlp.parameters()) {
            save_tensor(*p);
        }
    }
    
    file.close();
    Logger::info("Model saved!");
}

void TrainableGPT::load(const std::string& path) {
    Logger::info("Loading model from " + path);
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        Logger::error("Failed to open file: " + path);
        return;
    }
    
    ModelConfig loaded_config;
    file.read(reinterpret_cast<char*>(&loaded_config.vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.embed_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.num_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.max_seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_config.hidden_dim), sizeof(int));
    
    if (loaded_config.vocab_size != config.vocab_size || 
        loaded_config.embed_dim != config.embed_dim) {
        Logger::error("Config mismatch!");
        file.close();
        return;
    }
    
    auto load_tensor = [&](Tensor& t) {
        file.read(reinterpret_cast<char*>(t.data.data()), sizeof(float) * t.size());
    };
    
    load_tensor(token_embedding.weight);
    load_tensor(lm_head.weight);
    load_tensor(ln_f.weight);
    
    for (auto& layer : layers) {
        for (auto* p : layer.attn.parameters()) {
            load_tensor(*p);
        }
        for (auto* p : layer.mlp.parameters()) {
            load_tensor(*p);
        }
    }
    
    file.close();
    Logger::info("Model loaded!");
}

RL_GPT::RL_GPT(const ModelConfig& config) : TrainableGPT(config) {
    exit_head = Linear(config.embed_dim, 1, false);
    init_weights();
    
    for (auto* p : exit_head.parameters()) {
        all_params.push_back(p);
    }
}

RL_GPT::RLResult RL_GPT::recursive_forward(const std::vector<int>& input_ids,
                                           const std::vector<int>& target_ids) {
    RLResult result;
    result.steps_used = 0;
    result.exited_early = false;
    
    int seq_len = input_ids.size();
    
    Tensor token_emb = token_embedding.forward(input_ids);
    Tensor pos_emb = pos_embedding.forward(seq_len);
    
    Tensor state(token_emb.shape, false);
    for (int i = 0; i < state.size(); i++) {
        state.data[i] = token_emb.data[i] + pos_emb.data[i];
    }
    
    for (int step = 0; step < config.recursion_steps; step++) {
        state = ln_f.forward(state);
        
        for (auto& layer : layers) {
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
        
        result.steps_used++;
    }
    
    state = ln_f.forward(state);
    Tensor logits = lm_head.forward(state);
    
    float loss = 0.0f;
    for (int i = 0; i < seq_len && i < (int)target_ids.size(); i++) {
        loss += cross_entropy(logits, target_ids[i], i);
    }
    loss /= seq_len;
    
    result.logits = logits;
    result.loss = loss;
    return result;
}

void RL_GPT::init_weights(float std) {
    TrainableGPT::init_weights(std);
    math::fill(exit_head.weight, 0.0f);
}

int RL_GPT::num_parameters() const {
    return TrainableGPT::num_parameters() + exit_head.num_parameters();
}

}
