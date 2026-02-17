#pragma once
#include "../model/transformer.hpp"
#include <string>
#include <vector>

namespace microgpt {

struct ModelConfig {
    int vocab_size = 256;
    int embed_dim = 256;
    int num_layers = 6;
    int num_heads = 4;
    int max_seq_len = 256;
    int hidden_dim = 0;
    
    int recursion_steps = 3;
    float exit_threshold = 0.8f;
    int min_recursion_steps = 1;
};

class GPT {
public:
    ModelConfig config;
    Transformer transformer;
    
    int exit_token_id;
    
    GPT() = default;
    GPT(const ModelConfig& config);
    
    Tensor forward(const std::vector<int>& token_ids);
    Tensor forward_single(int token_id, int pos_id);
    
    int predict_next(const Tensor& logits);
    float cross_entropy(const Tensor& logits, int target_id);
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
    void save(const std::string& path);
    void load(const std::string& path);
};

class RL_GPT : public GPT {
public:
    Linear exit_head;
    
    RL_GPT() = default;
    RL_GPT(const ModelConfig& config);
    
    struct RecursionResult {
        Tensor logits;
        int steps_taken;
        bool exited_early;
    };
    
    RecursionResult recursive_forward(const std::vector<int>& token_ids);
    float get_exit_prob(const Tensor& state);
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

}
