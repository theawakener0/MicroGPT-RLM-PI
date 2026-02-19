#pragma once
#include "../core/tensor.hpp"
#include "../training/nn.hpp"
#include "../training/transformer.hpp"
#include <vector>
#include <string>

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

class TrainableGPT {
public:
    ModelConfig config;
    
    Embedding token_embedding;
    PositionalEmbedding pos_embedding;
    std::vector<TransformerBlock> layers;
    RMSNorm ln_f;
    Linear lm_head;
    
    std::vector<Tensor*> all_params;
    
    TrainableGPT() = default;
    TrainableGPT(const ModelConfig& config);
    
    struct ForwardOutput {
        Tensor logits;
        float loss;
    };
    
    Tensor forward(const std::vector<int>& input_ids);
    Tensor forward_single(int token_id, int pos_id);
    ForwardOutput forward(const std::vector<int>& input_ids, 
                         const std::vector<int>& target_ids);
    
    void backward(const ForwardOutput& output, const std::vector<int>& target_ids);
    
    void update_weights(float learning_rate);
    
    void zero_grad();
    
    void init_weights(float std = 0.02f);
    
    int num_parameters() const;
    
    int predict_next(const Tensor& logits);
    
    float cross_entropy(const Tensor& logits, int target_id, int pos);
    
    void save(const std::string& path);
    void load(const std::string& path);
};

class RL_GPT : public TrainableGPT {
public:
    Linear exit_head;
    
    RL_GPT() = default;
    RL_GPT(const ModelConfig& config);
    
    struct RLResult {
        Tensor logits;
        float loss;
        int steps_used;
        bool exited_early;
    };
    
    RLResult recursive_forward(const std::vector<int>& input_ids,
                               const std::vector<int>& target_ids);
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

}
