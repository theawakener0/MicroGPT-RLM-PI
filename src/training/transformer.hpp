#pragma once
#include "../training/nn.hpp"
#include <vector>
#include <memory>

namespace microgpt {

class RMSNorm {
public:
    int embed_dim;
    float eps;
    Tensor weight;
    Tensor weight_grad;
    
    Tensor input_cache;
    Tensor output_cache;
    
    RMSNorm() = default;
    RMSNorm(int embed_dim, float eps = 1e-5f);
    
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights();
    int num_parameters() const;
};

class MultiHeadAttention {
public:
    int embed_dim;
    int num_heads;
    int head_dim;
    float scale;
    
    Linear wq;
    Linear wk;
    Linear wv;
    Linear wo;
    
    Tensor q_cache, k_cache, v_cache;
    Tensor attn_cache;
    Tensor output_cache;
    
    MultiHeadAttention() = default;
    MultiHeadAttention(int embed_dim, int num_heads);
    
    Tensor forward(const Tensor& x, bool causal = true);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class FeedForward {
public:
    int embed_dim;
    int hidden_dim;
    
    Linear fc1;
    Linear fc2;
    
    Tensor hidden_cache;
    Tensor output_cache;
    
    FeedForward() = default;
    FeedForward(int embed_dim, int hidden_dim);
    
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class TransformerBlock {
public:
    int layer_idx;
    int embed_dim;
    
    RMSNorm ln_1;
    RMSNorm ln_2;
    MultiHeadAttention attn;
    FeedForward mlp;
    
    Tensor residual_1_cache;
    Tensor residual_2_cache;
    
    TransformerBlock() = default;
    TransformerBlock(int layer_idx, int embed_dim, int num_heads, int hidden_dim);
    
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

}
