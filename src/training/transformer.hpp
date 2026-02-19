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
    
    bool use_flash_attention;
    
    // KV cache for autoregressive generation
    bool use_kv_cache;
    Tensor k_cache_kv;  // Cached keys for generation
    Tensor v_cache_kv;  // Cached values for generation
    int cache_seq_len;
    
    MultiHeadAttention() = default;
    MultiHeadAttention(int embed_dim, int num_heads);
    
    Tensor forward(const Tensor& x, bool causal = true);
    Tensor forward_flash_attention(const Tensor& x, bool causal = true);
    
    // Forward with KV cache for generation
    Tensor forward_with_kv_cache(const Tensor& x, bool use_cache = true);
    
    Tensor backward(const Tensor& grad_output);
    Tensor backward_flash_attention(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
    
    void set_use_flash_attention(bool use) { use_flash_attention = use; }
    void set_use_kv_cache(bool use) { use_kv_cache = use; }
    void clear_kv_cache();
    void reset_cache() { cache_seq_len = 0; }
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
