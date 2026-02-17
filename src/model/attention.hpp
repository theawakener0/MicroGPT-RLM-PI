#pragma once
#include "../core/tensor.hpp"
#include "../model/embedding.hpp"
#include <vector>
#include <memory>

namespace microgpt {

struct KVCache {
    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    
    void reset() {
        keys.clear();
        values.clear();
    }
    
    void append(const Tensor& k, const Tensor& v) {
        keys.push_back(k);
        values.push_back(v);
    }
    
    int size() const {
        return static_cast<int>(keys.size());
    }
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
    
    bool use_cache;
    KVCache kv_cache;
    
    MultiHeadAttention() = default;
    MultiHeadAttention(int embed_dim, int num_heads, bool use_cache = false);
    
    Tensor forward(const Tensor& x, bool is_causal = true);
    Tensor forward_with_cache(const Tensor& x);
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class FeedForward {
public:
    int embed_dim;
    int hidden_dim;
    
    Linear fc1;
    Linear fc2;
    
    FeedForward() = default;
    FeedForward(int embed_dim, int hidden_dim);
    
    Tensor forward(const Tensor& x);
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class RMSNorm {
public:
    int embed_dim;
    Tensor weight;
    float eps;
    
    RMSNorm() = default;
    RMSNorm(int embed_dim, float eps = 1e-5f);
    
    Tensor forward(const Tensor& x);
    void init_weights();
    int num_parameters() const;
};

}
