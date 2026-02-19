#pragma once
#include "../core/tensor.hpp"
#include "../core/math_ops.hpp"
#include <vector>

namespace microgpt {

class Embedding {
public:
    int vocab_size;
    int embed_dim;
    int max_seq_len;
    
    Tensor wte;  // Token embeddings (vocab_size x embed_dim)
    Tensor wpe;  // Positional embeddings (max_seq_len x embed_dim)
    
    Embedding() = default;
    Embedding(int vocab_size, int embed_dim, int max_seq_len);
    
    Tensor forward(const std::vector<int>& token_ids, const std::vector<int>& pos_ids);
    Tensor token_embedding(int token_id);
    Tensor pos_embedding(int pos_id);
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class Linear {
public:
    int in_features;
    int out_features;
    bool use_bias;
    
    Tensor weight;
    Tensor bias;
    
    Linear() = default;
    Linear(int in_features, int out_features, bool use_bias = false);
    
    Tensor forward(const Tensor& x);
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

}
