#pragma once
#include "../model/attention.hpp"
#include "../model/embedding.hpp"

namespace microgpt {

class TransformerBlock {
public:
    int layer_idx;
    int embed_dim;
    int num_heads;
    int hidden_dim;
    
    RMSNorm ln_1;
    RMSNorm ln_2;
    MultiHeadAttention attn;
    FeedForward mlp;
    
    TransformerBlock() = default;
    TransformerBlock(int layer_idx, int embed_dim, int num_heads, int hidden_dim);
    
    Tensor forward(const Tensor& x, bool use_cache = false);
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class Transformer {
public:
    int vocab_size;
    int embed_dim;
    int num_layers;
    int num_heads;
    int hidden_dim;
    int max_seq_len;
    
    Embedding embedding;
    std::vector<TransformerBlock> layers;
    RMSNorm ln_f;
    Linear lm_head;
    
    Transformer() = default;
    Transformer(int vocab_size, int embed_dim, int num_layers, int num_heads, int max_seq_len);
    
    Tensor forward(const std::vector<int>& token_ids);
    std::vector<Tensor> forward_with_cache(const std::vector<int>& token_ids);
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

}
