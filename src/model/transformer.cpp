#include "transformer.hpp"
#include "../utils/logger.hpp"

namespace microgpt {

TransformerBlock::TransformerBlock(int layer_idx, int embed_dim, int num_heads, int hidden_dim)
    : layer_idx(layer_idx), embed_dim(embed_dim), num_heads(num_heads), hidden_dim(hidden_dim) {
    
    ln_1 = RMSNorm(embed_dim);
    ln_2 = RMSNorm(embed_dim);
    attn = MultiHeadAttention(embed_dim, num_heads, false);
    mlp = FeedForward(embed_dim, hidden_dim);
}

Tensor TransformerBlock::forward(const Tensor& x_input, bool use_cache) {
    Tensor x = x_input;
    Tensor x_residual = x;
    Tensor normalized = ln_1.forward(x);
    
    Tensor attn_out;
    if (use_cache) {
        attn_out = attn.forward_with_cache(normalized);
    } else {
        attn_out = attn.forward(normalized);
    }
    
    Tensor result(x.shape, false);
    for (int i = 0; i < result.size(); i++) {
        result.data[i] = x_residual.data[i] + attn_out.data[i];
    }
    
    x = result;
    x_residual = x;
    normalized = ln_2.forward(x);
    Tensor mlp_out = mlp.forward(normalized);
    
    for (int i = 0; i < result.size(); i++) {
        result.data[i] = x_residual.data[i] + mlp_out.data[i];
    }
    
    return result;
}

void TransformerBlock::init_weights(float std) {
    ln_1.init_weights();
    ln_2.init_weights();
    attn.init_weights(std);
    mlp.init_weights(std);
}

int TransformerBlock::num_parameters() const {
    return ln_1.num_parameters() + ln_2.num_parameters() + attn.num_parameters() + mlp.num_parameters();
}

Transformer::Transformer(int vocab_size, int embed_dim, int num_layers, int num_heads, int max_seq_len)
    : vocab_size(vocab_size), embed_dim(embed_dim), num_layers(num_layers), 
      num_heads(num_heads), max_seq_len(max_seq_len) {
    
    hidden_dim = embed_dim * 4;
    
    embedding = Embedding(vocab_size, embed_dim, max_seq_len);
    
    for (int i = 0; i < num_layers; i++) {
        layers.push_back(TransformerBlock(i, embed_dim, num_heads, hidden_dim));
    }
    
    ln_f = RMSNorm(embed_dim);
    lm_head = Linear(embed_dim, vocab_size, false);
}

Tensor Transformer::forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    
    std::vector<int> pos_ids(seq_len);
    for (int i = 0; i < seq_len; i++) {
        pos_ids[i] = i;
    }
    
    Tensor x = embedding.forward(token_ids, pos_ids);
    
    for (auto& layer : layers) {
        x = layer.forward(x);
    }
    
    x = ln_f.forward(x);
    
    x = lm_head.forward(x);
    
    return x;
}

std::vector<Tensor> Transformer::forward_with_cache(const std::vector<int>& token_ids) {
    std::vector<Tensor> logits_list;
    
    for (size_t i = 0; i < token_ids.size(); i++) {
        std::vector<int> single_token = {token_ids[i]};
        std::vector<int> pos_ids = {static_cast<int>(i)};
        
        Tensor x = embedding.forward(single_token, pos_ids);
        
        for (auto& layer : layers) {
            x = layer.forward(x, i > 0);
        }
        
        x = ln_f.forward(x);
        Tensor logits = lm_head.forward(x);
        
        logits_list.push_back(logits);
    }
    
    return logits_list;
}

void Transformer::init_weights(float std) {
    embedding.init_weights(std);
    
    for (auto& layer : layers) {
        layer.init_weights(std);
    }
    
    ln_f.init_weights();
    lm_head.init_weights(std);
}

int Transformer::num_parameters() const {
    int total = embedding.num_parameters() + ln_f.num_parameters() + lm_head.num_parameters();
    
    for (const auto& layer : layers) {
        total += layer.num_parameters();
    }
    
    return total;
}

}
