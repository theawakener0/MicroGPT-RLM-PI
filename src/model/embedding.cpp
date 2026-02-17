#include "embedding.hpp"
#include "../utils/logger.hpp"
#include "../utils/random.hpp"

namespace microgpt {

Embedding::Embedding(int vocab_size, int embed_dim, int max_seq_len)
    : vocab_size(vocab_size), embed_dim(embed_dim), max_seq_len(max_seq_len) {
    
    wte = Tensor(Shape{vocab_size, embed_dim}, true);
    wpe = Tensor(Shape{max_seq_len, embed_dim}, true);
    
    init_weights();
}

Tensor Embedding::forward(const std::vector<int>& token_ids, const std::vector<int>& pos_ids) {
    int seq_len = token_ids.size();
    Tensor result(Shape{seq_len, embed_dim}, false);
    
    for (int i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        int pos_id = pos_ids[i];
        
        for (int j = 0; j < embed_dim; j++) {
            result.data[i * embed_dim + j] = 
                wte.data[token_id * embed_dim + j] + 
                wpe.data[pos_id * embed_dim + j];
        }
    }
    
    return result;
}

Tensor Embedding::token_embedding(int token_id) {
    Tensor result(Shape{embed_dim}, false);
    
    for (int j = 0; j < embed_dim; j++) {
        result.data[j] = wte.data[token_id * embed_dim + j];
    }
    
    return result;
}

Tensor Embedding::pos_embedding(int pos_id) {
    Tensor result(Shape{embed_dim}, false);
    
    for (int j = 0; j < embed_dim; j++) {
        result.data[j] = wpe.data[pos_id * embed_dim + j];
    }
    
    return result;
}

void Embedding::init_weights(float std) {
    math::normal_(wte, 0.0f, std);
    math::normal_(wpe, 0.0f, std);
}

int Embedding::num_parameters() const {
    return vocab_size * embed_dim + max_seq_len * embed_dim;
}

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features(in_features), out_features(out_features), bias(bias) {
    
    weight = Tensor(Shape{out_features, in_features}, true);
    if (bias) {
        bias_vec = Tensor(Shape{out_features}, true);
    }
    
    init_weights();
}

Tensor Linear::forward(const Tensor& x) {
    int batch_size = x.rows();
    int in_feat = x.cols();
    
    Tensor result(Shape{batch_size, out_features}, false);
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int k = 0; k < in_feat; k++) {
                sum += x.data[i * in_feat + k] * weight.data[j * in_feat + k];
            }
            if (bias) {
                sum += bias_vec.data[j];
            }
            result.data[i * out_features + j] = sum;
        }
    }
    
    return result;
}

void Linear::init_weights(float std) {
    math::normal_(weight, 0.0f, std);
    if (bias) {
        math::fill(bias_vec, 0.0f);
    }
}

int Linear::num_parameters() const {
    return out_features * in_features + (bias ? out_features : 0);
}

}
