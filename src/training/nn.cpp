#include "nn.hpp"
#include "../utils/random.hpp"
#include <cmath>
#include <algorithm>

namespace microgpt {

Linear::Linear(int in_features, int out_features, bool use_bias)
    : in_features(in_features), out_features(out_features), use_bias(use_bias) {
    
    weight = Tensor(Shape{out_features, in_features}, true);
    if (use_bias) {
        bias = Tensor(Shape{out_features}, true);
    }
    weight_grad = Tensor(Shape{out_features, in_features}, false);
    if (use_bias) {
        bias_grad = Tensor(Shape{out_features}, false);
    }
    init_weights();
}

Tensor Linear::forward(const Tensor& x) {
    int batch_size = x.rows();
    int in_feat = x.cols();
    
    input_cache = x;
    
    Tensor result(Shape{batch_size, out_features}, false);
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (int k = 0; k < in_feat; k++) {
                sum += x.data[i * in_feat + k] * weight.data[j * in_feat + k];
            }
            if (use_bias) {
                sum += bias.data[j];
            }
            result.data[i * out_features + j] = sum;
        }
    }
    
    output_cache = result;
    return result;
}

Tensor Linear::backward(const Tensor& grad_output) {
    int batch_size = input_cache.rows();
    int in_feat = input_cache.cols();
    
    weight_grad = Tensor(Shape{out_features, in_feat}, false);
    
    for (int j = 0; j < out_features; j++) {
        for (int k = 0; k < in_feat; k++) {
            float grad_sum = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                grad_sum += grad_output.data[i * out_features + j] * input_cache.data[i * in_feat + k];
            }
            weight_grad.data[j * in_feat + k] = grad_sum;
        }
    }
    
    if (use_bias) {
        bias_grad = Tensor(Shape{out_features}, false);
        for (int j = 0; j < out_features; j++) {
            float grad_sum = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                grad_sum += grad_output.data[i * out_features + j];
            }
            bias_grad.data[j] = grad_sum;
        }
    }
    
    Tensor grad_input(Shape{batch_size, in_feat}, false);
    for (int i = 0; i < batch_size; i++) {
        for (int k = 0; k < in_feat; k++) {
            float sum = 0.0f;
            for (int j = 0; j < out_features; j++) {
                sum += grad_output.data[i * out_features + j] * weight.data[j * in_feat + k];
            }
            grad_input.data[i * in_feat + k] = sum;
        }
    }
    
    return grad_input;
}

void Linear::zero_grad() {
    math::fill(weight_grad, 0.0f);
    if (use_bias) {
        math::fill(bias_grad, 0.0f);
    }
}

std::vector<Tensor*> Linear::parameters() {
    std::vector<Tensor*> params;
    params.push_back(&weight);
    if (use_bias) {
        params.push_back(&bias);
    }
    return params;
}

void Linear::init_weights(float std) {
    math::normal_(weight, 0.0f, std);
    if (use_bias) {
        math::fill(bias, 0.0f);
    }
}

int Linear::num_parameters() const {
    return out_features * in_features + (use_bias ? out_features : 0);
}

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings(num_embeddings), embedding_dim(embedding_dim) {
    
    weight = Tensor(Shape{num_embeddings, embedding_dim}, true);
    weight_grad = Tensor(Shape{num_embeddings, embedding_dim}, false);
    init_weights();
}

Tensor Embedding::forward(const std::vector<int>& indices) {
    int seq_len = indices.size();
    input_cache = indices;
    
    Tensor result(Shape{seq_len, embedding_dim}, false);
    
    for (int i = 0; i < seq_len; i++) {
        int idx = indices[i];
        for (int j = 0; j < embedding_dim; j++) {
            result.data[i * embedding_dim + j] = weight.data[idx * embedding_dim + j];
        }
    }
    
    output_cache = result;
    return result;
}

Tensor Embedding::backward(const Tensor& grad_output) {
    int seq_len = grad_output.rows();
    
    math::fill(weight_grad, 0.0f);
    
    for (int i = 0; i < seq_len; i++) {
        int idx = input_cache[i];
        for (int j = 0; j < embedding_dim; j++) {
            weight_grad.data[idx * embedding_dim + j] += grad_output.data[i * embedding_dim + j];
        }
    }
    
    Tensor grad_input(Shape{seq_len, embedding_dim}, false);
    for (int i = 0; i < seq_len; i++) {
        int idx = input_cache[i];
        for (int j = 0; j < embedding_dim; j++) {
            grad_input.data[i * embedding_dim + j] = weight.data[idx * embedding_dim + j];
        }
    }
    
    return grad_input;
}

void Embedding::zero_grad() {
    math::fill(weight_grad, 0.0f);
}

std::vector<Tensor*> Embedding::parameters() {
    return {&weight};
}

void Embedding::init_weights(float std) {
    math::normal_(weight, 0.0f, std);
}

int Embedding::num_parameters() const {
    return num_embeddings * embedding_dim;
}

PositionalEmbedding::PositionalEmbedding(int max_seq_len, int embed_dim)
    : max_seq_len(max_seq_len), embed_dim(embed_dim) {
    
    weight = Tensor(Shape{max_seq_len, embed_dim}, true);
    weight_grad = Tensor(Shape{max_seq_len, embed_dim}, false);
    init_weights();
}

Tensor PositionalEmbedding::forward(int seq_len) {
    if (seq_len > max_seq_len) {
        seq_len = max_seq_len;
    }
    
    Tensor result(Shape{seq_len, embed_dim}, false);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            float angle = i / std::pow(10000.0f, (2.0f * (j / 2)) / embed_dim);
            if (j % 2 == 0) {
                result.data[i * embed_dim + j] = std::sin(angle);
            } else {
                result.data[i * embed_dim + j] = std::cos(angle);
            }
        }
    }
    
    return result;
}

void PositionalEmbedding::backward(const Tensor& grad_output) {
    int seq_len = grad_output.rows();
    math::fill(weight_grad, 0.0f);
}

void PositionalEmbedding::zero_grad() {
    math::fill(weight_grad, 0.0f);
}

std::vector<Tensor*> PositionalEmbedding::parameters() {
    return {};
}

void PositionalEmbedding::init_weights(float std) {
}

int PositionalEmbedding::num_parameters() const {
    return 0;
}

}
