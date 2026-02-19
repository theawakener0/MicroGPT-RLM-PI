#pragma once
#include "../core/tensor.hpp"
#include "../core/math_ops.hpp"
#include <vector>
#include <string>

namespace microgpt {

class Linear {
public:
    int in_features;
    int out_features;
    bool use_bias;
    
    Tensor weight;
    Tensor bias;
    Tensor weight_grad;
    Tensor bias_grad;
    
    Tensor input_cache;
    Tensor output_cache;
    
    Linear() = default;
    Linear(int in_features, int out_features, bool use_bias = false);
    
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class Embedding {
public:
    int num_embeddings;
    int embedding_dim;
    
    Tensor weight;
    Tensor weight_grad;
    
    std::vector<int> input_cache;
    Tensor output_cache;
    
    Embedding() = default;
    Embedding(int num_embeddings, int embedding_dim);
    
    Tensor forward(const std::vector<int>& indices);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class PositionalEmbedding {
public:
    int max_seq_len;
    int embed_dim;
    
    Tensor weight;
    Tensor weight_grad;
    
    PositionalEmbedding() = default;
    PositionalEmbedding(int max_seq_len, int embed_dim);
    
    Tensor forward(int seq_len);
    void backward(const Tensor& grad_output);
    void zero_grad();
    std::vector<Tensor*> parameters();
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

class QuantizedLinear {
public:
    int in_features;
    int out_features;
    bool use_bias;
    
    // Quantized weights (INT8)
    math::QuantizedTensor weight_quantized;
    Tensor bias;
    
    // For training: keep FP32 weights
    Tensor weight_fp32;
    Tensor weight_grad;
    Tensor bias_grad;
    
    Tensor input_cache;
    Tensor output_cache;
    
    bool quantized;
    bool use_per_channel;
    
    QuantizedLinear() = default;
    QuantizedLinear(int in_features, int out_features, bool use_bias = false);
    
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    
    void quantize_weights();
    void dequantize_weights();
    bool is_quantized() const { return quantized; }
    void set_per_channel(bool per_channel) { use_per_channel = per_channel; }
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
};

}
