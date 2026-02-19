#pragma once
#include "tensor.hpp"
#include <vector>
#include <optional>

namespace microgpt {
namespace math {

// Element-wise operations
void add_inplace(Tensor& a, const Tensor& b);
void add_scalar_inplace(Tensor& a, float scalar);
void multiply_inplace(Tensor& a, const Tensor& b);
void multiply_scalar_inplace(Tensor& a, float scalar);
void relu_inplace(Tensor& a);
void sigmoid_inplace(Tensor& a);
void tanh_inplace(Tensor& a);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);  // a: (m,n), b: (n,k) -> (m,k)
Tensor transpose(const Tensor& a);

// Normalization
Tensor rmsnorm(const Tensor& x, float eps = 1e-5);
Tensor layernorm(const Tensor& x, float eps = 1e-5);

// Softmax
Tensor softmax(const Tensor& logits);

// Attention
Tensor attention_scores(const Tensor& q, const Tensor& k, float scale);
Tensor attention_apply(const Tensor& weights, const Tensor& v);

void attention_scores_neon(float* scores, const float* q, const float* k, 
                           int seq_len, int head_dim, float scale);
void softmax_causal_neon(float* scores, int seq_len);
void attention_apply_neon(float* out, const float* weights, const float* v,
                          int seq_len, int head_dim);

void scaled_dot_product_attention(float* out, const float* q, const float* k, 
                                  const float* v, int seq_len, int head_dim, float scale);

// Flash Attention
Tensor flash_attention_forward(
    const Tensor& q,    // (seq_len, head_dim)
    const Tensor& k,    // (seq_len, head_dim)
    const Tensor& v,    // (seq_len, head_dim)
    float scale,
    bool causal = true
);

struct FlashAttentionGrad {
    Tensor grad_q;  // (seq_len, head_dim)
    Tensor grad_k;  // (seq_len, head_dim)
    Tensor grad_v;  // (seq_len, head_dim)
};

FlashAttentionGrad flash_attention_backward(
    const Tensor& grad_output,  // (seq_len, head_dim)
    const Tensor& q,            // (seq_len, head_dim)
    const Tensor& k,            // (seq_len, head_dim)
    const Tensor& v,            // (seq_len, head_dim)
    float scale,
    bool causal = true
);

// ============ Operator Fusion ============

// Fused QKV projection: single matmul for Q, K, V
// Input x: (seq_len, embed_dim)
// Weight w_qkv: (embed_dim, 3 * embed_dim) - concatenated Q, K, V weights
// Returns tuple of (Q, K, V) tensors
struct QKVTensors {
    Tensor q;  // (seq_len, embed_dim)
    Tensor k;  // (seq_len, embed_dim)
    Tensor v;  // (seq_len, embed_dim)
};

QKVTensors fused_qkv_projection(
    const Tensor& x,              // (seq_len, embed_dim)
    const Tensor& w_qkv,          // (embed_dim, 3 * embed_dim) - concatenated weights
    const std::optional<Tensor>& b_qkv = std::nullopt  // optional bias (3 * embed_dim)
);

// Fused FFN: Linear + GELU/SiLU + Linear
// Avoids intermediate memory allocation
Tensor fused_ffn(
    const Tensor& x,              // (seq_len, embed_dim)
    const Tensor& w1,             // (embed_dim, hidden_dim)
    const Tensor& w2,             // (hidden_dim, embed_dim)
    const std::optional<Tensor>& b1 = std::nullopt,
    const std::optional<Tensor>& b2 = std::nullopt,
    bool use_gelu = true          // true=GELU, false=SiLU
);

// Fused RMSNorm + Residual: output = RMSNorm(x + residual)
Tensor fused_rmsnorm_residual(
    const Tensor& x,
    const Tensor& residual,
    const Tensor& weight,
    float eps = 1e-5f
);

// ============ Quantization ============

struct QuantizedTensor {
    std::vector<int8_t> data;
    std::vector<float> scale;  // Per-channel scales
    int rows;
    int cols;
    
    QuantizedTensor() : rows(0), cols(0) {}
    QuantizedTensor(int rows, int cols, bool per_channel = false) 
        : rows(rows), cols(cols) {
        data.resize(rows * cols);
        scale.resize(per_channel ? rows : 1);
    }
    
    int size() const { return rows * cols; }
};

// Quantize FP32 tensor to INT8
// per_channel: if true, compute scale per row; otherwise per-tensor
QuantizedTensor quantize(const Tensor& input, bool per_channel = false);

// Dequantize INT8 tensor back to FP32
Tensor dequantize(const QuantizedTensor& input);

// INT8 matrix multiplication with dequantization
// C = dequantize(A) @ dequantize(B)
// Input A: (m, k) INT8, B: (k, n) FP32 (for weights we keep FP32)
// For inference: weights are quantized once, activations quantized per-forward
Tensor int8_matmul(const QuantizedTensor& a, const Tensor& b, const std::vector<float>& a_scale);

// INT8 matrix multiplication with NEON (ARM only)
#ifdef __ARM_NEON
Tensor int8_matmul_neon(const QuantizedTensor& a, const Tensor& b, const std::vector<float>& a_scale);
#endif

// Straight-Through Estimator (STE) for quantization-aware training
// Forward: quantize to INT8, Backward: pass through gradient as if no quantization
Tensor ste_quantize(const Tensor& input, float* scale);

// Utilities
void zero(Tensor& t);
void fill(Tensor& t, float value);
Tensor clone(const Tensor& t);
float sum(const Tensor& t);
float mean(const Tensor& t);
float max(const Tensor& t);

// Random initialization
void uniform_(Tensor& t, float min_val, float max_val);
void normal_(Tensor& t, float mean, float std);
void xavier_uniform_(Tensor& t);
void xavier_normal_(Tensor& t);
void kaiming_uniform_(Tensor& t);
void kaiming_normal_(Tensor& t);

} // namespace math
} // namespace microgpt
