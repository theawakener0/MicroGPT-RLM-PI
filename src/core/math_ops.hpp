#pragma once
#include "tensor.hpp"
#include <vector>

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
