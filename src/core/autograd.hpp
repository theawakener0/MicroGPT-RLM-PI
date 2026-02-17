#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <cmath>

namespace microgpt {
namespace autograd {

// Binary operations
std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
std::shared_ptr<Value> multiply(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
std::shared_ptr<Value> subtract(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
std::shared_ptr<Value> divide(std::shared_ptr<Value> a, std::shared_ptr<Value> b);

// Scalar operations
std::shared_ptr<Value> add_scalar(std::shared_ptr<Value> a, float scalar);
std::shared_ptr<Value> multiply_scalar(std::shared_ptr<Value> a, float scalar);

// Unary operations  
std::shared_ptr<Value> relu(std::shared_ptr<Value> a);
std::shared_ptr<Value> sigmoid(std::shared_ptr<Value> a);
std::shared_ptr<Value> tanh(std::shared_ptr<Value> a);
std::shared_ptr<Value> log(std::shared_ptr<Value> a);
std::shared_ptr<Value> exp(std::shared_ptr<Value> a);
std::shared_ptr<Value> pow(std::shared_ptr<Value> a, float exponent);
std::shared_ptr<Value> sqrt(std::shared_ptr<Value> a);
std::shared_ptr<Value> sum(std::shared_ptr<Value> a);

// Matrix operations (for attention, linear layers)
std::shared_ptr<Value> dot(std::shared_ptr<Value> a, std::shared_ptr<Value> b);
std::shared_ptr<Value> matmul(std::shared_ptr<Value> a, std::shared_ptr<Value> b);

// Activation functions
std::shared_ptr<Value> softmax(std::shared_ptr<Value> logits);

// Loss functions
std::shared_ptr<Value> cross_entropy(std::shared_ptr<Value> logits, int target_idx);
std::shared_ptr<Value> mse_loss(std::shared_ptr<Value> pred, std::shared_ptr<Value> target);

} // namespace autograd
} // namespace microgpt
