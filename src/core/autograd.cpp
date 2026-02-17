#include "autograd.hpp"
#include <algorithm>
#include <numeric>

namespace microgpt {
namespace autograd {

// Helper to create operation node
std::shared_ptr<Value> make_op(float data, std::string op_name,
                                std::vector<std::shared_ptr<Value>> children,
                                std::vector<float> local_grads) {
    auto result = std::make_shared<Value>(data, true);
    result->op = op_name;
    result->children = std::move(children);
    result->local_grads = std::move(local_grads);
    result->is_leaf = false;
    return result;
}

// ============ Binary Operations ============

std::shared_ptr<Value> add(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    float result_data = a->data + b->data;
    return make_op(result_data, "+", {a, b}, {1.0f, 1.0f});
}

std::shared_ptr<Value> multiply(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    float result_data = a->data * b->data;
    return make_op(result_data, "*", {a, b}, {b->data, a->data});
}

std::shared_ptr<Value> subtract(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    float result_data = a->data - b->data;
    return make_op(result_data, "-", {a, b}, {1.0f, -1.0f});
}

std::shared_ptr<Value> divide(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    float result_data = a->data / b->data;
    float grad_a = 1.0f / b->data;
    float grad_b = -a->data / (b->data * b->data);
    return make_op(result_data, "/", {a, b}, {grad_a, grad_b});
}

// ============ Scalar Operations ============

std::shared_ptr<Value> add_scalar(std::shared_ptr<Value> a, float scalar) {
    float result_data = a->data + scalar;
    return make_op(result_data, "+s", {a}, {1.0f});
}

std::shared_ptr<Value> multiply_scalar(std::shared_ptr<Value> a, float scalar) {
    float result_data = a->data * scalar;
    return make_op(result_data, "*s", {a}, {scalar});
}

// ============ Unary Operations ============

std::shared_ptr<Value> relu(std::shared_ptr<Value> a) {
    float result_data = std::max(0.0f, a->data);
    float local_grad = a->data > 0 ? 1.0f : 0.0f;
    return make_op(result_data, "relu", {a}, {local_grad});
}

std::shared_ptr<Value> sigmoid(std::shared_ptr<Value> a) {
    float sigmoid_val = 1.0f / (1.0f + std::exp(-a->data));
    float local_grad = sigmoid_val * (1.0f - sigmoid_val);
    return make_op(sigmoid_val, "sigmoid", {a}, {local_grad});
}

std::shared_ptr<Value> tanh(std::shared_ptr<Value> a) {
    float result_data = std::tanh(a->data);
    float local_grad = 1.0f - result_data * result_data;
    return make_op(result_data, "tanh", {a}, {local_grad});
}

std::shared_ptr<Value> log(std::shared_ptr<Value> a) {
    float result_data = std::log(std::max(1e-7f, a->data));
    float local_grad = 1.0f / a->data;
    return make_op(result_data, "log", {a}, {local_grad});
}

std::shared_ptr<Value> exp(std::shared_ptr<Value> a) {
    float result_data = std::exp(a->data);
    float local_grad = result_data;
    return make_op(result_data, "exp", {a}, {local_grad});
}

std::shared_ptr<Value> pow(std::shared_ptr<Value> a, float exponent) {
    float result_data = std::pow(a->data, exponent);
    float local_grad = exponent * std::pow(a->data, exponent - 1.0f);
    return make_op(result_data, "pow", {a}, {local_grad});
}

std::shared_ptr<Value> sqrt(std::shared_ptr<Value> a) {
    float result_data = std::sqrt(std::max(0.0f, a->data));
    float local_grad = 0.5f / result_data;
    return make_op(result_data, "sqrt", {a}, {local_grad});
}

std::shared_ptr<Value> sum(std::shared_ptr<Value> a) {
    // Sum assumes a is already a scalar (single value)
    // This is used for aggregating losses
    return a; // Pass through for scalar
}

// ============ Matrix Operations ============

std::shared_ptr<Value> dot(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    float result_data = a->data * b->data;  // For scalars
    return make_op(result_data, "dot", {a, b}, {b->data, a->data});
}

std::shared_ptr<Value> matmul(std::shared_ptr<Value> a, std::shared_ptr<Value> b) {
    // Matrix multiply for scalars is just multiply
    return multiply(a, b);
}

// ============ Activation Functions ============

std::shared_ptr<Value> softmax(std::shared_ptr<Value> logits) {
    // Softmax on a single logit is just sigmoid
    return sigmoid(logits);
}

// ============ Loss Functions ============

std::shared_ptr<Value> cross_entropy(std::shared_ptr<Value> logits, int target_idx) {
    // For single logit: -log(softmax(logits)[target])
    // Since we use sigmoid for single-value softmax
    float prob = 1.0f / (1.0f + std::exp(-logits->data));
    prob = std::max(1e-7f, std::min(1e-7f, prob)); // Clamp
    
    // For target=1: -log(prob), for target=0: -log(1-prob)
    float loss = (target_idx == 1) ? -std::log(prob) : -std::log(1.0f - prob);
    
    return make_op(loss, "cross_entropy", {logits}, {1.0f});
}

std::shared_ptr<Value> mse_loss(std::shared_ptr<Value> pred, std::shared_ptr<Value> target) {
    float diff = pred->data - target->data;
    float loss = diff * diff;
    float grad_pred = 2.0f * diff;
    float grad_target = -2.0f * diff;
    return make_op(loss, "mse", {pred, target}, {grad_pred, grad_target});
}

} // namespace autograd
} // namespace microgpt
