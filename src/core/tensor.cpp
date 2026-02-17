#include "tensor.hpp"
#include "../utils/random.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <set>

namespace microgpt {

// Tensor implementation
Tensor::Tensor() : requires_grad(false) {}

Tensor::Tensor(const Shape& shape) 
    : shape(shape), requires_grad(false) {
    data.resize(size(), 0.0f);
}

Tensor::Tensor(std::initializer_list<int> dims) 
    : requires_grad(false) {
    for (int d : dims) {
        shape.push_back(d);
    }
    data.resize(size(), 0.0f);
}

Tensor::Tensor(const Shape& shape, bool requires_grad) 
    : shape(shape), requires_grad(requires_grad) {
    data.resize(size(), 0.0f);
    if (requires_grad) {
        grad.resize(size(), 0.0f);
    }
}

Tensor::Tensor(const std::vector<float>& data) 
    : data(data), requires_grad(false) {
    shape = {static_cast<int>(data.size())};
}

Tensor::Tensor(const Shape& shape, float fill_value)
    : shape(shape), requires_grad(false) {
    data.resize(size(), fill_value);
}

Tensor::Tensor(const Shape& shape, float fill_value, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {
    data.resize(size(), fill_value);
    if (requires_grad) {
        grad.resize(size(), 0.0f);
    }
}

// Factory methods
Tensor Tensor::ones(const Shape& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::zeros(const Shape& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::randn(const Shape& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::randn(const Shape& shape, float mean, float std) {
    Tensor t(shape, 0.0f);
    for (float& v : t.data) {
        v = Random::normal(mean, std);
    }
    return t;
}

void Tensor::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
}

void Tensor::set_grad(float value) {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), value);
    }
}

Tensor Tensor::view(const Shape& new_shape) const {
    Tensor result;
    result.data = data;  // Share data
    result.shape = new_shape;
    result.requires_grad = requires_grad;
    if (requires_grad) {
        result.grad = grad;  // Share grad memory
    }
    return result;
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " = ";
    }
    std::cout << "[";
    int limit = std::min(10, size());
    for (int i = 0; i < limit; i++) {
        std::cout << data[i];
        if (i < limit - 1) std::cout << ", ";
    }
    if (size() > limit) std::cout << "...";
    std::cout << "] shape=(";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ",";
    }
    std::cout << ")" << std::endl;
}

// Value implementation
std::vector<std::shared_ptr<Value>> Value::tensor_to_values(const Tensor& t) {
    std::vector<std::shared_ptr<Value>> result;
    result.reserve(t.size());
    for (int i = 0; i < t.size(); i++) {
        auto v = std::make_shared<Value>(t.data[i], t.requires_grad);
        v->is_leaf = t.is_leaf;
        result.push_back(v);
    }
    return result;
}

Tensor Value::values_to_tensor(const std::vector<std::shared_ptr<Value>>& v) {
    Tensor result(Shape{static_cast<int>(v.size())}, false);
    for (size_t i = 0; i < v.size(); i++) {
        result.data[i] = v[i]->data;
    }
    return result;
}

void Value::accumulate_grad(Tensor& tensor) const {
    for (size_t i = 0; i < tensor.grad.size() && i < tensor.data.size(); i++) {
        tensor.grad[i] += grad;
    }
}

void Value::build_topo(std::shared_ptr<Value> v, 
                       std::vector<std::shared_ptr<Value>>& topo,
                       std::set<uintptr_t>& visited_set) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(v.get());
    if (visited_set.count(addr)) return;
    visited_set.insert(addr);
    
    for (auto& child : v->children) {
        build_topo(child, topo, visited_set);
    }
    topo.push_back(v);
}

void Value::backward_from(std::shared_ptr<Value> start_node) {
    start_node->grad = 1.0f;
    
    std::vector<std::shared_ptr<Value>> topo;
    std::set<uintptr_t> visited_set;
    
    // Build topological order from start node
    std::vector<std::shared_ptr<Value>> stack;
    stack.push_back(start_node);
    
    while (!stack.empty()) {
        auto v = stack.back();
        stack.pop_back();
        
        uintptr_t addr = reinterpret_cast<uintptr_t>(v.get());
        if (visited_set.count(addr)) continue;
        visited_set.insert(addr);
        
        topo.push_back(v);
        
        for (auto& child : v->children) {
            stack.push_back(child);
        }
    }
    
    // Process in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto v = *it;
        for (size_t i = 0; i < v->children.size(); i++) {
            auto& child = v->children[i];
            float local_grad = v->local_grads[i];
            child->grad += local_grad * v->grad;
        }
    }
}

void Value::backward() {
    backward_from(shared_from_this());
}

// Factory functions
std::shared_ptr<Value> make_value(float data, bool requires_grad) {
    auto v = std::make_shared<Value>(data, requires_grad);
    v->is_leaf = true;
    return v;
}

std::shared_ptr<Value> make_value(std::shared_ptr<Value> v) {
    return v;
}

}
