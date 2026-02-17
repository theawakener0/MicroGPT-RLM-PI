#pragma once
#include <vector>
#include <memory>
#include <cstring>
#include <string>
#include <set>

namespace microgpt {

// Forward declaration
class Value;

// Tensor shape
using Shape = std::vector<int>;

// Simple tensor class with optional gradient tracking
class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    Shape shape;
    bool requires_grad = false;
    bool is_leaf = true;
    
    // Computation graph connections
    std::vector<std::shared_ptr<Value>> grad_connected;
    
    // Constructors - use explicit to avoid ambiguity
    Tensor();
    explicit Tensor(const Shape& shape);
    explicit Tensor(const std::vector<float>& data);
    
    // Initializer list constructor (preferred way)
    Tensor(std::initializer_list<int> dims);
    
    // With requires_grad
    Tensor(const Shape& shape, bool requires_grad);
    Tensor(const std::vector<float>& data, bool requires_grad);
    
    // With fill value
    Tensor(const Shape& shape, float fill_value);
    Tensor(const Shape& shape, float fill_value, bool requires_grad);
    
    // Factory methods
    static Tensor ones(const Shape& shape);
    static Tensor zeros(const Shape& shape);
    static Tensor randn(const Shape& shape);
    static Tensor randn(const Shape& shape, float mean, float std);
    
    int size() const { 
        int s = 1;
        for (int dim : shape) s *= dim;
        return s;
    }
    
    int rows() const { return shape.empty() ? 1 : shape[0]; }
    int cols() const { return shape.size() > 1 ? shape[1] : 1; }
    
    void zero_grad();
    void set_grad(float value);
    
    // Reshape view (no data copy)
    Tensor view(const Shape& new_shape) const;
    
    // Element access
    float& operator[](int idx) { return data[idx]; }
    float operator[](int idx) const { return data[idx]; }
    
    // Print for debugging
    void print(const std::string& name = "") const;
};

// Value class for automatic differentiation (like micrograd)
class Value : public std::enable_shared_from_this<Value> {
public:
    float data;
    float grad;
    
    // Graph connections
    std::vector<std::shared_ptr<Value>> children;
    std::vector<float> local_grads;
    std::string op;  // Operation name for debugging
    
    // For optimization
    bool is_leaf = false;
    bool requires_grad = true;
    
    Value() : data(0), grad(0) {}
    Value(float d) : data(d), grad(0) {}
    Value(float d, bool requires_grad) : data(d), grad(0), requires_grad(requires_grad) {}
    
    // Tensor to Value conversion
    static std::vector<std::shared_ptr<Value>> tensor_to_values(const Tensor& t);
    static Tensor values_to_tensor(const std::vector<std::shared_ptr<Value>>& v);
    
    // Copy values from Value gradient to Tensor gradient
    void accumulate_grad(Tensor& tensor) const;
    
    // Operations
    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator+(float scalar);
    std::shared_ptr<Value> operator*(float scalar);
    std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator-(float scalar);
    std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator/(float scalar);
    
    std::shared_ptr<Value> pow(float exp);
    std::shared_ptr<Value> log();
    std::shared_ptr<Value> exp();
    std::shared_ptr<Value> relu();
    std::shared_ptr<Value> sigmoid();
    
    // Negation
    std::shared_ptr<Value> operator-();
    
    // Backward pass
    void backward();
    static void backward_from(std::shared_ptr<Value> start_node);
    
    // Build topological order for backward (static helper)
    static void build_topo(std::shared_ptr<Value> v, 
                           std::vector<std::shared_ptr<Value>>& topo,
                           std::set<uintptr_t>& visited_set);
};

// Helper functions
std::shared_ptr<Value> make_value(float data, bool requires_grad = true);
std::shared_ptr<Value> make_value(std::shared_ptr<Value> v);

}
