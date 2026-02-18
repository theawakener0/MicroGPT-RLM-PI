#pragma once
#include <vector>
#include <memory>
#include <string>

namespace microgpt {

class Tensor;

class Module {
public:
    virtual ~Module() = default;
    
    virtual Tensor forward(const Tensor& x) = 0;
    virtual void backward(const Tensor& grad_output) = 0;
    virtual void zero_grad() = 0;
    virtual std::vector<Tensor*> parameters() = 0;
    
    int num_params() const {
        int count = 0;
        for (auto* p : parameters()) {
            count += p->size();
        }
        return count;
    }
    
    void print_parameters() const {
        for (auto* p : parameters()) {
            p->print();
        }
    }
};

}
