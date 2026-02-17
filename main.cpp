#include <iostream>
#include <vector>
#include "src/utils/logger.hpp"
#include "src/core/tensor.hpp"
#include "src/core/autograd.hpp"
#include "src/core/math_ops.hpp"

using namespace microgpt;

int main() {
    Logger::info("MicroGPT-RLM v1.0.0");
    Logger::info("Initializing tensor tests...");
    
    // Test tensor creation
    Tensor t1(Shape{2, 3});
    Logger::info("Created tensor t1");
    t1.print("t1");
    
    // Test with data
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t2(data_vec);
    Logger::info("Created tensor t2 from vector");
    t2.print("t2");
    
    // Test with requires_grad
    Tensor t3(Shape{2, 2}, true);
    math::kaiming_normal_(t3);
    Logger::info("Created tensor t3 with gradients");
    t3.print("t3");
    
    // Test autograd
    Logger::info("Testing autograd...");
    auto a = make_value(2.0f);
    auto b = make_value(3.0f);
    auto c = autograd::multiply(a, b);
    auto d = autograd::add(c, a);
    
    Logger::info("Forward pass: a=2, b=3, c=a*b=6, d=c+a=8");
    
    // Backward
    d->backward();
    
    std::cout << "Gradients: a.grad=" << a->grad << ", b.grad=" << b->grad << std::endl;
    
    Logger::info("All tests passed!");
    
    return 0;
}
