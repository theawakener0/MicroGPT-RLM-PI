#pragma once
#include <random>
#include <cstdint>

namespace microgpt {

class Random {
public:
    static void seed(uint64_t s) {
        gen.seed(s);
    }
    
    static void seed_time() {
        gen.seed(std::random_device{}());
    }
    
    static float uniform(float min = 0.0f, float max = 1.0f) {
        return std::uniform_real_distribution<float>(min, max)(gen);
    }
    
    static float normal(float mean = 0.0f, float std = 1.0f) {
        return std::normal_distribution<float>(mean, std)(gen);
    }
    
    static int uniform_int(int min, int max) {
        return std::uniform_int_distribution<int>(min, max)(gen);
    }
    
private:
    static std::mt19937_64 gen;
};

}
