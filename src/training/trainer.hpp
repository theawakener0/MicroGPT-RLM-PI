#pragma once
#include "../training/trainable_model.hpp"
#include "../training/tokenizer.hpp"
#include <string>
#include <vector>
#include <random>

namespace microgpt {

using GPT = TrainableGPT;

class AdamOptimizer {
public:
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    int param_count;
    
    std::vector<float> m;
    std::vector<float> v;
    int current_step;
    
    AdamOptimizer() = default;
    AdamOptimizer(int param_count, float lr = 0.01f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
    
    void update(std::vector<float>& params, const std::vector<float>& grads, int step);
};

class Dataset {
public:
    std::vector<std::string> documents;
    std::mt19937 rng;
    
    Dataset() {
        rng.seed(42);
    }
    
    void load_file(const std::string& path);
    void shuffle();
    std::string get_document(int idx);
    size_t size() const { return documents.size(); }
};

class Trainer {
public:
    GPT* model;
    Tokenizer* tokenizer;
    AdamOptimizer* optimizer;
    
    int batch_size;
    int gradient_accumulation_steps;
    int max_steps;
    float learning_rate;
    int log_interval;
    
    Trainer() = default;
    Trainer(GPT* model, Tokenizer* tokenizer, int batch_size = 1, int max_steps = 1000);
    
    void train(Dataset& dataset);
    void train_step(const std::string& doc);
    float compute_loss(const std::vector<int>& tokens);
};

}
