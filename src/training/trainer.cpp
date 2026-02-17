#include "trainer.hpp"
#include "../utils/logger.hpp"
#include "../utils/random.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>

namespace microgpt {

AdamOptimizer::AdamOptimizer(int param_count, float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), eps(eps), param_count(param_count), current_step(0) {
    
    m.resize(param_count, 0.0f);
    v.resize(param_count, 0.0f);
}

void AdamOptimizer::update(std::vector<float>& params, const std::vector<float>& grads, int step) {
    float lr_t = learning_rate * std::sqrt(1.0f - std::pow(beta2, step + 1)) / (1.0f - std::pow(beta1, step + 1));
    
    for (size_t i = 0; i < params.size(); i++) {
        float grad = grads[i];
        
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        
        float m_hat = m[i] / (1.0f - std::pow(beta1, step + 1));
        float v_hat = v[i] / (1.0f - std::pow(beta2, step + 1));
        
        params[i] -= lr_t * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void Dataset::load_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        Logger::error("Failed to open file: " + path);
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            documents.push_back(line);
        }
    }
    
    Logger::info("Loaded " + std::to_string(documents.size()) + " documents");
}

void Dataset::shuffle() {
    std::shuffle(documents.begin(), documents.end(), rng);
}

std::string Dataset::get_document(int idx) {
    if (idx >= 0 && idx < (int)documents.size()) {
        return documents[idx];
    }
    return "";
}

Trainer::Trainer(GPT* model, Tokenizer* tokenizer, int batch_size, int max_steps)
    : model(model), tokenizer(tokenizer), batch_size(batch_size), max_steps(max_steps) {
    
    learning_rate = 0.01f;
    gradient_accumulation_steps = 1;
    log_interval = 10;
    
    int param_count = model->num_parameters();
    optimizer = new AdamOptimizer(param_count, learning_rate);
}

void Trainer::train(Dataset& dataset) {
    dataset.shuffle();
    
    Logger::info("Starting training for " + std::to_string(max_steps) + " steps...");
    
    for (int step = 0; step < max_steps; step++) {
        int doc_idx = step % dataset.size();
        std::string doc = dataset.get_document(doc_idx);
        
        train_step(doc);
        
        if (step % log_interval == 0) {
            Logger::info("Step " + std::to_string(step) + "/" + std::to_string(max_steps));
        }
    }
    
    Logger::info("Training complete!");
}

void Trainer::train_step(const std::string& doc) {
    std::vector<int> tokens = tokenizer->encode(doc);
    
    if (tokens.size() < 2) return;
    
    int n = tokens.size() - 1;
    if (n > model->config.max_seq_len) {
        n = model->config.max_seq_len;
    }
    
    std::vector<int> input_tokens(tokens.begin(), tokens.begin() + n);
    std::vector<int> target_tokens(tokens.begin() + 1, tokens.begin() + n + 1);
    
    Tensor logits = model->forward(input_tokens);
    
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        int target = target_tokens[i];
        
        int seq_len = logits.rows();
        int vocab = logits.cols();
        
        float max_val = logits.data[0];
        for (int j = 1; j < vocab; j++) {
            max_val = std::max(max_val, logits.data[i * vocab + j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            sum_exp += std::exp(logits.data[i * vocab + j] - max_val);
        }
        
        float prob = std::exp(logits.data[i * vocab + target] - max_val) / sum_exp;
        prob = std::max(1e-7f, std::min(1.0f - 1e-7f, prob));
        
        loss += -std::log(prob);
    }
    
    loss /= n;
}

float Trainer::compute_loss(const std::vector<int>& tokens) {
    return 0.0f;
}

}
