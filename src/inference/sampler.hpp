#pragma once
#include "../core/tensor.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace microgpt {

class Sampler {
public:
    float temperature;
    int top_k;
    float top_p;
    
    std::mt19937 rng;
    
    Sampler() : temperature(1.0f), top_k(0), top_p(0.0f) {
        std::random_device rd;
        rng.seed(rd());
    }
    
    Sampler(float temperature, int top_k = 0, float top_p = 0.0f) 
        : temperature(temperature), top_k(top_k), top_p(top_p) {
        std::random_device rd;
        rng.seed(rd());
    }
    
    int sample(const Tensor& logits) {
        int seq_len = logits.rows();
        int vocab = logits.cols();
        int idx = seq_len - 1;
        
        std::vector<float> probs(vocab);
        
        float max_val = logits.data[idx * vocab];
        for (int i = 1; i < vocab; i++) {
            max_val = std::max(max_val, logits.data[idx * vocab + i]);
        }
        
        for (int i = 0; i < vocab; i++) {
            float val = (logits.data[idx * vocab + i] - max_val) / temperature;
            probs[i] = std::exp(val);
        }
        
        if (top_k > 0) {
            std::vector<std::pair<float, int>> indexed_probs;
            for (int i = 0; i < vocab; i++) {
                indexed_probs.push_back({probs[i], i});
            }
            std::partial_sort(indexed_probs.begin(), indexed_probs.begin() + top_k,
                            indexed_probs.end(), std::greater<std::pair<float, int>>());
            
            float top_k_sum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                probs[indexed_probs[i].second] = indexed_probs[i].first;
                top_k_sum += probs[indexed_probs[i].second];
            }
            for (int i = 0; i < vocab; i++) {
                bool in_top_k = false;
                for (int j = 0; j < top_k; j++) {
                    if (indexed_probs[j].second == i) {
                        in_top_k = true;
                        break;
                    }
                }
                if (!in_top_k) {
                    probs[i] = 0.0f;
                }
            }
        }
        
        if (top_p > 0.0f) {
            std::vector<std::pair<float, int>> indexed_probs;
            for (int i = 0; i < vocab; i++) {
                indexed_probs.push_back({probs[i], i});
            }
            std::sort(indexed_probs.begin(), indexed_probs.end(), std::greater<std::pair<float, int>>());
            
            float cumulative = 0.0f;
            int n = vocab;
            for (int i = 0; i < vocab; i++) {
                cumulative += indexed_probs[i].first;
                if (cumulative >= top_p) {
                    n = i + 1;
                    break;
                }
            }
            
            for (int i = 0; i < vocab; i++) {
                bool keep = false;
                for (int j = 0; j < n; j++) {
                    if (indexed_probs[j].second == i) {
                        keep = true;
                        break;
                    }
                }
                if (!keep) {
                    probs[i] = 0.0f;
                }
            }
        }
        
        float sum = 0.0f;
        for (float p : probs) {
            sum += p;
        }
        
        if (sum <= 0.0f) {
            return 0;
        }
        
        for (int i = 0; i < vocab; i++) {
            probs[i] /= sum;
        }
        
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        
        float cumulative = 0.0f;
        for (int i = 0; i < vocab; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return i;
            }
        }
        
        return vocab - 1;
    }
    
    int greedy_sample(const Tensor& logits) {
        int seq_len = logits.rows();
        int vocab = logits.cols();
        int idx = seq_len - 1;
        
        float max_val = logits.data[idx * vocab];
        int max_id = 0;
        
        for (int i = 1; i < vocab; i++) {
            float val = logits.data[idx * vocab + i];
            if (val > max_val) {
                max_val = val;
                max_id = i;
            }
        }
        
        return max_id;
    }
    
    int sample_with_penalty(const Tensor& logits, const std::vector<int>& prev_tokens, float penalty = 1.0f) {
        int seq_len = logits.rows();
        int vocab = logits.cols();
        int idx = seq_len - 1;
        
        std::vector<float> probs(vocab);
        
        float max_val = logits.data[idx * vocab];
        for (int i = 1; i < vocab; i++) {
            max_val = std::max(max_val, logits.data[idx * vocab + i]);
        }
        
        for (int i = 0; i < vocab; i++) {
            float val = (logits.data[idx * vocab + i] - max_val) / temperature;
            probs[i] = std::exp(val);
        }
        
        if (penalty != 1.0f && !prev_tokens.empty()) {
            for (int token : prev_tokens) {
                if (token < vocab) {
                    probs[token] = std::max(0.0f, probs[token] / penalty);
                }
            }
        }
        
        float sum = 0.0f;
        for (float p : probs) {
            sum += p;
        }
        
        if (sum <= 0.0f) {
            return 0;
        }
        
        for (int i = 0; i < vocab; i++) {
            probs[i] /= sum;
        }
        
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        
        float cumulative = 0.0f;
        for (int i = 0; i < vocab; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return i;
            }
        }
        
        return vocab - 1;
    }
};

}
