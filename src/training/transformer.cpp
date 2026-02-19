#include "transformer.hpp"
#include "../utils/logger.hpp"
#include <cmath>
#include <algorithm>

namespace microgpt {

RMSNorm::RMSNorm(int embed_dim, float eps) 
    : embed_dim(embed_dim), eps(eps) {
    
    weight = Tensor(Shape{embed_dim}, true);
    weight_grad = Tensor(Shape{embed_dim}, false);
    init_weights();
}

Tensor RMSNorm::forward(const Tensor& x) {
    int seq_len = x.rows();
    input_cache = x;
    
    Tensor result(x.shape, false);
    
    for (int i = 0; i < seq_len; i++) {
        float ms = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            float val = x.data[i * embed_dim + j];
            ms += val * val;
        }
        ms = ms / embed_dim + eps;
        float scale = 1.0f / std::sqrt(ms);
        float inv_scale = 1.0f / (std::sqrt(ms) * embed_dim);
        
        for (int j = 0; j < embed_dim; j++) {
            result.data[i * embed_dim + j] = x.data[i * embed_dim + j] * scale * weight.data[j];
        }
    }
    
    output_cache = result;
    return result;
}

Tensor RMSNorm::backward(const Tensor& grad_output) {
    int seq_len = input_cache.rows();
    
    Tensor grad_input(input_cache.shape, false);
    
    for (int i = 0; i < seq_len; i++) {
        float ms = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            float val = input_cache.data[i * embed_dim + j];
            ms += val * val;
        }
        ms = ms / embed_dim + eps;
        float scale = 1.0f / std::sqrt(ms);
        
        float w_dot_g = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            w_dot_g += weight.data[j] * grad_output.data[i * embed_dim + j];
        }
        
        for (int j = 0; j < embed_dim; j++) {
            float x = input_cache.data[i * embed_dim + j];
            float g = grad_output.data[i * embed_dim + j];
            float w = weight.data[j];
            
            float grad_norm = -x * w * w_dot_g / (ms * embed_dim);
            float grad_weight = g * scale * x;
            weight_grad.data[j] += grad_weight;
            
            grad_input.data[i * embed_dim + j] = scale * w * g + grad_norm;
        }
    }
    
    return grad_input;
}

void RMSNorm::zero_grad() {
    math::fill(weight_grad, 0.0f);
}

std::vector<Tensor*> RMSNorm::parameters() {
    std::vector<Tensor*> params;
    params.push_back(&weight);
    return params;
}

void RMSNorm::init_weights() {
    math::fill(weight, 1.0f);
}

int RMSNorm::num_parameters() const {
    return embed_dim;
}

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
    
    head_dim = embed_dim / num_heads;
    scale = 1.0f / std::sqrt((float)head_dim);
    
    wq = Linear(embed_dim, embed_dim, false);
    wk = Linear(embed_dim, embed_dim, false);
    wv = Linear(embed_dim, embed_dim, false);
    wo = Linear(embed_dim, embed_dim, false);
}

Tensor MultiHeadAttention::forward(const Tensor& x, bool causal) {
    int seq_len = x.rows();
    
    Tensor q = wq.forward(x);
    Tensor k = wk.forward(x);
    Tensor v = wv.forward(x);
    
    q_cache = q;
    k_cache = k;
    v_cache = v;
    
    attn_cache = Tensor(Shape{seq_len, seq_len * num_heads}, false);
    
    Tensor result(Shape{seq_len, embed_dim}, false);
    
    for (int h = 0; h < num_heads; h++) {
        int hs = h * head_dim;
        
        float* scores_ptr = attn_cache.data.data() + h * seq_len * seq_len;
        math::attention_scores_neon(scores_ptr, 
                                    q.data.data() + hs, 
                                    k.data.data() + hs, 
                                    seq_len, head_dim, scale);
        
        for (int i = 0; i < seq_len; i++) {
            float max_val = scores_ptr[i * seq_len];
            for (int j = 1; j <= i; j++) {
                max_val = std::max(max_val, scores_ptr[i * seq_len + j]);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores_ptr[i * seq_len + j] = std::exp(scores_ptr[i * seq_len + j] - max_val);
                sum_exp += scores_ptr[i * seq_len + j];
            }
            
            for (int j = 0; j <= i; j++) {
                scores_ptr[i * seq_len + j] /= sum_exp;
            }
            
            for (int j = i + 1; j < seq_len; j++) {
                scores_ptr[i * seq_len + j] = 0.0f;
            }
        }
        
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j <= i; j++) {
                    sum += scores_ptr[i * seq_len + j] * v.data[j * embed_dim + hs + d];
                }
                result.data[i * embed_dim + hs + d] = sum;
            }
        }
    }
    
    result = wo.forward(result);
    output_cache = result;
    return result;
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output) {
    wq.zero_grad();
    wk.zero_grad();
    wv.zero_grad();
    wo.zero_grad();
    
    int seq_len = q_cache.rows();
    
    Tensor grad_wo = wo.backward(grad_output);
    
    Tensor grad_result(Shape{seq_len, embed_dim}, false);
    for (int i = 0; i < seq_len * embed_dim; i++) {
        grad_result.data[i] = grad_wo.data[i];
    }
    
    std::vector<Tensor> grad_qkv;
    for (int h = 0; h < num_heads; h++) {
        int hs = h * head_dim;
        
        for (int i = 0; i < seq_len; i++) {
            Tensor grad_head(Shape{head_dim}, false);
            for (int j = 0; j < head_dim; j++) {
                grad_head.data[j] = grad_result.data[i * embed_dim + hs + j];
            }
            
            int start = 0;
            int end = i + 1;
            
            Tensor attn_logits(Shape{end - start}, false);
            for (int tj = start; tj < end; tj++) {
                float dot = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    dot += q_cache.data[i * embed_dim + hs + j] * k_cache.data[tj * embed_dim + hs + j];
                }
                attn_logits.data[tj - start] = dot * scale;
            }
            
            float max_val = attn_logits.data[0];
            for (int j = 1; j < attn_logits.size(); j++) {
                max_val = std::max(max_val, attn_logits.data[j]);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < attn_logits.size(); j++) {
                attn_logits.data[j] = std::exp(attn_logits.data[j] - max_val);
                sum_exp += attn_logits.data[j];
            }
            
            for (int j = 0; j < attn_logits.size(); j++) {
                attn_logits.data[j] /= sum_exp;
            }
            
            Tensor grad_v(Shape{head_dim}, false);
            for (int j = 0; j < head_dim; j++) {
                float sum = 0.0f;
                for (int tj = start; tj < end; tj++) {
                    sum += attn_logits.data[tj - start] * v_cache.data[tj * embed_dim + hs + j];
                }
                grad_v.data[j] = grad_head.data[j];
            }
            
            Tensor grad_attn(Shape{end - start}, false);
            for (int tj = start; tj < end; tj++) {
                float sum = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    sum += grad_head.data[j] * v_cache.data[tj * embed_dim + hs + j];
                }
                float prob = attn_logits.data[tj - start];
                grad_attn.data[tj - start] = prob * sum - prob * grad_head.data[0];
            }
            
            Tensor grad_q_i(Shape{head_dim}, false);
            Tensor grad_k_i(Shape{head_dim}, false);
            
            for (int j = 0; j < head_dim; j++) {
                float grad_q_sum = 0.0f;
                for (int tj = start; tj < end; tj++) {
                    grad_q_sum += grad_attn.data[tj - start] * k_cache.data[tj * embed_dim + hs + j];
                }
                grad_q_i.data[j] = grad_q_sum * scale;
                
                float grad_k_sum = 0.0f;
                for (int qi = start; qi < i + 1; qi++) {
                    grad_k_sum += grad_attn.data[qi - start] * q_cache.data[i * embed_dim + hs + j];
                }
                grad_k_i.data[j] = grad_k_sum * scale;
            }
        }
    }
    
    Tensor grad_q = wq.backward(grad_result);
    Tensor grad_k = wk.backward(grad_result);
    Tensor grad_v = wv.backward(grad_result);
    
    Tensor grad_input(grad_q.shape, false);
    for (int i = 0; i < grad_q.size(); i++) {
        grad_input.data[i] = grad_q.data[i] + grad_k.data[i] + grad_v.data[i];
    }
    
    return grad_input;
}

void MultiHeadAttention::zero_grad() {
    wq.zero_grad();
    wk.zero_grad();
    wv.zero_grad();
    wo.zero_grad();
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : wq.parameters()) params.push_back(p);
    for (auto* p : wk.parameters()) params.push_back(p);
    for (auto* p : wv.parameters()) params.push_back(p);
    for (auto* p : wo.parameters()) params.push_back(p);
    return params;
}

void MultiHeadAttention::init_weights(float std) {
    wq.init_weights(std);
    wk.init_weights(std);
    wv.init_weights(std);
    wo.init_weights(std);
}

int MultiHeadAttention::num_parameters() const {
    return wq.num_parameters() + wk.num_parameters() + wv.num_parameters() + wo.num_parameters();
}

FeedForward::FeedForward(int embed_dim, int hidden_dim)
    : embed_dim(embed_dim), hidden_dim(hidden_dim) {
    
    fc1 = Linear(embed_dim, hidden_dim, false);
    fc2 = Linear(hidden_dim, embed_dim, false);
}

Tensor FeedForward::forward(const Tensor& x) {
    Tensor hidden = fc1.forward(x);
    
    for (int i = 0; i < hidden.size(); i++) {
        hidden.data[i] = std::max(0.0f, hidden.data[i]);
    }
    
    hidden_cache = hidden;
    Tensor output = fc2.forward(hidden);
    output_cache = output;
    return output;
}

Tensor FeedForward::backward(const Tensor& grad_output) {
    fc1.zero_grad();
    fc2.zero_grad();
    
    Tensor grad_fc2 = fc2.backward(grad_output);
    
    Tensor grad_hidden(hidden_cache.shape, false);
    int hidden_size = hidden_cache.size();
    int batch_size = hidden_cache.rows();
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            int idx = i * hidden_dim + j;
            if (hidden_cache.data[idx] > 0) {
                grad_hidden.data[idx] = grad_fc2.data[i * embed_dim + j];
            } else {
                grad_hidden.data[idx] = 0.0f;
            }
        }
    }
    
    Tensor grad_input = fc1.backward(grad_hidden);
    return grad_input;
}

void FeedForward::zero_grad() {
    fc1.zero_grad();
    fc2.zero_grad();
}

std::vector<Tensor*> FeedForward::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : fc1.parameters()) params.push_back(p);
    for (auto* p : fc2.parameters()) params.push_back(p);
    return params;
}

void FeedForward::init_weights(float std) {
    fc1.init_weights(std);
    fc2.init_weights(std);
}

int FeedForward::num_parameters() const {
    return fc1.num_parameters() + fc2.num_parameters();
}

TransformerBlock::TransformerBlock(int layer_idx, int embed_dim, int num_heads, int hidden_dim)
    : layer_idx(layer_idx), embed_dim(embed_dim) {
    
    ln_1 = RMSNorm(embed_dim);
    ln_2 = RMSNorm(embed_dim);
    attn = MultiHeadAttention(embed_dim, num_heads);
    mlp = FeedForward(embed_dim, hidden_dim);
}

Tensor TransformerBlock::forward(const Tensor& x) {
    Tensor x_residual = x;
    Tensor normalized = ln_1.forward(x);
    
    Tensor attn_out = attn.forward(normalized);
    
    Tensor result1(x.shape, false);
    for (int i = 0; i < result1.size(); i++) {
        result1.data[i] = x_residual.data[i] + attn_out.data[i];
    }
    
    residual_1_cache = result1;
    
    Tensor x2 = result1;
    Tensor normalized2 = ln_2.forward(x2);
    Tensor mlp_out = mlp.forward(normalized2);
    
    for (int i = 0; i < result1.size(); i++) {
        result1.data[i] = x2.data[i] + mlp_out.data[i];
    }
    
    residual_2_cache = result1;
    
    return result1;
}

Tensor TransformerBlock::backward(const Tensor& grad_output) {
    ln_1.zero_grad();
    ln_2.zero_grad();
    attn.zero_grad();
    mlp.zero_grad();
    
    Tensor grad_residual2(grad_output.shape, false);
    for (int i = 0; i < grad_residual2.size(); i++) {
        grad_residual2.data[i] = grad_output.data[i];
    }
    
    Tensor grad_mlp_in = mlp.backward(grad_residual2);
    Tensor grad_ln2_in = ln_2.backward(grad_mlp_in);
    Tensor grad_attn_in = attn.backward(grad_ln2_in);
    Tensor grad_ln1_in = ln_1.backward(grad_attn_in);
    
    Tensor grad_input(grad_output.shape, false);
    for (int i = 0; i < grad_input.size(); i++) {
        grad_input.data[i] = grad_ln1_in.data[i] + grad_residual2.data[i];
    }
    
    return grad_input;
}

void TransformerBlock::zero_grad() {
    ln_1.zero_grad();
    ln_2.zero_grad();
    attn.zero_grad();
    mlp.zero_grad();
}

std::vector<Tensor*> TransformerBlock::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : ln_1.parameters()) params.push_back(p);
    for (auto* p : attn.parameters()) params.push_back(p);
    for (auto* p : mlp.parameters()) params.push_back(p);
    for (auto* p : ln_2.parameters()) params.push_back(p);
    return params;
}

void TransformerBlock::init_weights(float std) {
    ln_1.init_weights();
    ln_2.init_weights();
    attn.init_weights(std);
    mlp.init_weights(std);
}

int TransformerBlock::num_parameters() const {
    return ln_1.num_parameters() + attn.num_parameters() + mlp.num_parameters() + ln_2.num_parameters();
}

}
