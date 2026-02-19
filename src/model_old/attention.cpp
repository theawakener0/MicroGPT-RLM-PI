#include "attention.hpp"
#include "../utils/logger.hpp"
#include "../utils/random.hpp"
#include <cmath>
#include <algorithm>

namespace microgpt {

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, bool use_cache)
    : embed_dim(embed_dim), num_heads(num_heads), use_cache(use_cache) {
    
    head_dim = embed_dim / num_heads;
    scale = 1.0f / std::sqrt((float)head_dim);
    
    wq = Linear(embed_dim, embed_dim, false);
    wk = Linear(embed_dim, embed_dim, false);
    wv = Linear(embed_dim, embed_dim, false);
    wo = Linear(embed_dim, embed_dim, false);
}

Tensor MultiHeadAttention::forward(const Tensor& x, bool is_causal) {
    int seq_len = x.rows();
    int batch = 1;
    
    Tensor q = wq.forward(x);
    Tensor k = wk.forward(x);
    Tensor v = wv.forward(x);
    
    Tensor result(Shape{seq_len, embed_dim}, false);
    
    for (int h = 0; h < num_heads; h++) {
        int hs = h * head_dim;
        
        Tensor q_h(Shape{seq_len, head_dim}, false);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                q_h.data[i * head_dim + j] = q.data[i * embed_dim + hs + j];
            }
        }
        
        std::vector<Tensor> k_h(seq_len);
        std::vector<Tensor> v_h(seq_len);
        
        for (int i = 0; i < seq_len; i++) {
            k_h[i] = Tensor(Shape{head_dim}, false);
            v_h[i] = Tensor(Shape{head_dim}, false);
            for (int j = 0; j < head_dim; j++) {
                k_h[i].data[j] = k.data[i * embed_dim + hs + j];
                v_h[i].data[j] = v.data[i * embed_dim + hs + j];
            }
        }
        
        for (int i = 0; i < seq_len; i++) {
            int start = is_causal ? 0 : 0;
            int end = i + 1;
            
            Tensor attn_logits(Shape{end - start}, false);
            for (int tj = start; tj < end; tj++) {
                float dot = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    dot += q_h.data[i * head_dim + j] * k_h[tj].data[j];
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
            
            Tensor head_out(Shape{head_dim}, false);
            for (int j = 0; j < head_dim; j++) {
                float sum = 0.0f;
                for (int tj = start; tj < end; tj++) {
                    sum += attn_logits.data[tj - start] * v_h[tj].data[j];
                }
                head_out.data[j] = sum;
            }
            
            for (int j = 0; j < head_dim; j++) {
                result.data[i * embed_dim + hs + j] = head_out.data[j];
            }
        }
    }
    
    result = wo.forward(result);
    return result;
}

Tensor MultiHeadAttention::forward_with_cache(const Tensor& x) {
    int seq_len = x.rows();
    
    Tensor q = wq.forward(x);
    Tensor k = wk.forward(x);
    Tensor v = wv.forward(x);
    
    Tensor k_last(Shape{head_dim}, false);
    Tensor v_last(Shape{head_dim}, false);
    for (int j = 0; j < head_dim; j++) {
        k_last.data[j] = k.data[(seq_len - 1) * embed_dim + j];
        v_last.data[j] = v.data[(seq_len - 1) * embed_dim + j];
    }
    
    kv_cache.append(k_last, v_last);
    
    int cache_len = kv_cache.size();
    
    Tensor result(Shape{1, embed_dim}, false);
    
    for (int h = 0; h < num_heads; h++) {
        int hs = h * head_dim;
        
        Tensor q_h(Shape{head_dim}, false);
        for (int j = 0; j < head_dim; j++) {
            q_h.data[j] = q.data[(seq_len - 1) * embed_dim + hs + j];
        }
        
        Tensor attn_logits(Shape{cache_len}, false);
        for (int i = 0; i < cache_len; i++) {
            float dot = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                dot += q_h.data[j] * kv_cache.keys[i].data[hs + j];
            }
            attn_logits.data[i] = dot * scale;
        }
        
        float max_val = attn_logits.data[0];
        for (int j = 1; j < cache_len; j++) {
            max_val = std::max(max_val, attn_logits.data[j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < cache_len; j++) {
            attn_logits.data[j] = std::exp(attn_logits.data[j] - max_val);
            sum_exp += attn_logits.data[j];
        }
        
        for (int j = 0; j < cache_len; j++) {
            attn_logits.data[j] /= sum_exp;
        }
        
        Tensor head_out(Shape{head_dim}, false);
        for (int j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (int i = 0; i < cache_len; i++) {
                sum += attn_logits.data[i] * kv_cache.values[i].data[hs + j];
            }
            head_out.data[j] = sum;
        }
        
        for (int j = 0; j < head_dim; j++) {
            result.data[(seq_len - 1) * embed_dim + hs + j] = head_out.data[j];
        }
    }
    
    result = wo.forward(result);
    return result;
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
    
    Tensor output = fc2.forward(hidden);
    return output;
}

void FeedForward::init_weights(float std) {
    fc1.init_weights(std);
    fc2.init_weights(std);
}

int FeedForward::num_parameters() const {
    return fc1.num_parameters() + fc2.num_parameters();
}

RMSNorm::RMSNorm(int embed_dim, float eps)
    : embed_dim(embed_dim), eps(eps) {
    
    weight = Tensor(Shape{embed_dim}, true);
    init_weights();
}

Tensor RMSNorm::forward(const Tensor& x) {
    int seq_len = x.rows();
    Tensor result(x.shape, false);
    
    for (int i = 0; i < seq_len; i++) {
        float ms = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            float val = x.data[i * embed_dim + j];
            ms += val * val;
        }
        ms = ms / embed_dim + eps;
        float scale = 1.0f / std::sqrt(ms);
        
        for (int j = 0; j < embed_dim; j++) {
            result.data[i * embed_dim + j] = x.data[i * embed_dim + j] * scale * weight.data[j];
        }
    }
    
    return result;
}

void RMSNorm::init_weights() {
    math::fill(weight, 1.0f);
}

int RMSNorm::num_parameters() const {
    return embed_dim;
}

}
