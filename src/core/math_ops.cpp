#include "math_ops.hpp"
#include "../utils/random.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

// Detect ARM NEON
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

namespace microgpt {
namespace math {

// ============ Element-wise Operations ============

void add_inplace(Tensor& a, const Tensor& b) {
    #if USE_NEON
    int n = a.size();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a.data[i]);
        float32x4_t vb = vld1q_f32(&b.data[i]);
        va = vaddq_f32(va, vb);
        vst1q_f32(&a.data[i], va);
    }
    for (; i < n; i++) {
        a.data[i] += b.data[i];
    }
    #else
    for (int i = 0; i < a.size(); i++) {
        a.data[i] += b.data[i];
    }
    #endif
}

void add_scalar_inplace(Tensor& a, float scalar) {
    #if USE_NEON
    int n = a.size();
    int i = 0;
    float32x4_t vs = vdupq_n_f32(scalar);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a.data[i]);
        va = vaddq_f32(va, vs);
        vst1q_f32(&a.data[i], va);
    }
    for (; i < n; i++) {
        a.data[i] += scalar;
    }
    #else
    for (int i = 0; i < a.size(); i++) {
        a.data[i] += scalar;
    }
    #endif
}

void multiply_inplace(Tensor& a, const Tensor& b) {
    #if USE_NEON
    int n = a.size();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a.data[i]);
        float32x4_t vb = vld1q_f32(&b.data[i]);
        va = vmulq_f32(va, vb);
        vst1q_f32(&a.data[i], va);
    }
    for (; i < n; i++) {
        a.data[i] *= b.data[i];
    }
    #else
    for (int i = 0; i < a.size(); i++) {
        a.data[i] *= b.data[i];
    }
    #endif
}

void multiply_scalar_inplace(Tensor& a, float scalar) {
    #if USE_NEON
    int n = a.size();
    int i = 0;
    float32x4_t vs = vdupq_n_f32(scalar);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a.data[i]);
        va = vmulq_f32(va, vs);
        vst1q_f32(&a.data[i], va);
    }
    for (; i < n; i++) {
        a.data[i] *= scalar;
    }
    #else
    for (int i = 0; i < a.size(); i++) {
        a.data[i] *= scalar;
    }
    #endif
}

void relu_inplace(Tensor& a) {
    #if USE_NEON
    int n = a.size();
    int i = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a.data[i]);
        va = vmaxq_f32(va, vzero);
        vst1q_f32(&a.data[i], va);
    }
    for (; i < n; i++) {
        a.data[i] = std::max(0.0f, a.data[i]);
    }
    #else
    for (int i = 0; i < a.size(); i++) {
        a.data[i] = std::max(0.0f, a.data[i]);
    }
    #endif
}

void sigmoid_inplace(Tensor& a) {
    for (int i = 0; i < a.size(); i++) {
        a.data[i] = 1.0f / (1.0f + std::exp(-a.data[i]));
    }
}

void tanh_inplace(Tensor& a) {
    for (int i = 0; i < a.size(); i++) {
        a.data[i] = std::tanh(a.data[i]);
    }
}

// ============ Matrix Operations ============

Tensor matmul(const Tensor& a, const Tensor& b) {
    int m = a.rows();
    int n = a.cols();
    int k = b.cols();
    
    Tensor result(Shape{m, k}, false);
    
    #if USE_NEON
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            int l = 0;
            
            float32x4_t vsum = vdupq_n_f32(0.0f);
            
            for (; l + 3 < n; l += 4) {
                float32x4_t va = vld1q_f32(&a.data[i * n + l]);
                float32x4_t vb = vld1q_f32(&b.data[l * k + j]);
                vsum = vmlaq_f32(vsum, va, vb);
            }
            
            float32x2_t vsum_low = vget_low_f32(vsum);
            float32x2_t vsum_high = vget_high_f32(vsum);
            float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
            sum = vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);
            
            for (; l < n; l++) {
                sum += a.data[i * n + l] * b.data[l * k + j];
            }
            
            result.data[i * k + j] = sum;
        }
    }
    #else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += a.data[i * n + l] * b.data[l * k + j];
            }
            result.data[i * k + j] = sum;
        }
    }
    #endif
    
    return result;
}

Tensor transpose(const Tensor& a) {
    int rows = a.rows();
    int cols = a.cols();
    Tensor result(Shape{cols, rows}, false);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j * rows + i] = a.data[i * cols + j];
        }
    }
    
    return result;
}

// ============ Normalization ============

Tensor rmsnorm(const Tensor& x, float eps) {
    int n = x.size();
    float ms = 0.0f;
    for (int i = 0; i < n; i++) {
        ms += x.data[i] * x.data[i];
    }
    ms = ms / n + eps;
    float scale = 1.0f / std::sqrt(ms);
    
    Tensor result(x.shape, false);
    for (int i = 0; i < n; i++) {
        result.data[i] = x.data[i] * scale;
    }
    
    return result;
}

Tensor layernorm(const Tensor& x, float eps) {
    int n = x.size();
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += x.data[i];
    }
    mean /= n;
    
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x.data[i] - mean;
        var += diff * diff;
    }
    var /= n;
    
    float scale = 1.0f / std::sqrt(var + eps);
    
    Tensor result(x.shape, false);
    for (int i = 0; i < n; i++) {
        result.data[i] = (x.data[i] - mean) * scale;
    }
    
    return result;
}

// ============ Softmax ============

Tensor softmax(const Tensor& logits) {
    int n = logits.size();
    
    #if USE_NEON
    float max_val = logits.data[0];
    for (int i = 1; i < n; i++) {
        max_val = std::max(max_val, logits.data[i]);
    }
    #else
    float max_val = logits.data[0];
    for (int i = 1; i < n; i++) {
        max_val = std::max(max_val, logits.data[i]);
    }
    #endif
    
    Tensor exp_logits(logits.shape, false);
    float sum_exp = 0.0f;
    
    #if USE_NEON
    float32x4_t vmax = vdupq_n_f32(max_val);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(&logits.data[i]);
        float32x4_t vdiff = vsubq_f32(vx, vmax);
        float32x4_t vexp = exp_ps(vdiff);
        vst1q_f32(&exp_logits.data[i], vexp);
        
        float32x2_t vsum_low = vget_low_f32(vexp);
        float32x2_t vsum_high = vget_high_f32(vexp);
        float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
        sum_exp += vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);
    }
    for (; i < n; i++) {
        exp_logits.data[i] = std::exp(logits.data[i] - max_val);
        sum_exp += exp_logits.data[i];
    }
    #else
    for (int i = 0; i < n; i++) {
        exp_logits.data[i] = std::exp(logits.data[i] - max_val);
        sum_exp += exp_logits.data[i];
    }
    #endif
    
    Tensor result(logits.shape, false);
    
    #if USE_NEON
    float32x4_t vsum = vdupq_n_f32(sum_exp);
    int j = 0;
    for (; j + 3 < n; j += 4) {
        float32x4_t vexp = vld1q_f32(&exp_logits.data[j]);
        float32x4_t vnorm = vdivq_f32(vexp, vsum);
        vst1q_f32(&result.data[j], vnorm);
    }
    for (; j < n; j++) {
        result.data[j] = exp_logits.data[j] / sum_exp;
    }
    #else
    for (int i = 0; i < n; i++) {
        result.data[i] = exp_logits.data[i] / sum_exp;
    }
    #endif
    
    return result;
}

// ============ Attention ============

Tensor attention_scores(const Tensor& q, const Tensor& k, float scale) {
    // q: (seq_len, head_dim), k: (seq_len, head_dim)
    // Returns: (seq_len, seq_len) attention scores
    int seq_len = q.rows();
    int head_dim = q.cols();
    
    Tensor scores(Shape{seq_len, seq_len}, false);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q.data[i * head_dim + d] * k.data[j * head_dim + d];
            }
            scores.data[i * seq_len + j] = dot * scale;
        }
    }
    
    return scores;
}

Tensor attention_apply(const Tensor& weights, const Tensor& v) {
    // weights: (seq_len, seq_len), v: (seq_len, head_dim)
    // Returns: (seq_len, head_dim)
    int seq_len = v.rows();
    int head_dim = v.cols();
    
    Tensor result(Shape{seq_len, head_dim}, false);
    
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += weights.data[i * seq_len + j] * v.data[j * head_dim + d];
            }
            result.data[i * head_dim + d] = sum;
        }
    }
    
    return result;
}

// ============ Utilities ============

void zero(Tensor& t) {
    std::fill(t.data.begin(), t.data.end(), 0.0f);
}

void fill(Tensor& t, float value) {
    std::fill(t.data.begin(), t.data.end(), value);
}

Tensor clone(const Tensor& t) {
    Tensor result(t.shape, t.requires_grad);
    result.data = t.data;
    if (t.requires_grad) {
        result.grad = t.grad;
    }
    return result;
}

float sum(const Tensor& t) {
    float s = 0.0f;
    for (float v : t.data) s += v;
    return s;
}

float mean(const Tensor& t) {
    return sum(t) / t.size();
}

float max(const Tensor& t) {
    float m = t.data[0];
    for (float v : t.data) m = std::max(m, v);
    return m;
}

// ============ Random Initialization ============

void uniform_(Tensor& t, float min_val, float max_val) {
    for (float& v : t.data) {
        v = Random::uniform(min_val, max_val);
    }
}

void normal_(Tensor& t, float mean, float std) {
    for (float& v : t.data) {
        v = Random::normal(mean, std);
    }
}

void xavier_uniform_(Tensor& t) {
    int fan_in = t.cols();
    int fan_out = t.rows();
    float gain = 1.0f;
    float bound = gain * std::sqrt(6.0f / (fan_in + fan_out));
    uniform_(t, -bound, bound);
}

void xavier_normal_(Tensor& t) {
    int fan_in = t.cols();
    int fan_out = t.rows();
    float gain = 1.0f;
    float std = gain * std::sqrt(2.0f / (fan_in + fan_out));
    normal_(t, 0.0f, std);
}

void kaiming_uniform_(Tensor& t) {
    int fan_in = t.cols();
    float bound = std::sqrt(3.0f / fan_in);
    uniform_(t, -bound, bound);
}

void kaiming_normal_(Tensor& t) {
    int fan_in = t.cols();
    float std = std::sqrt(2.0f / fan_in);
    normal_(t, 0.0f, std);
}

} // namespace math
} // namespace microgpt
