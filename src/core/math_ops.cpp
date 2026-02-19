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

// ============ NEON-Optimized Attention ============

void attention_scores_neon(float* scores, const float* q, const float* k, 
                           int seq_len, int head_dim, float scale) {
    #if USE_NEON
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            int d = 0;
            float32x4_t vsum = vdupq_n_f32(0.0f);
            
            for (; d + 3 < head_dim; d += 4) {
                float32x4_t vq = vld1q_f32(&q[i * head_dim + d]);
                float32x4_t vk = vld1q_f32(&k[j * head_dim + d]);
                vsum = vmlaq_f32(vsum, vq, vk);
            }
            
            float32x2_t vsum_low = vget_low_f32(vsum);
            float32x2_t vsum_high = vget_high_f32(vsum);
            float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
            dot = vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);
            
            for (; d < head_dim; d++) {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            
            scores[i * seq_len + j] = dot * scale;
        }
    }
    #else
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
    #endif
}

void softmax_causal_neon(float* scores, int seq_len) {
    #if USE_NEON
    for (int i = 0; i < seq_len; i++) {
        float max_val = scores[i * seq_len];
        for (int j = 1; j <= i; j++) {
            max_val = std::max(max_val, scores[i * seq_len + j]);
        }
        
        float sum_exp = 0.0f;
        int j = 0;
        float32x4_t vmax = vdupq_n_f32(max_val);
        
        for (; j + 3 <= i + 1; j += 4) {
            float32x4_t vx = vld1q_f32(&scores[i * seq_len + j]);
            float32x4_t vdiff = vsubq_f32(vx, vmax);
            float32x4_t vexp = exp_ps(vdiff);
            vst1q_f32(&scores[i * seq_len + j], vexp);
            
            float32x2_t vsum_low = vget_low_f32(vexp);
            float32x2_t vsum_high = vget_high_f32(vexp);
            float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
            sum_exp += vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);
        }
        
        for (; j <= i; j++) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_val);
            sum_exp += scores[i * seq_len + j];
        }
        
        j = 0;
        float32x4_t vsum = vdupq_n_f32(sum_exp);
        for (; j + 3 <= i + 1; j += 4) {
            float32x4_t vexp = vld1q_f32(&scores[i * seq_len + j]);
            float32x4_t vnorm = vdivq_f32(vexp, vsum);
            vst1q_f32(&scores[i * seq_len + j], vnorm);
        }
        for (; j <= i; j++) {
            scores[i * seq_len + j] /= sum_exp;
        }
        
        for (int j = i + 1; j < seq_len; j++) {
            scores[i * seq_len + j] = 0.0f;
        }
    }
    #else
    for (int i = 0; i < seq_len; i++) {
        float max_val = scores[i * seq_len];
        for (int j = 1; j <= i; j++) {
            max_val = std::max(max_val, scores[i * seq_len + j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j <= i; j++) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_val);
            sum_exp += scores[i * seq_len + j];
        }
        
        for (int j = 0; j <= i; j++) {
            scores[i * seq_len + j] /= sum_exp;
        }
        
        for (int j = i + 1; j < seq_len; j++) {
            scores[i * seq_len + j] = 0.0f;
        }
    }
    #endif
}

void attention_apply_neon(float* out, const float* weights, const float* v,
                          int seq_len, int head_dim) {
    #if USE_NEON
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            int j = 0;
            float32x4_t vsum = vdupq_n_f32(0.0f);
            
            for (; j + 3 <= i + 1; j += 4) {
                float32x4_t vw = vld1q_f32(&weights[i * seq_len + j]);
                float32x4_t vv;
                vv = vsetq_lane_f32(v[(j + 0) * head_dim + d], vv, 0);
                vv = vsetq_lane_f32(v[(j + 1) * head_dim + d], vv, 1);
                vv = vsetq_lane_f32(v[(j + 2) * head_dim + d], vv, 2);
                vv = vsetq_lane_f32(v[(j + 3) * head_dim + d], vv, 3);
                vsum = vmlaq_f32(vsum, vw, vv);
            }
            
            float32x2_t vsum_low = vget_low_f32(vsum);
            float32x2_t vsum_high = vget_high_f32(vsum);
            float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
            sum = vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);
            
            for (; j <= i; j++) {
                sum += weights[i * seq_len + j] * v[j * head_dim + d];
            }
            
            out[i * head_dim + d] = sum;
        }
    }
    #else
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                sum += weights[i * seq_len + j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = sum;
        }
    }
    #endif
}

void scaled_dot_product_attention(float* out, const float* q, const float* k, 
                                  const float* v, int seq_len, int head_dim, float scale) {
    std::vector<float> scores(seq_len * seq_len);
    
    attention_scores_neon(scores.data(), q, k, seq_len, head_dim, scale);
    softmax_causal_neon(scores.data(), seq_len);
    attention_apply_neon(out, scores.data(), v, seq_len, head_dim);
}

// ============ Flash Attention ============

// Block size optimized for Pi 5 L2 cache (~512KB)
// Each block: 64 x 64 floats = 16KB
// Q, K, V blocks + accumulator = ~64KB per head
// With 4 heads: ~256KB, leaving room for other data
constexpr int FLASH_ATTN_BLOCK_SIZE = 64;

Tensor flash_attention_forward(
    const Tensor& q,    // (seq_len, head_dim)
    const Tensor& k,    // (seq_len, head_dim)
    const Tensor& v,    // (seq_len, head_dim)
    float scale,
    bool causal
) {
    int seq_len = q.rows();
    int head_dim = q.cols();
    
    Tensor output(Shape{seq_len, head_dim}, false);
    
    // Online softmax statistics: running max and sum
    std::vector<float> m(seq_len, -std::numeric_limits<float>::infinity());
    std::vector<float> l(seq_len, 0.0f);
    
    // Process in blocks
    int num_blocks = (seq_len + FLASH_ATTN_BLOCK_SIZE - 1) / FLASH_ATTN_BLOCK_SIZE;
    
    // Temporary storage for block computations
    std::vector<float> s_block(FLASH_ATTN_BLOCK_SIZE * FLASH_ATTN_BLOCK_SIZE);
    std::vector<float> p_block(FLASH_ATTN_BLOCK_SIZE * FLASH_ATTN_BLOCK_SIZE);
    std::vector<float> o_block(FLASH_ATTN_BLOCK_SIZE * head_dim);
    
    for (int i = 0; i < num_blocks; i++) {
        int row_start = i * FLASH_ATTN_BLOCK_SIZE;
        int row_end = std::min(row_start + FLASH_ATTN_BLOCK_SIZE, seq_len);
        int row_count = row_end - row_start;
        
        // Initialize output accumulator for this block
        std::fill(o_block.begin(), o_block.begin() + row_count * head_dim, 0.0f);
        
        for (int j = 0; j <= (causal ? i : num_blocks - 1); j++) {
            int col_start = j * FLASH_ATTN_BLOCK_SIZE;
            int col_end = std::min(col_start + FLASH_ATTN_BLOCK_SIZE, seq_len);
            int col_count = col_end - col_start;
            
            // Compute S = Q[i] @ K[j]^T
            for (int ii = 0; ii < row_count; ii++) {
                for (int jj = 0; jj < col_count; jj++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q.data[(row_start + ii) * head_dim + d] * 
                               k.data[(col_start + jj) * head_dim + d];
                    }
                    s_block[ii * col_count + jj] = dot * scale;
                }
                
                // Apply causal mask if needed
                if (causal && j == i) {
                    for (int jj = 0; jj < col_count; jj++) {
                        if (col_start + jj > row_start + ii) {
                            s_block[ii * col_count + jj] = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
            
            // Online softmax update
            for (int ii = 0; ii < row_count; ii++) {
                int global_row = row_start + ii;
                
                // Find max in this block
                float m_block = s_block[ii * col_count];
                for (int jj = 1; jj < col_count; jj++) {
                    m_block = std::max(m_block, s_block[ii * col_count + jj]);
                }
                
                // Compute exp and sum
                float l_block = 0.0f;
                for (int jj = 0; jj < col_count; jj++) {
                    p_block[ii * col_count + jj] = std::exp(s_block[ii * col_count + jj] - m_block);
                    l_block += p_block[ii * col_count + jj];
                }
                
                // Update running statistics
                float m_new = std::max(m[global_row], m_block);
                float exp_m_old = std::exp(m[global_row] - m_new);
                float exp_m_block = std::exp(m_block - m_new);
                
                l[global_row] = l[global_row] * exp_m_old + l_block * exp_m_block;
                
                // Rescale and accumulate output
                float rescale_old = (m[global_row] == -std::numeric_limits<float>::infinity()) ? 0.0f : exp_m_old;
                
                for (int d = 0; d < head_dim; d++) {
                    o_block[ii * head_dim + d] *= rescale_old;
                    for (int jj = 0; jj < col_count; jj++) {
                        o_block[ii * head_dim + d] += exp_m_block * p_block[ii * col_count + jj] * 
                                                      v.data[(col_start + jj) * head_dim + d];
                    }
                }
                
                m[global_row] = m_new;
            }
        }
        
        // Normalize output
        for (int ii = 0; ii < row_count; ii++) {
            int global_row = row_start + ii;
            for (int d = 0; d < head_dim; d++) {
                output.data[global_row * head_dim + d] = o_block[ii * head_dim + d] / l[global_row];
            }
        }
    }
    
    return output;
}

FlashAttentionGrad flash_attention_backward(
    const Tensor& grad_output,  // (seq_len, head_dim)
    const Tensor& q,            // (seq_len, head_dim)
    const Tensor& k,            // (seq_len, head_dim)
    const Tensor& v,            // (seq_len, head_dim)
    float scale,
    bool causal
) {
    int seq_len = q.rows();
    int head_dim = q.cols();
    
    FlashAttentionGrad grad;
    grad.grad_q = Tensor(Shape{seq_len, head_dim}, false);
    grad.grad_k = Tensor(Shape{seq_len, head_dim}, false);
    grad.grad_v = Tensor(Shape{seq_len, head_dim}, false);
    
    math::fill(grad.grad_q, 0.0f);
    math::fill(grad.grad_k, 0.0f);
    math::fill(grad.grad_v, 0.0f);
    
    // Compute attention scores and softmax
    Tensor scores = attention_scores(q, k, scale);
    
    // Apply causal mask and softmax
    for (int i = 0; i < seq_len; i++) {
        float max_val = scores.data[i * seq_len];
        for (int j = 1; j < seq_len; j++) {
            max_val = std::max(max_val, scores.data[i * seq_len + j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            if (causal && j > i) {
                scores.data[i * seq_len + j] = 0.0f;
            } else {
                scores.data[i * seq_len + j] = std::exp(scores.data[i * seq_len + j] - max_val);
                sum_exp += scores.data[i * seq_len + j];
            }
        }
        
        for (int j = 0; j < seq_len; j++) {
            scores.data[i * seq_len + j] /= sum_exp;
        }
    }
    
    // dL/dV = P^T @ dO
    for (int j = 0; j < seq_len; j++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                if (!causal || j <= i) {
                    sum += scores.data[i * seq_len + j] * grad_output.data[i * head_dim + d];
                }
            }
            grad.grad_v.data[j * head_dim + d] = sum;
        }
    }
    
    // dL/dP = dO @ V^T
    Tensor grad_scores(Shape{seq_len, seq_len}, false);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            if (causal && j > i) {
                grad_scores.data[i * seq_len + j] = 0.0f;
            } else {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    sum += grad_output.data[i * head_dim + d] * v.data[j * head_dim + d];
                }
                grad_scores.data[i * seq_len + j] = sum;
            }
        }
    }
    
    // dL/dS = P * (dL/dP - sum(dL/dP * P))
    for (int i = 0; i < seq_len; i++) {
        float sum_p_grad = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum_p_grad += grad_scores.data[i * seq_len + j] * scores.data[i * seq_len + j];
        }
        
        for (int j = 0; j < seq_len; j++) {
            grad_scores.data[i * seq_len + j] = scores.data[i * seq_len + j] * 
                                                (grad_scores.data[i * seq_len + j] - sum_p_grad);
        }
    }
    
    // dL/dQ = dL/dS @ K
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                if (!causal || j <= i) {
                    sum += grad_scores.data[i * seq_len + j] * k.data[j * head_dim + d];
                }
            }
            grad.grad_q.data[i * head_dim + d] = sum * scale;
        }
    }
    
    // dL/dK = dL/dS^T @ Q
    for (int j = 0; j < seq_len; j++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                if (!causal || j <= i) {
                    sum += grad_scores.data[i * seq_len + j] * q.data[i * head_dim + d];
                }
            }
            grad.grad_k.data[j * head_dim + d] = sum * scale;
        }
    }
    
    return grad;
}

// ============ Quantization ============

QuantizedTensor quantize(const Tensor& input, bool per_channel) {
    int rows = input.rows();
    int cols = input.cols();
    
    QuantizedTensor result(rows, cols, per_channel);
    
    if (per_channel) {
        // Per-row quantization
        for (int i = 0; i < rows; i++) {
            // Find min and max for this row
            float min_val = input.data[i * cols];
            float max_val = input.data[i * cols];
            for (int j = 1; j < cols; j++) {
                min_val = std::min(min_val, input.data[i * cols + j]);
                max_val = std::max(max_val, input.data[i * cols + j]);
            }
            
            // Compute scale: map [min, max] to [-127, 127]
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (abs_max < 1e-8f) abs_max = 1e-8f;
            result.scale[i] = abs_max / 127.0f;
            
            // Quantize
            for (int j = 0; j < cols; j++) {
                float val = input.data[i * cols + j];
                int32_t qval = static_cast<int32_t>(std::round(val / result.scale[i]));
                qval = std::max(-127, std::min(127, qval));  // Clamp to INT8 range
                result.data[i * cols + j] = static_cast<int8_t>(qval);
            }
        }
    } else {
        // Per-tensor quantization
        float min_val = input.data[0];
        float max_val = input.data[0];
        for (int i = 1; i < rows * cols; i++) {
            min_val = std::min(min_val, input.data[i]);
            max_val = std::max(max_val, input.data[i]);
        }
        
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        if (abs_max < 1e-8f) abs_max = 1e-8f;
        result.scale[0] = abs_max / 127.0f;
        
        for (int i = 0; i < rows * cols; i++) {
            int32_t qval = static_cast<int32_t>(std::round(input.data[i] / result.scale[0]));
            qval = std::max(-127, std::min(127, qval));
            result.data[i] = static_cast<int8_t>(qval);
        }
    }
    
    return result;
}

Tensor dequantize(const QuantizedTensor& input) {
    Tensor result(Shape{input.rows, input.cols}, false);
    
    if (input.scale.size() == 1) {
        // Per-tensor dequantization
        for (int i = 0; i < input.size(); i++) {
            result.data[i] = static_cast<float>(input.data[i]) * input.scale[0];
        }
    } else {
        // Per-channel dequantization
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                result.data[i * input.cols + j] = 
                    static_cast<float>(input.data[i * input.cols + j]) * input.scale[i];
            }
        }
    }
    
    return result;
}

Tensor int8_matmul(const QuantizedTensor& a, const Tensor& b, const std::vector<float>& a_scale) {
    int m = a.rows;
    int k = a.cols;
    int n = b.cols();
    
    Tensor result(Shape{m, n}, false);
    
    #if USE_NEON
    // Use NEON-optimized version if available
    if (a_scale.size() == 1) {
        // Per-tensor quantization
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int32_t sum = 0;
                int l = 0;
                
                // Process 8 elements at a time with NEON
                int8x8_t va;
                for (; l + 7 < k; l += 8) {
                    // Load 8 int8 values
                    int8_t vals[8];
                    for (int v = 0; v < 8; v++) {
                        vals[v] = a.data[i * k + l + v];
                    }
                    va = vld1_s8(vals);
                    
                    // Load 8 float values, convert to int8
                    float32x4_t vb_low = vld1q_f32(&b.data[l * n + j]);
                    float32x4_t vb_high = vld1q_f32(&b.data[(l + 4) * n + j]);
                    
                    // Convert to int8 (simplified - in practice you'd want proper rounding)
                    int8_t bvals[8];
                    for (int v = 0; v < 8; v++) {
                        bvals[v] = static_cast<int8_t>(b.data[(l + v) * n + j] * 127.0f);
                    }
                    int8x8_t vb = vld1_s8(bvals);
                    
                    // Dot product
                    int16x8_t prod = vmull_s8(va, vb);
                    int32x4_t prod_low = vmovl_s16(vget_low_s16(prod));
                    int32x4_t prod_high = vmovl_s16(vget_high_s16(prod));
                    
                    sum += vgetq_lane_s32(prod_low, 0) + vgetq_lane_s32(prod_low, 1) +
                           vgetq_lane_s32(prod_low, 2) + vgetq_lane_s32(prod_low, 3) +
                           vgetq_lane_s32(prod_high, 0) + vgetq_lane_s32(prod_high, 1) +
                           vgetq_lane_s32(prod_high, 2) + vgetq_lane_s32(prod_high, 3);
                }
                
                // Handle remaining elements
                for (; l < k; l++) {
                    sum += static_cast<int32_t>(a.data[i * k + l]) * 
                           static_cast<int32_t>(b.data[l * n + j] * 127.0f);
                }
                
                result.data[i * n + j] = static_cast<float>(sum) * a_scale[0] / 127.0f;
            }
        }
    } else {
        // Per-channel quantization
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += static_cast<float>(a.data[i * k + l]) * b.data[l * n + j];
                }
                result.data[i * n + j] = sum * a_scale[i];
            }
        }
    }
    #else
    // Fallback scalar implementation
    if (a_scale.size() == 1) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += static_cast<float>(a.data[i * k + l]) * b.data[l * n + j];
                }
                result.data[i * n + j] = sum * a_scale[0];
            }
        }
    } else {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += static_cast<float>(a.data[i * k + l]) * b.data[l * n + j];
                }
                result.data[i * n + j] = sum * a_scale[i];
            }
        }
    }
    #endif
    
    return result;
}

Tensor ste_quantize(const Tensor& input, float* scale) {
    // Compute scale
    float min_val = input.data[0];
    float max_val = input.data[0];
    for (size_t i = 1; i < input.data.size(); i++) {
        min_val = std::min(min_val, input.data[i]);
        max_val = std::max(max_val, input.data[i]);
    }
    
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    if (abs_max < 1e-8f) abs_max = 1e-8f;
    *scale = abs_max / 127.0f;
    
    // Forward: quantize then dequantize (simulated quantization)
    Tensor output(input.shape, input.requires_grad);
    for (size_t i = 0; i < input.data.size(); i++) {
        int32_t qval = static_cast<int32_t>(std::round(input.data[i] / *scale));
        qval = std::max(-127, std::min(127, qval));
        output.data[i] = static_cast<float>(qval) * (*scale);
    }
    
    // For backward pass, we need to store the original gradient
    // This is handled by the autograd system
    
    return output;
}

// ============ Operator Fusion ============

QKVTensors fused_qkv_projection(
    const Tensor& x,              // (seq_len, embed_dim)
    const Tensor& w_qkv,          // (embed_dim, 3 * embed_dim) - concatenated weights
    const std::optional<Tensor>& b_qkv  // optional bias (3 * embed_dim)
) {
    int seq_len = x.rows();
    int embed_dim = x.cols();
    
    // Single matmul: (seq_len, embed_dim) @ (embed_dim, 3*embed_dim) -> (seq_len, 3*embed_dim)
    Tensor qkv_combined = matmul(x, w_qkv);
    
    // Add bias if provided
    if (b_qkv.has_value()) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < 3 * embed_dim; j++) {
                qkv_combined.data[i * 3 * embed_dim + j] += b_qkv->data[j];
            }
        }
    }
    
    // Split into Q, K, V
    QKVTensors result;
    result.q = Tensor(Shape{seq_len, embed_dim}, false);
    result.k = Tensor(Shape{seq_len, embed_dim}, false);
    result.v = Tensor(Shape{seq_len, embed_dim}, false);
    
    for (int i = 0; i < seq_len; i++) {
        // Q: first embed_dim columns
        for (int j = 0; j < embed_dim; j++) {
            result.q.data[i * embed_dim + j] = qkv_combined.data[i * 3 * embed_dim + j];
        }
        // K: second embed_dim columns
        for (int j = 0; j < embed_dim; j++) {
            result.k.data[i * embed_dim + j] = qkv_combined.data[i * 3 * embed_dim + embed_dim + j];
        }
        // V: third embed_dim columns
        for (int j = 0; j < embed_dim; j++) {
            result.v.data[i * embed_dim + j] = qkv_combined.data[i * 3 * embed_dim + 2 * embed_dim + j];
        }
    }
    
    return result;
}

Tensor fused_ffn(
    const Tensor& x,              // (seq_len, embed_dim)
    const Tensor& w1,             // (embed_dim, hidden_dim)
    const Tensor& w2,             // (hidden_dim, embed_dim)
    const std::optional<Tensor>& b1,
    const std::optional<Tensor>& b2,
    bool use_gelu
) {
    int seq_len = x.rows();
    int embed_dim = x.cols();
    int hidden_dim = w1.cols();
    
    // First linear layer
    Tensor hidden = matmul(x, w1);
    
    // Add bias and activation
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            float val = hidden.data[i * hidden_dim + j];
            if (b1.has_value()) {
                val += b1->data[j];
            }
            // Activation
            if (use_gelu) {
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float x3 = val * val * val;
                val = val * 0.5f * (1.0f + std::tanh(0.7978845608f * (val + 0.044715f * x3)));
            } else {
                // SiLU (Swish): x * sigmoid(x)
                val = val / (1.0f + std::exp(-val));
            }
            hidden.data[i * hidden_dim + j] = val;
        }
    }
    
    // Second linear layer
    Tensor output = matmul(hidden, w2);
    
    // Add bias
    if (b2.has_value()) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embed_dim; j++) {
                output.data[i * embed_dim + j] += b2->data[j];
            }
        }
    }
    
    return output;
}

Tensor fused_rmsnorm_residual(
    const Tensor& x,
    const Tensor& residual,
    const Tensor& weight,
    float eps
) {
    int seq_len = x.rows();
    int embed_dim = x.cols();
    
    Tensor output(Shape{seq_len, embed_dim}, false);
    
    // Add residual
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            output.data[i * embed_dim + j] = x.data[i * embed_dim + j] + residual.data[i * embed_dim + j];
        }
    }
    
    // RMSNorm
    for (int i = 0; i < seq_len; i++) {
        float ms = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            float val = output.data[i * embed_dim + j];
            ms += val * val;
        }
        ms = ms / embed_dim + eps;
        float norm_scale = 1.0f / std::sqrt(ms);
        
        for (int j = 0; j < embed_dim; j++) {
            output.data[i * embed_dim + j] = output.data[i * embed_dim + j] * norm_scale * weight.data[j];
        }
    }
    
    return output;
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
