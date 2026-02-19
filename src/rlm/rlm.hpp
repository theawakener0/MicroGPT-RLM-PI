#pragma once
#include "../training/trainable_model.hpp"
#include "../inference/generator.hpp"
#include "../training/tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

namespace microgpt {

using GPT = TrainableGPT;

struct RLMQuery {
    std::string text;
    int depth;
    std::optional<std::string> chunk_id;
    
    RLMQuery(const std::string& text, int depth = 0) 
        : text(text), depth(depth), chunk_id(std::nullopt) {}
};

struct RLMResponse {
    std::string text;
    float confidence;
    int tokens_used;
    bool is_final;
    int recursion_depth;
    
    RLMResponse() : confidence(0.0f), tokens_used(0), is_final(false), recursion_depth(0) {}
    
    RLMResponse(const std::string& text, float confidence, int tokens, bool is_final, int depth)
        : text(text), confidence(confidence), tokens_used(tokens), 
          is_final(is_final), recursion_depth(depth) {}
};

class ContextBuffer {
private:
    std::string context;
    std::vector<size_t> chunk_boundaries;
    size_t default_chunk_size;
    
public:
    ContextBuffer(size_t chunk_size = 256) : default_chunk_size(chunk_size) {}
    
    void set_context(const std::string& ctx) {
        context = ctx;
        compute_chunk_boundaries();
    }
    
    void append(const std::string& text) {
        context += text;
        compute_chunk_boundaries();
    }
    
    void clear() {
        context.clear();
        chunk_boundaries.clear();
    }
    
    const std::string& get_context() const { return context; }
    
    size_t size() const { return context.size(); }
    
    std::string get_chunk(size_t start, size_t end) const {
        if (start >= context.size()) return "";
        end = std::min(end, context.size());
        return context.substr(start, end - start);
    }
    
    std::string get_chunk_by_id(size_t chunk_id) const {
        if (chunk_id >= chunk_boundaries.size() - 1) return "";
        size_t start = chunk_boundaries[chunk_id];
        size_t end = (chunk_id + 1 < chunk_boundaries.size()) 
                     ? chunk_boundaries[chunk_id + 1] 
                     : context.size();
        return context.substr(start, end - start);
    }
    
    size_t num_chunks() const {
        return std::max(size_t(1), chunk_boundaries.size());
    }
    
    void set_chunk_size(size_t size) {
        default_chunk_size = size;
        compute_chunk_boundaries();
    }
    
private:
    void compute_chunk_boundaries() {
        chunk_boundaries.clear();
        if (context.empty()) return;
        
        chunk_boundaries.push_back(0);
        size_t pos = 0;
        while (pos < context.size()) {
            pos += default_chunk_size;
            if (pos < context.size()) {
                size_t newline_pos = context.find('\n', pos);
                if (newline_pos != std::string::npos && newline_pos < pos + 50) {
                    pos = newline_pos + 1;
                }
            }
            chunk_boundaries.push_back(std::min(pos, context.size()));
        }
    }
};

class ChunkProcessor {
private:
    size_t chunk_size;
    size_t overlap;
    
public:
    ChunkProcessor(size_t chunk_sz = 256, size_t overlap_sz = 32)
        : chunk_size(chunk_sz), overlap(overlap_sz) {}
    
    std::vector<std::string> chunk_by_tokens(const std::string& text) const;
    std::vector<std::string> chunk_by_sentences(const std::string& text) const;
    std::vector<std::string> get_sliding_windows(const std::string& text) const;
    
    void set_chunk_size(size_t size) { chunk_size = size; }
    void set_overlap(size_t ov) { overlap = ov; }
};

class RLMScaffolding {
private:
    std::unique_ptr<GPT> model;
    std::unique_ptr<Tokenizer> tokenizer;
    ContextBuffer external_context;
    
    int max_depth;
    float exit_threshold;
    int min_depth;
    bool verbose;
    
    std::vector<RLMResponse> response_history;
    
public:
    RLMScaffolding();
    RLMScaffolding(const ModelConfig& config, int max_depth = 3, 
                   float exit_thresh = 0.8f, int min_d = 1);
    
    void set_model(std::unique_ptr<GPT> m);
    void set_tokenizer(std::unique_ptr<Tokenizer> t);
    void set_context(const std::string& ctx);
    void append_context(const std::string& text);
    void clear_context();
    
    RLMResponse completion(const std::string& query);
    RLMResponse recursive_query(const RLMQuery& query);
    RLMResponse process_chunk(const std::string& chunk, int depth);
    RLMResponse aggregate(const std::vector<RLMResponse>& chunk_results);
    
    bool should_continue(float confidence, int depth) const;
    bool should_exit_early(float confidence, int depth) const;
    
    const std::vector<RLMResponse>& get_history() const { return response_history; }
    void clear_history();
    
    void set_verbose(bool v) { verbose = v; }
    
    std::string generate_text(const std::string& prompt, int max_tokens = 100);
    
    struct Stats {
        int total_calls;
        int recursive_calls;
        int total_tokens;
        float avg_confidence;
    };
    
    Stats get_stats() const;
    
private:
    RLMResponse call_model(const std::string& prompt);
    float compute_confidence(const Tensor& logits);
    std::string build_system_prompt() const;
    std::string build_chunk_prompt(const std::string& chunk, const std::string& query) const;
};

class NativeRLM : public GPT {
public:
    NativeRLM() : GPT(ModelConfig{}) {}
    NativeRLM(const ModelConfig& config);
    
    Linear exit_head;
    
    int max_recursion_depth;
    float exit_confidence_threshold;
    int min_recursion_depth;
    
    std::vector<Tensor*> all_params;
    
    struct RecursionResult {
        Tensor logits;
        int steps_used;
        bool exited_early;
        float exit_confidence;
    };
    
    RecursionResult recursive_forward(const std::vector<int>& token_ids, int max_depth = -1);
    
    void init_weights(float std = 0.02f);
    int num_parameters() const;
    
    float get_exit_prob(const Tensor& state);
    
    void backward_with_exit(const std::vector<int>& target_ids, 
                          const std::vector<bool>& exit_targets);
};

class RLMTokenizer {
private:
    std::unique_ptr<Tokenizer> base_tokenizer;
    std::vector<int> recursive_tokens;
    std::vector<int> aggregate_tokens;
    std::vector<int> exit_tokens;
    
public:
    RLMTokenizer();
    
    void build(const std::vector<std::string>& texts);
    
    std::vector<int> encode_recursive_call(const std::string& query);
    std::vector<int> encode_aggregate(const std::vector<std::string>& chunk_results);
    std::vector<int> encode_exit();
    
    std::string decode(const std::vector<int>& ids) const;
    
    int recursive_token_id() const;
    int aggregate_token_id() const;
    int exit_token_id() const;
};

}
