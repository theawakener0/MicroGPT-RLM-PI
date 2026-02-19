#include "rlm.hpp"
#include "../utils/logger.hpp"
#include "../utils/random.hpp"
#include <algorithm>
#include <sstream>
#include <cmath>

namespace microgpt {

std::vector<std::string> ChunkProcessor::chunk_by_tokens(const std::string& text) const {
    std::vector<std::string> chunks;
    size_t pos = 0;
    
    while (pos < text.size()) {
        size_t end = std::min(pos + chunk_size, text.size());
        chunks.push_back(text.substr(pos, end - pos));
        pos = end;
        
        if (pos < text.size() && overlap > 0 && pos > overlap) {
            pos -= overlap;
        }
    }
    
    return chunks;
}

std::vector<std::string> ChunkProcessor::chunk_by_sentences(const std::string& text) const {
    std::vector<std::string> chunks;
    std::istringstream iss(text);
    std::string sentence;
    std::string current_chunk;
    
    while (std::getline(iss, sentence, '.')) {
        if (!sentence.empty() && sentence.back() != '\n') {
            sentence += ".";
        }
        
        if (current_chunk.size() + sentence.size() > chunk_size && !current_chunk.empty()) {
            chunks.push_back(current_chunk);
            current_chunk.clear();
        }
        
        current_chunk += sentence;
    }
    
    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }
    
    return chunks;
}

std::vector<std::string> ChunkProcessor::get_sliding_windows(const std::string& text) const {
    std::vector<std::string> windows;
    
    if (text.size() <= chunk_size) {
        windows.push_back(text);
        return windows;
    }
    
    size_t step = chunk_size - overlap;
    size_t pos = 0;
    
    while (pos < text.size()) {
        size_t end = std::min(pos + chunk_size, text.size());
        windows.push_back(text.substr(pos, end - pos));
        
        if (end >= text.size()) break;
        pos += step;
    }
    
    return windows;
}

RLMScaffolding::RLMScaffolding()
    : max_depth(3), exit_threshold(0.8f), min_depth(1), verbose(false) {}

RLMScaffolding::RLMScaffolding(const ModelConfig& config, int max_depth, 
                               float exit_thresh, int min_d)
    : max_depth(max_depth), exit_threshold(exit_thresh), min_depth(min_d), 
      verbose(false) {
    model = std::make_unique<GPT>(config);
    model->init_weights();
}

void RLMScaffolding::set_model(std::unique_ptr<GPT> m) {
    model = std::move(m);
}

void RLMScaffolding::set_tokenizer(std::unique_ptr<Tokenizer> t) {
    tokenizer = std::move(t);
}

void RLMScaffolding::set_context(const std::string& ctx) {
    external_context.set_context(ctx);
}

void RLMScaffolding::append_context(const std::string& text) {
    external_context.append(text);
}

void RLMScaffolding::clear_context() {
    external_context.clear();
}

RLMResponse RLMScaffolding::completion(const std::string& query) {
    response_history.clear();
    
    RLMQuery root_query(query, 0);
    return recursive_query(root_query);
}

RLMResponse RLMScaffolding::recursive_query(const RLMQuery& query) {
    if (verbose) {
        Logger::info("RLM Query [depth=" + std::to_string(query.depth) + "]: " + 
                    query.text.substr(0, 50) + "...");
    }
    
    if (query.depth >= max_depth || should_exit_early(1.0f, query.depth)) {
        return call_model(query.text);
    }
    
    size_t num_chunks = external_context.num_chunks();
    
    if (num_chunks <= 1 || external_context.size() < 500) {
        return call_model(query.text);
    }
    
    std::vector<RLMResponse> chunk_results;
    
    for (size_t i = 0; i < num_chunks; i++) {
        std::string chunk = external_context.get_chunk_by_id(i);
        
        if (chunk.empty()) continue;
        
        RLMResponse chunk_response = process_chunk(chunk, query.depth + 1);
        chunk_results.push_back(chunk_response);
        
        if (verbose) {
            Logger::info("Chunk " + std::to_string(i) + "/" + std::to_string(num_chunks) + 
                        " done, confidence: " + std::to_string(chunk_response.confidence));
        }
    }
    
    RLMResponse aggregated = aggregate(chunk_results);
    response_history.push_back(aggregated);
    
    return aggregated;
}

RLMResponse RLMScaffolding::process_chunk(const std::string& chunk, int depth) {
    std::string prompt = build_chunk_prompt(chunk, "");
    
    RLMResponse response = call_model(prompt);
    response.recursion_depth = depth;
    
    if (should_continue(response.confidence, depth)) {
        RLMQuery sub_query(response.text, depth);
        return recursive_query(sub_query);
    }
    
    return response;
}

RLMResponse RLMScaffolding::aggregate(const std::vector<RLMResponse>& chunk_results) {
    if (chunk_results.empty()) {
        return RLMResponse("", 0.0f, 0, true, 0);
    }
    
    if (chunk_results.size() == 1) {
        return chunk_results[0];
    }
    
    float total_confidence = 0.0f;
    int total_tokens = 0;
    std::string combined_text;
    
    for (const auto& r : chunk_results) {
        total_confidence += r.confidence;
        total_tokens += r.tokens_used;
        
        if (!r.text.empty()) {
            if (!combined_text.empty() && combined_text.back() != ' ' && 
                combined_text.back() != '\n' && r.text.front() != ' ') {
                combined_text += " ";
            }
            combined_text += r.text;
        }
    }
    
    float avg_confidence = total_confidence / chunk_results.size();
    
    return RLMResponse(combined_text, avg_confidence, total_tokens, true, 0);
}

bool RLMScaffolding::should_continue(float confidence, int depth) const {
    if (depth >= max_depth) return false;
    if (depth < min_depth) return true;
    
    return confidence < exit_threshold;
}

bool RLMScaffolding::should_exit_early(float confidence, int depth) const {
    if (depth < min_depth) return false;
    return confidence >= exit_threshold;
}

void RLMScaffolding::clear_history() {
    response_history.clear();
}

std::string RLMScaffolding::generate_text(const std::string& prompt, int max_tokens) {
    if (!model || !tokenizer) {
        Logger::warning("Model or tokenizer not set for RLM");
        return "";
    }
    
    Generator gen(max_tokens, tokenizer->eos_id);
    gen.echo = false;
    
    std::string result = gen.generate(*model, *tokenizer, prompt);
    
    return result;
}

RLMResponse RLMScaffolding::call_model(const std::string& prompt) {
    if (!model || !tokenizer) {
        Logger::error("Model or tokenizer not set");
        return RLMResponse("", 0.0f, 0, true, 0);
    }
    
    std::string full_prompt = build_system_prompt() + "\n\nContext: " + 
                             external_context.get_context() + 
                             "\n\nQuery: " + prompt + "\n\nAnswer:";
    
    std::string result = generate_text(full_prompt, 100);
    
    Tensor logits = model->forward(tokenizer->encode(full_prompt));
    float confidence = compute_confidence(logits);
    
    int tokens = tokenizer->encode(result).size();
    
    return RLMResponse(result, confidence, tokens, true, 0);
}

float RLMScaffolding::compute_confidence(const Tensor& logits) {
    int seq_len = logits.rows();
    int vocab = logits.cols();
    
    if (seq_len == 0 || vocab == 0) return 0.0f;
    
    int last_idx = seq_len - 1;
    
    float max_val = logits.data[last_idx * vocab];
    for (int i = 1; i < vocab; i++) {
        max_val = std::max(max_val, logits.data[last_idx * vocab + i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab; i++) {
        sum_exp += std::exp(logits.data[last_idx * vocab + i] - max_val);
    }
    
    float max_prob = std::exp(max_val - max_val) / sum_exp;
    
    return max_prob;
}

std::string RLMScaffolding::build_system_prompt() const {
    return "You are aRecursive Language Model (RLM) that processes long context "
           "by examining it in chunks. Break down complex queries into parts, "
           "analyze each relevant section, then provide a synthesized answer.";
}

std::string RLMScaffolding::build_chunk_prompt(const std::string& chunk, 
                                               const std::string& query) const {
    return "Context chunk:\n" + chunk + "\n\n" +
           (query.empty() ? "Provide a summary of this chunk." : "Query: " + query);
}

RLMScaffolding::Stats RLMScaffolding::get_stats() const {
    Stats s{};
    s.total_calls = 1;
    s.recursive_calls = 0;
    s.total_tokens = 0;
    s.avg_confidence = 0.0f;
    
    float total_conf = 0.0f;
    
    for (const auto& r : response_history) {
        s.total_tokens += r.tokens_used;
        s.recursive_calls += r.recursion_depth;
        total_conf += r.confidence;
    }
    
    if (!response_history.empty()) {
        s.avg_confidence = total_conf / response_history.size();
    }
    
    return s;
}

NativeRLM::NativeRLM(const ModelConfig& config) : GPT(config) {
    max_recursion_depth = config.recursion_steps;
    exit_confidence_threshold = config.exit_threshold;
    min_recursion_depth = config.min_recursion_steps;
    
    exit_head = Linear(config.embed_dim, 1, false);
    math::fill(exit_head.weight, 0.0f);
}

NativeRLM::RecursionResult NativeRLM::recursive_forward(const std::vector<int>& token_ids, 
                                                       int max_depth) {
    RecursionResult result;
    result.steps_used = 0;
    result.exited_early = false;
    result.exit_confidence = 0.0f;
    
    if (max_depth < 0) {
        max_depth = max_recursion_depth;
    }
    
    int seq_len = token_ids.size();
    Tensor token_emb = token_embedding.forward(token_ids);
    Tensor pos_emb = pos_embedding.forward(seq_len);
    
    Tensor state(token_emb.shape, false);
    for (int i = 0; i < state.size(); i++) {
        state.data[i] = token_emb.data[i] + pos_emb.data[i];
    }
    
    for (int step = 0; step < max_depth; step++) {
        Tensor normalized = ln_f.forward(state);
        
        Tensor new_state = normalized;
        for (auto& layer : layers) {
            new_state = layer.forward(new_state);
        }
        
        for (int i = 0; i < state.size(); i++) {
            state.data[i] = state.data[i] + new_state.data[i];
        }
        
        float exit_prob = get_exit_prob(state);
        
        if (step >= min_recursion_depth - 1 && exit_prob > exit_confidence_threshold) {
            result.exited_early = true;
            result.exit_confidence = exit_prob;
            break;
        }
        
        result.steps_used++;
    }
    
    state = ln_f.forward(state);
    Tensor logits = lm_head.forward(state);
    result.logits = logits;
    
    return result;
}

float NativeRLM::get_exit_prob(const Tensor& state) {
    Tensor exit_logits = exit_head.forward(state);
    
    int last_idx = state.rows() - 1;
    float exit_score = exit_logits.data[last_idx];
    
    return 1.0f / (1.0f + std::exp(-exit_score));
}

void NativeRLM::init_weights(float std) {
    GPT::init_weights(std);
    math::fill(exit_head.weight, 0.0f);
}

int NativeRLM::num_parameters() const {
    return GPT::num_parameters() + exit_head.num_parameters();
}

RLMTokenizer::RLMTokenizer() {}

void RLMTokenizer::build(const std::vector<std::string>& texts) {
    base_tokenizer = std::make_unique<Tokenizer>();
    base_tokenizer->build(texts);
    
    recursive_tokens = {base_tokenizer->bos_id};
    aggregate_tokens = {base_tokenizer->eos_id};
    exit_tokens = {base_tokenizer->eos_id};
}

std::vector<int> RLMTokenizer::encode_recursive_call(const std::string& query) {
    std::vector<int> ids = base_tokenizer->encode(query);
    ids.insert(ids.begin(), recursive_tokens.begin(), recursive_tokens.end());
    return ids;
}

std::vector<int> RLMTokenizer::encode_aggregate(const std::vector<std::string>& chunk_results) {
    std::vector<int> ids;
    for (const auto& chunk : chunk_results) {
        std::vector<int> chunk_ids = base_tokenizer->encode(chunk);
        ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
        ids.insert(ids.end(), aggregate_tokens.begin(), aggregate_tokens.end());
    }
    return ids;
}

std::vector<int> RLMTokenizer::encode_exit() {
    return exit_tokens;
}

std::string RLMTokenizer::decode(const std::vector<int>& ids) const {
    if (!base_tokenizer) return "";
    return base_tokenizer->decode(ids);
}

int RLMTokenizer::recursive_token_id() const {
    return recursive_tokens.empty() ? -1 : recursive_tokens[0];
}

int RLMTokenizer::aggregate_token_id() const {
    return aggregate_tokens.empty() ? -1 : aggregate_tokens[0];
}

int RLMTokenizer::exit_token_id() const {
    return exit_tokens.empty() ? -1 : exit_tokens[0];
}

}
