#include "tokenizer.hpp"
#include "../utils/logger.hpp"
#include <fstream>
#include <algorithm>

namespace microgpt {

void Tokenizer::build(const std::vector<std::string>& docs) {
    std::set<char> unique_chars;
    
    for (const auto& doc : docs) {
        for (char c : doc) {
            unique_chars.insert(c);
        }
    }
    
    int id = 0;
    for (char c : unique_chars) {
        char_to_id[c] = id;
        id_to_char[id] = c;
        id++;
    }
    
    bos_id = id;
    id_to_char[bos_id] = '<BOS>';
    char_to_id['<BOS>'] = bos_id;
    id++;
    
    eos_id = id;
    id_to_char[eos_id] = '<EOS>';
    char_to_id['<EOS>'] = eos_id;
    id++;
    
    vocab_size = id;
    
    Logger::info("Tokenizer built: vocab_size=" + std::to_string(vocab_size));
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_id);
    
    for (char c : text) {
        auto it = char_to_id.find(c);
        if (it != char_to_id.end()) {
            tokens.push_back(it->second);
        }
    }
    
    tokens.push_back(eos_id);
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string text;
    
    for (int id : ids) {
        if (id == bos_id || id == eos_id) continue;
        
        auto it = id_to_char.find(id);
        if (it != id_to_char.end()) {
            text += it->second;
        }
    }
    
    return text;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream file(path);
    file << vocab_size << "\n";
    file << bos_id << "\n";
    file << eos_id << "\n";
    
    for (const auto& [c, id] : char_to_id) {
        if (c == '<BOS>' || c == '<EOS>') continue;
        file << c << " " << id << "\n";
    }
    
    file << "<BOS> " << bos_id << "\n";
    file << "<EOS> " << eos_id << "\n";
}

void Tokenizer::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        Logger::error("Failed to load tokenizer from " + path);
        return;
    }
    
    file >> vocab_size >> bos_id >> eos_id;
    
    std::string token;
    int id;
    while (file >> token >> id) {
        if (token == "<BOS>") {
            bos_id = id;
            id_to_char[bos_id] = '<BOS>';
        } else if (token == "<EOS>") {
            eos_id = id;
            id_to_char[eos_id] = '<EOS>';
        } else if (token.length() == 1) {
            char c = token[0];
            char_to_id[c] = id;
            id_to_char[id] = c;
        }
    }
}

}
