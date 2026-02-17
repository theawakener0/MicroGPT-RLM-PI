#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

namespace microgpt {

class Tokenizer {
public:
    int vocab_size = 0;
    int bos_id = 0;
    int eos_id = 0;
    
    std::unordered_map<char, int> char_to_id;
    std::unordered_map<int, char> id_to_char;
    
    Tokenizer() = default;
    
    void build(const std::vector<std::string>& docs);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    int size() const { return vocab_size; }
    void save(const std::string& path) const;
    void load(const std::string& path);
};

}
