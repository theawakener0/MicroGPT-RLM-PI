#include "memory_pool.hpp"
#include <algorithm>
#include <cstring>

namespace microgpt {

TensorPool::TensorPool(size_t max_memory)
    : max_total_memory(max_memory), current_memory(0) {}

TensorPool::~TensorPool() {
    clear();
}

Tensor TensorPool::get(const Shape& shape) {
    PoolKey key = make_key(shape);
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    auto it = pools.find(key);
    if (it != pools.end() && !it->second.empty()) {
        Tensor t = std::move(it->second.back());
        it->second.pop_back();
        current_memory -= key.size * sizeof(float);
        t.data.resize(key.size);
        return t;
    }
    
    return create_tensor(shape);
}

Tensor TensorPool::get(size_t rows, size_t cols) {
    return get(Shape{static_cast<int>(rows), static_cast<int>(cols)});
}

Tensor TensorPool::create_tensor(const Shape& shape) {
    Tensor t(shape, false);
    return t;
}

void TensorPool::release(Tensor& t) {
    if (t.data.empty()) return;
    
    PoolKey key = make_key(t.shape);
    size_t tensor_size = key.size * sizeof(float);
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (current_memory + tensor_size > max_total_memory) {
        trim_pool_if_needed();
    }
    
    auto& pool = pools[key];
    t.data.clear();
    t.data.shrink_to_fit();
    pool.push_back(std::move(t));
    current_memory += tensor_size;
}

void TensorPool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    pools.clear();
    current_memory = 0;
}

void TensorPool::clear_unused() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    for (auto& kv : pools) {
        kv.second.clear();
    }
    current_memory = 0;
}

void TensorPool::trim_pool_if_needed() {
    size_t target_memory = max_total_memory / 2;
    
    for (auto& kv : pools) {
        if (current_memory <= target_memory) break;
        
        auto& pool = kv.second;
        while (!pool.empty() && current_memory > target_memory) {
            pool.pop_back();
            current_memory -= kv.first.size * sizeof(float);
        }
    }
}

void TensorPool::set_max_memory(size_t bytes) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    max_total_memory = bytes;
    trim_pool_if_needed();
}

void TensorPool::preallocate(const Shape& shape, int count) {
    PoolKey key = make_key(shape);
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    auto& pool = pools[key];
    for (int i = 0; i < count; i++) {
        pool.push_back(create_tensor(shape));
    }
    current_memory += key.size * sizeof(float) * count;
}

void TensorPool::preallocate(size_t rows, size_t cols, int count) {
    preallocate(Shape{static_cast<int>(rows), static_cast<int>(cols)}, count);
}

TensorPool::Stats TensorPool::get_stats() const {
    Stats s{};
    s.pools_count = pools.size();
    s.total_tensors = 0;
    s.memory_usage = current_memory;
    
    for (const auto& kv : pools) {
        s.total_tensors += kv.second.size();
    }
    
    return s;
}

MemoryArena::MemoryArena(size_t size, size_t align)
    : buffer(size), offset(0), alignment(align) {}

MemoryArena::~MemoryArena() {}

void* MemoryArena::allocate(size_t size) {
    size_t aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
    
    if (aligned_offset + size > buffer.size()) {
        return nullptr;
    }
    
    void* ptr = &buffer[aligned_offset];
    offset = aligned_offset + size;
    
    return ptr;
}

void MemoryArena::reset() {
    offset = 0;
}

}
