#pragma once
#include "tensor.hpp"
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace microgpt {

class TensorPool {
private:
    struct PoolKey {
        size_t size;
        size_t shape_hash;
        
        bool operator==(const PoolKey& other) const {
            return size == other.size && shape_hash == other.shape_hash;
        }
    };
    
    struct PoolKeyHash {
        size_t operator()(const PoolKey& k) const {
            return k.size ^ (k.shape_hash << 1);
        }
    };
    
    std::unordered_map<PoolKey, std::vector<Tensor>, PoolKeyHash> pools;
    std::mutex pool_mutex;
    
    size_t max_total_memory;
    size_t current_memory;
    
    static size_t compute_shape_hash(const Shape& shape) {
        size_t h = 0;
        for (int d : shape) {
            h = h * 31 + static_cast<size_t>(d);
        }
        return h;
    }
    
    static PoolKey make_key(const Shape& shape) {
        size_t total_size = 1;
        for (int d : shape) total_size *= d;
        return {total_size, compute_shape_hash(shape)};
    }

public:
    TensorPool(size_t max_memory = 512 * 1024 * 1024);
    ~TensorPool();
    
    Tensor get(const Shape& shape);
    Tensor get(size_t rows, size_t cols);
    
    void release(Tensor& t);
    void clear();
    void clear_unused();
    
    size_t get_total_allocated() const { return current_memory; }
    size_t get_max_memory() const { return max_total_memory; }
    
    void set_max_memory(size_t bytes);
    
    void preallocate(const Shape& shape, int count);
    void preallocate(size_t rows, size_t cols, int count);
    
    struct Stats {
        size_t pools_count;
        size_t total_tensors;
        size_t memory_usage;
    };
    
    Stats get_stats() const;

private:
    Tensor create_tensor(const Shape& shape);
    void trim_pool_if_needed();
};

class ScopedTensor {
private:
    TensorPool* pool;
    Tensor tensor;
    
public:
    ScopedTensor(TensorPool* p, const Shape& shape) 
        : pool(p), tensor(p->get(shape)) {}
    
    ScopedTensor(TensorPool* p, size_t rows, size_t cols) 
        : pool(p), tensor(p->get(rows, cols)) {}
    
    ~ScopedTensor() {
        if (pool && tensor.data.size() > 0) {
            pool->release(tensor);
        }
    }
    
    Tensor& get() { return tensor; }
    const Tensor& get() const { return tensor; }
    
    Tensor* operator->() { return &tensor; }
    const Tensor* operator->() const { return &tensor; }
    
    Tensor& operator*() { return tensor; }
    const Tensor& operator*() const { return tensor; }
    
    ScopedTensor(const ScopedTensor&) = delete;
    ScopedTensor& operator=(const ScopedTensor&) = delete;
    
    ScopedTensor(ScopedTensor&&) = default;
    ScopedTensor& operator=(ScopedTensor&&) = default;
};

class MemoryArena {
private:
    std::vector<char> buffer;
    size_t offset;
    size_t alignment;
    
public:
    MemoryArena(size_t size, size_t align = 32);
    ~MemoryArena();
    
    void* allocate(size_t size);
    void reset();
    size_t used() const { return offset; }
    size_t capacity() const { return buffer.size(); }
    
    template<typename T>
    T* allocate_array(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T)));
    }
};

}
