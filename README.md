# MicroGPT-RLM

A native C++ implementation of a GPT-like language model with Recursive Language Model (RLM) architecture, optimized for Raspberry Pi 5.

## Overview

This project implements:
- **MicroGPT**: A minimal GPT-2-like transformer from scratch (based on Andrej Karpathy's microgpt)
- **RLM Integration**: Recursive Language Model for iterative reasoning without increasing model size

## Architecture

### Standard GPT
```
input → transformer(1 pass) → output
```

### RLM-Enhanced
```
input → state₀
state₁ = transformer(state₀ + input)
state₂ = transformer(state₁ + input)
...
output = decode(state_n)
```

The model refines its internal representation through multiple recursive passes, trading compute time for intelligence - perfect for constrained hardware like Raspberry Pi.

## Features

- Native C++ with no external ML dependencies
- Custom autograd engine (like micrograd)
- ARM NEON SIMD optimization for Pi 5
- Mixed precision training (bf16)
- Gradient checkpointing for memory efficiency
- Early exit token for adaptive recursion depth
- Shared weights across recursion steps (RNN-like)
- Character-level tokenizer

## Current Status

- **Working**: Forward pass, loss computation, generation
- **Model size**: ~800K parameters (configurable)
- **Test**: Runs on Raspberry Pi 5

## Requirements

- Raspberry Pi 5 (8GB+ recommended)
- CMake 3.16+
- C++20 compiler
- OpenMP

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Run

```bash
./microgpt
```

## Training

### Quick Start (Demo)

The built-in demo trains on sample names:

```bash
./microgpt
```

### Training with Custom Data

1. **Prepare your dataset** (text file, one document per line):

```bash
echo -e "emma\nolivia\nava\nisabella\nsophia" > data/names.txt
```

2. **Modify main.cpp** to load your data:

```cpp
// Load your dataset
Dataset dataset;
dataset.load_file("data/names.txt");

// Train
Trainer trainer(&model, &tokenizer, batch_size=1, max_steps=10000);
trainer.train(dataset);
```

3. **Configure model** (in main.cpp):

```cpp
ModelConfig config;
config.vocab_size = tokenizer.size();
config.embed_dim = 128;      // Increase for larger model
config.num_layers = 6;       // More layers = more capacity
config.num_heads = 4;
config.max_seq_len = 32;     // Max sequence length
config.hidden_dim = 512;     // FFN hidden size (typically 4x embed_dim)
```

4. **Training parameters**:

```cpp
config.learning_rate = 0.001f;     // Typical: 1e-4 to 1e-3
config.batch_size = 1;              // Small for Pi 5 memory
config.gradient_accumulation = 32;  // Effective batch = 32
config.max_steps = 100000;          // More steps = better model
```

### Training Tips for Pi 5

- **Start small**: 128 embed_dim, 2 layers
- **Use gradient accumulation**: Small batch + accumulation = larger effective batch
- **Monitor memory**: Watch RAM usage, reduce batch if OOM
- **Active cooling**: Training heats up Pi 5 significantly
- **Use NVMe storage**: If available, for faster data loading
- **Be patient**: Training on Pi 5 is slow but works!

### Training Metrics to Watch

- **Loss**: Should decrease over time (starts ~3.0 for random)
- **Perplexity**: `exp(loss)` - lower is better
- **Generation quality**: Check output periodically

### Example Training Output

```
[INFO] 00:00:00 | Starting training...
[INFO] 00:00:10 | Step 0 | Loss: 2.985
[INFO] 00:00:20 | Step 100 | Loss: 2.752
[INFO] 00:00:30 | Step 200 | Loss: 2.431
[INFO] 00:00:40 | Step 300 | Loss: 1.982
...
[INFO] 00:05:00 | Training complete!
[INFO] Generated: 'emily'
```

## Model Configuration

| Model | Parameters | Layers | Heads | Embed Dim |
|-------|------------|--------|-------|------------|
| Micro | ~1M | 2-4 | 2 | 64-128 |
| Small | ~10M | 6 | 4 | 256 |
| Medium | ~50M | 8 | 8 | 512 |
| Large | ~100M | 12 | 12 | 768 |

## Training on Pi 5

With 8GB RAM and gradient checkpointing:
- 1M model: ~1-2 hours for 1M tokens
- 10M model: ~3-5 days for 1B tokens
- 50M model: ~2-3 weeks for 5B tokens

## References

- MicroGPT: https://karpathy.github.io/2026/02/12/microgpt/
- RLM Paper: https://arxiv.org/abs/2512.24601

## License

MIT License - see LICENSE file
