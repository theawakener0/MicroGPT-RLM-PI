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
- **Test**: Runs on Raspberry Pi 5 ✅

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
