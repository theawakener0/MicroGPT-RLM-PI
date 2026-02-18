# MicroGPT-RLM-PI

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
state₁ = transformer(state₀) + state₀  (residual)
state₂ = transformer(state₁) + state₁
...
output = lm_head(ln_f(state_n))
```

The model refines its internal representation through multiple recursive passes with residual connections, trading compute time for intelligence - perfect for constrained hardware like Raspberry Pi.

## Features

- Native C++ with no external ML dependencies
- **Complete autograd engine** - Full backward propagation implementation
- **ARM NEON SIMD optimization** for Raspberry Pi 5
- **Advanced inference sampling** - Temperature, top-k, top-p, repetition penalty
- Character-level tokenizer
- Chat-style inference mode
- **RLM recursive forward** with early exit capability

## Current Status

| Feature | Status |
|---------|--------|
| Forward pass | ✅ Working |
| Backpropagation | ✅ Complete |
| Weight updates (Adam) | ✅ Implemented |
| Loss computation | ✅ Working |
| Text generation | ✅ Working |
| Chat mode | ✅ Working |
| RLM recursive forward | ✅ Working |
| NEON optimizations | ✅ Working |
| Sampler (temp/top-k/top-p) | ✅ Working |
| Model save/load | ✅ Working |

### Model Specifications

- **Parameters**: ~800K (default config)
- **Vocab Size**: 256 (expandable)
- **Sequence Length**: 256 (configurable)
- **Tested on**: Raspberry Pi 5

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

## Quick Start

```bash
# 1. Download training data
./scripts/download_data.sh

# 2. Train on names
./microgpt --train --data data/names.txt --steps 50000

# 3. Generate
./microgpt
```

## Training Data

### Included Datasets

| File | Size | Lines | Description |
|------|------|-------|-------------|
| names.txt | 223KB | 32K | Baby names |
| pride_prejudice.txt | 718KB | 11K | Romance/dialogue |
| frankenstein.txt | 411KB | 6K | Gothic/horror |
| moby_dick.txt | 1.2MB | 19K | Adventure |
| sherlock_holmes.txt | 569KB | 9K | Mystery/dialogue |
| alice_wonderland.txt | 147KB | 2K | Children's |
| wikitext.txt | **11MB** | 37K | Wikipedia articles |
| code.txt | 4KB | ~50 | Code examples |
| training_data.txt | **14MB** | 85K | Combined corpus |

### Download Script

```bash
./scripts/download_data.sh
```

This downloads and cleans:
- 6 classic books from Project Gutenberg (cleaned)
- WikiText-2 (Wikipedia articles)
- 32K baby names
- Code samples

**All data cleaned** - Gutenberg headers/footers removed automatically.

## Training

### Complete Workflow

#### 1. Demo (Quick Test)
```bash
./microgpt
```

#### 2. Train a Model
```bash
# Train on names
./microgpt --train --data data/names.txt --steps 10000 --save_checkpoint model.bin

# Train on literature
./microgpt --train --data data/training_data.txt --steps 50000 --save_checkpoint model.bin

# Continue training from checkpoint
./microgpt --train --data data/names.txt --steps 10000 --load_checkpoint model.bin --save_checkpoint model_new.bin
```

#### 3. Generate Text
```bash
# Generate with trained checkpoint
./microgpt --generate --checkpoint model.bin --prompt "Once upon"

# Generate with random weights (for testing)
./microgpt --generate --prompt "Hello"
```

#### 4. Chat Mode
```bash
# Start interactive chat
./microgpt --chat --checkpoint model.bin

# Chat with random weights (for testing)
./microgpt --chat
```

### Training Options

| Option | Description | Example |
|--------|-------------|---------|
| `--train` | Training mode | |
| `--data <path>` | Training data file | `--data data/names.txt` |
| `--steps <n>` | Number of steps | `--steps 50000` |
| `--save_checkpoint <path>` | Save model to file | `--save_checkpoint model.bin` |
| `--load_checkpoint <path>` | Load model from file | `--load_checkpoint model.bin` |
| `--generate` | Text generation mode | |
| `--prompt <text>` | Prompt for generation | `--prompt "Hello"` |
| `--chat` | Interactive chat mode | |
| `--checkpoint <path>` | Shorthand for load | `--checkpoint model.bin` |

The checkpoint file contains:
- Model configuration (vocab size, layers, etc.)
- All model weights (embeddings, attention, MLP layers)

### What to Expect

#### Current Capabilities
- **Forward pass** - Model processes tokens correctly
- **Loss computation** - Cross-entropy loss calculated
- **Backpropagation** - Full gradient computation through all layers
- **Weight updates** - Adam optimizer integration
- **Generation** - Produces text token by token with advanced sampling
- **Save/Load** - Checkpoint saving and loading works
- **RLM recursive forward** - Multiple refinement passes with early exit

#### After Full Training
| Dataset | What Model Learns |
|---------|------------------|
| names.txt | Generate new names like "emily", "olivia" |
| training_data.txt | English text patterns |
| code.txt | Python-like syntax |

### Training Tips for Pi 5

| Tip | Description |
|-----|-------------|
| **Start small** | 128 embed_dim, 2 layers |
| **Gradient accumulation** | Small batch + accumulation = larger effective batch |
| **Monitor memory** | Watch RAM usage, reduce batch if OOM |
| **Active cooling** | Training heats up Pi 5 significantly |
| **Be patient** | Training on Pi 5 is slow but works! |

### Recommended Configurations

**For 8GB Pi 5:**
```cpp
config.embed_dim = 128;
config.num_layers = 4;
config.num_heads = 4;
config.max_seq_len = 64;
config.hidden_dim = 512;
```

**For 16GB Pi 5:**
```cpp
config.embed_dim = 256;
config.num_layers = 6;
config.num_heads = 8;
config.max_seq_len = 128;
config.hidden_dim = 1024;
```

## RLM Configuration

The Recursive Language Model allows multiple refinement passes:

```cpp
config.recursion_steps = 3;       // Number of recursive passes
config.exit_threshold = 0.8f;      // Early exit probability threshold
config.min_recursion_steps = 1;   // Minimum passes before early exit
```

### How RLM Works

1. Input embeddings are computed once
2. For each recursion step:
   - Apply layer normalization
   - Pass through transformer layers
   - Add residual connection: `state = state + new_state`
   - Check exit head for early termination
3. Final layer norm and lm_head produce logits

## Inference Sampling

The sampler supports multiple generation strategies:

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `temperature` | Controls randomness (1.0 = default, lower = more deterministic) | 0.7-1.0 |
| `top_k` | Limits to k most likely tokens (0 = disabled) | 40 |
| `top_p` | Nucleus sampling threshold (0.0 = disabled) | 0.9 |
| `repetition_penalty` | Penalizes repeated tokens (1.0 = no penalty) | 1.1-1.5 |

## Performance

### Forward Pass Complexity
```
O(L × S² × D) where:
  L = number of layers
  S = sequence length
  D = embedding dimension
```

### Estimated Performance on Raspberry Pi 5

| Operation | Time (approx) |
|-----------|---------------|
| Forward pass (32 tokens, 4 layers) | 50-100ms |
| Token generation | 10-20ms/token |
| Training step (small batch) | 200-500ms |

### NEON Optimizations

The following operations use ARM NEON SIMD:
- Matrix multiplication (fused multiply-add)
- Softmax (vectorized exp/divide)
- Element-wise operations (add, multiply, relu)

## Model Configuration

| Model | Parameters | Layers | Heads | Embed Dim |
|-------|------------|--------|-------|------------|
| Micro | ~1M | 2-4 | 2 | 64-128 |
| Small | ~10M | 6 | 4 | 256 |
| Medium | ~50M | 8 | 8 | 512 |

## Project Structure

```
src/
├── core/
│   ├── tensor.hpp/cpp      # Tensor data structure
│   ├── math_ops.hpp/cpp    # Math operations (NEON optimized)
│   └── autograd.hpp/cpp   # Autograd engine
├── model/
│   ├── model.hpp/cpp      # GPT model
│   ├── transformer.hpp/cpp # Transformer blocks
│   ├── attention.hpp/cpp  # Multi-head attention
│   └── embedding.hpp/cpp  # Token/positional embeddings
├── training/
│   ├── trainable_model.hpp/cpp # Trainable version with backprop
│   ├── transformer.hpp/cpp # Trainable transformer layers
│   ├── nn.hpp/cpp        # Neural network modules
│   ├── tokenizer.hpp/cpp  # Character-level tokenizer
│   ├── trainer.hpp/cpp    # Training loop
│   └── dataset.hpp/cpp    # Data loading
├── inference/
│   ├── sampler.hpp/cpp    # Sampling strategies
│   └── generator.hpp/cpp  # Text generation
└── utils/
    ├── logger.hpp/cpp     # Logging utilities
    └── random.hpp/cpp     # Random number generation
```

## References

- MicroGPT: https://karpathy.github.io/2026/02/12/microgpt/
- RLM Paper: https://arxiv.org/abs/2512.24601

## License

MIT License - see [LICENSE](LICENSE) file
