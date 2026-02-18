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
- Character-level tokenizer
- Chat-style inference mode

## Current Status

- **Working**: Forward pass, loss computation, generation, chat mode
- **Model size**: ~800K parameters (configurable)
- **Test**: Runs on Raspberry Pi 5
- **Data**: ~14MB cleaned corpus included

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

### Training Commands

```bash
# Train on names (learns to generate names)
./microgpt --train --data data/names.txt --steps 50000

# Train on literature (learns English text)
./microgpt --train --data data/training_data.txt --steps 100000

# Train on code (learns programming)
./microgpt --train --data data/code.txt --steps 50000
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | - | Training mode |
| `--data` | required | Path to training data |
| `--steps` | 1000 | Number of training steps |
| `--chat` | - | Interactive chat mode |

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

## Chat Mode

```bash
./microgpt --chat
```

Example:
```
> Hello
Hello! How are you today?

> Tell me a story
Once upon a time in a land far away...
```

## Model Configuration

| Model | Parameters | Layers | Heads | Embed Dim |
|-------|------------|--------|-------|------------|
| Micro | ~1M | 2-4 | 2 | 64-128 |
| Small | ~10M | 6 | 4 | 256 |
| Medium | ~50M | 8 | 8 | 512 |

## References

- MicroGPT: https://karpathy.github.io/2026/02/12/microgpt/
- RLM Paper: https://arxiv.org/abs/2512.24601

## License

MIT License - see [LICENSE](LICENSE) file
