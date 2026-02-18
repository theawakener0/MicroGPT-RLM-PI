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
- **Generation** - Produces text token by token
- **Save/Load** - Checkpoint saving and loading works
- **Training loop** - Iterates over data (forward-only, no weight updates yet)

#### Limitations
- **No backprop** - Forward-only, model stays random without weight updates
- **Basic model** - ~800K parameters
- **Character-level** - No BPE tokenizer

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
