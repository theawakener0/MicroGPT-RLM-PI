# MicroGPT-RLM-PI: A Native C++ GPT with Recursive Language Models
# Target: Raspberry Pi 5 (8GB RAM)
# Architecture: Transformer with RLM iterative refinement

cmake_minimum_required(VERSION 3.16)
project(MicroGPT_RLM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a76")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Find OpenMP for multi-threading
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Source files (training/ has full implementation with backprop)
set(SOURCES
    src/core/tensor.cpp
    src/core/autograd.cpp
    src/core/math_ops.cpp
    src/core/memory_pool.cpp
    src/training/tokenizer.cpp
    src/training/dataset.cpp
    src/training/trainer.cpp
    src/training/trainable_model.cpp
    src/training/transformer.cpp
    src/training/nn.cpp
    src/inference/sampler.cpp
    src/inference/generator.cpp
    src/rlm/rlm.cpp
    src/utils/random.cpp
    src/utils/logger.cpp
)

set(HEADERS
    src/core/tensor.hpp
    src/core/autograd.hpp
    src/core/math_ops.hpp
    src/core/memory_pool.hpp
    src/training/tokenizer.hpp
    src/training/dataset.hpp
    src/training/trainer.hpp
    src/training/trainable_model.hpp
    src/training/transformer.hpp
    src/training/nn.hpp
    src/inference/sampler.hpp
    src/inference/generator.hpp
    src/rlm/rlm.hpp
    src/utils/random.hpp
    src/utils/logger.hpp
    src/utils/timer.hpp
)

# Create executable
add_executable(microgpt ${SOURCES} ${HEADERS} main.cpp)

# Link libraries
target_link_libraries(microgpt OpenMP::OpenMP_CXX)

# Installation
install(TARGETS microgpt DESTINATION bin)
