# MicroGPT-RLM: A Native C++ GPT with Recursive Language Models
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

# Source files
set(SOURCES
    src/core/tensor.cpp
    src/core/autograd.cpp
    src/core/math_ops.cpp
    src/model/embedding.cpp
    src/model/attention.cpp
    src/model/transformer.cpp
    src/model/model.cpp
    src/training/tokenizer.cpp
    src/training/dataset.cpp
    src/training/trainer.cpp
    src/inference/sampler.cpp
    src/inference/generator.cpp
    src/utils/random.cpp
    src/utils/logger.cpp
)

set(HEADERS
    src/core/tensor.hpp
    src/core/autograd.hpp
    src/core/math_ops.hpp
    src/core/memory_pool.hpp
    src/model/embedding.hpp
    src/model/attention.hpp
    src/model/transformer.hpp
    src/model/model.hpp
    src/training/tokenizer.hpp
    src/training/dataset.hpp
    src/training/trainer.hpp
    src/inference/sampler.hpp
    src/inference/generator.hpp
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
