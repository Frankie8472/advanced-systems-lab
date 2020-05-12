cmake_minimum_required(VERSION 3.10)
project(ASL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -O3 -fno-tree-vectorize -mavx2 -mfma -march=x86-64")

add_executable(verification
    common.cpp
    helper_utilities.cpp
    verifications.cpp
    implementations/baseline.cpp
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_playground.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/vector_optimized.cpp
    implementations/vectoworized.cpp
    implementations/combined_optimized.cpp
)

add_executable(benchmarks
    common.cpp
    helper_utilities.cpp
    benchmarks.cpp
    implementations/baseline.cpp
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_playground.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/vector_optimized.cpp
    implementations/vectoworized.cpp
    implementations/combined_optimized.cpp
)
