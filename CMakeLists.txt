cmake_minimum_required(VERSION 3.10)
project(ASL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -O3 -fno-tree-vectorize -ffast-math -mavx2 -mfma -march=native")

SET_SOURCE_FILES_PROPERTIES(implementations/vector_optimized.cpp PROPERTIES COMPILE_FLAGS "-fno-tree-vectorize")
SET_SOURCE_FILES_PROPERTIES(helper_utilities.cpp PROPERTIES COMPILE_FLAGS "-fno-fast-math -O0")
SET_SOURCE_FILES_PROPERTIES(verifications.cpp PROPERTIES COMPILE_FLAGS "-fno-fast-math -O0")
SET_SOURCE_FILES_PROPERTIES(benchmarks.cpp PROPERTIES COMPILE_FLAGS "-fno-fast-math -O0")

add_executable(verification
    common.cpp
    helper_utilities.cpp
    verifications.cpp
    implementations/baseline.cpp
#    implementations/baseline_transposed_emit_prob.cpp # Test transpose of emit_prob
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_playground.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/vector_optimized.cpp
    implementations/combined_optimized.cpp
    implementations/unrolled_emit_prob.cpp
    implementations/unrolled_optimized.cpp
)

add_executable(benchmarks
    common.cpp
    helper_utilities.cpp
    benchmarks.cpp
    implementations/baseline.cpp
#    implementations/baseline_transposed_emit_prob.cpp # Test transpose of emit_prob
    implementations/reordered_algorithm.cpp
    implementations/scalar_optimized_playground.cpp
    implementations/scalar_optimized_reorder.cpp
    implementations/scalar_optimized_blocking.cpp
    implementations/vector_optimized.cpp
    implementations/combined_optimized.cpp
    implementations/unrolled_emit_prob.cpp
    implementations/unrolled_optimized.cpp
)
