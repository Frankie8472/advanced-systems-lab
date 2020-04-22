cmake_minimum_required(VERSION 3.10)
project(ASL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

add_executable(verification
    common.cpp
    verifications.cpp
    implementations/baseline.cpp
)

add_executable(benchmarks
    common.cpp
    benchmarks.cpp
    implementations/baseline.cpp
)