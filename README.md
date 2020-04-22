
# Advanced Systems Lab (How to Write Fast Numerical Code) - Spring 2020
Semester Project: Baum-Welch algorithm

### Authors

Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi

ETH Computer Science MSc, Computer Science Department ETH Zurich

## Overleaf

https://www.overleaf.com/read/fsqqnxhfhnvx

View-only link for supervisors to observe progress on the report.

## Compiler

We use gcc 9.2.1.

## Building the project

The project uses CMake. 

Create a folder named `build` and change into it. Then run `cmake ..` to generate the Makefile and then `make` to build the project. 

The project generates two executables: `benchmarks` nad `verifications`. 

## Running the project

`benchmarks` executes the performance benchmark test without verifications if the implementations are correct. It requires a number as an argument. This is the number of maximum iterations the algorithm is allowed to execute. 

`verification` checks if the implementations behave correctly.

## Goal

Starting with a baseline version, we implement various optimizations to significantly speed up the performance of the Baum-Welch algorithm.

## Implementations

All implementations are found in the `implementations` folder. To create a new implementation follow those steps:

1. Create a new `.cpp` file in the folder `implementations`
2. Add the file to both executables in the `CMakeLists.txt`
3. Implement

Your implementation must have a function with the following signature to allow it to be called by the benchmark and verification system.

```C
size_t func_name(const BWdata& bw){...}
```

To register your function in the benchmark system add the following line to your file. You can register multiple functions in one file.

```C
REGISTER_FUNCTION(func_name, "A description about the implementation");
```

**CAUTION**: Be aware that you cannot name your function the same as another implementation in a different file. The linker is not able to do that right now.


### "baseline.cpp" Implementation

Should be thoroughly verified. There's a chance of probabilities degenerating to NaNs if any of the values K, N, M or T are too small. This is due to how the model works and nothing to bother about.

For simplicity reasons and to be able to optimize as best as possible, we assume anyway that K, N, M and T each are at least 4 and divisible by 4, which solves the problem in most to all cases. It could only occur if the (random) initialization of the observational data is very unlucky.

### "sota.cpp" Implementation

TODO

### "scalar_optimized.cpp" Implementation

TODO

### "vector_optimized.cpp" Implementation

TODO

### "combined_optimized.cpp" Implementation

TODO


