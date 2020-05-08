
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

## Assumptions

K >= 16 and divisible by 16
T >= 16 and divisible by 16
N >= 16 and divisible by 16
M >= 16 and divisible by 16

This is sufficiently large to take most to all optimization possibilities into account.

## Verification

We have some explicit test cases, which are marked as "test_case_x" that only check the baseline, as the assumptions above may (safely) be violated.

For the optimizations, we do the following
1. Randomly initialize K, N, M, T, observations, init_prob, trans_prob and emit_prob, by generating random numbers and normalize where needed.
2. Run the baseline implementation until convergence or max_iterations.
3. Check whether the sequence of the negative log likelihood is monotonously decreasing in each iteration, which is guaranteed by the expectation-maximization algorithm.
4. Check whether the rows of init_prob, trans_prob and emit_prob sum to 1.0 each, as they represent (learned) probability distributions, both before and after the run. 
5. For each optimization: Run until convergence or max_iterations and check the probability tables of init_prob, trans_prob and emit_prob against the corresponding ones from the baseline.

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

Implementation of the Baum-Welch algorithm with scaling taken into account for numerical stability.

Main references used
https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf

https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf

### "scalar_optimized.cpp" Implementation

TODO

### "vector_optimized.cpp" Implementation

TODO

### "combined_optimized.cpp" Implementation

TODO


