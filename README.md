
# Advanced Systems Lab (How to Write Fast Numerical Code) - Spring 2020
Semester Project: Baum-Welch algorithm

### Authors

Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi

ETH Computer Science MSc, Computer Science Department ETH Zurich

## Compiler

We use gcc 9.2.1.

## Goal

Starting with a baseline version, we implement various optimizations to significantly speed up the performance of the Baum-Welch algorithm.

## "baseline.h" Implementation

Should be thoroughly verified. There's a chance of probabilities degenerating to NaNs if any of the values
K, N, M or T is too small. This is due to how the model works and nothing to bother about.

For simplicity reasons and to be able to optimize as best as possible, we assume anyway that K, N, M and T each are at least 4 and divisible by 4, which solves the problem in most to all cases. It could only occur if the (random) initialization of the observational data is very unlucky.

## "sota.h" Implementation

TODO

## "scalar_optimized.h" Implementation

TODO

## "vector_optimized.h" Implementation

TODO

## "combined_optimized.h" Implementation

TODO


## How to compile Benchmarks
Go to top level directory, do
```
gcc -03 -o benchmarks benchmarks.cpp -lm
```
Additional flags should be added.


## How to run Benchmarks
```
./benchmarks 100
```
The maximal number of iterations (here 100) can be ajust.


## How to compile Verifications
Go to top level directory, do
```
gcc -O3 -o verifications verifications.cpp -lm
```

## How to run Verifications
```
./verifications
```
The test cases are hard coded to check the implementations.
