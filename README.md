
# Advanced Systems Lab (How to Write Fast Numerical Code) - Spring 2020
Semester Project: Baum-Welch algorithm

### Authors

Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi

ETH Computer Science MSc, Computer Science Department ETH Zurich

## Goal

Starting with a baseline version, we implement various optimizations to significantly speed up the performance of the Baum-Welch algorithm.


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
Additional flags should be added.


## How to run Verifications
```
./verifications
```
The test cases are hard coded to check the implementations.
