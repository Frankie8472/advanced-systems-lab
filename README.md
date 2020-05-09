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

T >= 18 and congruent to 2 modulo 16 [T = 2 (mod 16)]

N >= 16 and divisible by 16

M >= 16 and divisible by 16

This is sufficiently large to take most to all optimization possibilities into account.

Furthermore, to check equality for doubles, we use EPSILON 1e-6, to not get caught up in numerical instabilities and other sources of randomness.

Lastly, we omitted the convergence criterion by the minimization of the monotonously decreasing negative log likelihood sequence, because it adds an unnecessary source of randomness.
Note that Expectation-Maximization is provably guaranteed to not change after convergence, so running more than fewer iterations causes no harm, except for overfitting (irrelevant for our purposes) and increased runtime (wanted for benchmarking).

## Verification

### Baseline

We have "test_case_ghmm_x" functions to check against hardcoded examples that were verified using the Python 2.x, Linux only ghmm library ( http://ghmm.sourceforge.net/index.html and download https://sourceforge.net/projects/ghmm/ ).

For reproducibility purposes, the code can be found in misc/asl_baum_welch_ghmm_experiments.ipynb (jupyter notebook) or, alternatively, misc/asl_baum_welch_ghmm_experiments.py.

Note that Python 3 does not work with ghmm and to install the library, there are quite some dependencies to take into account.

### Wikipedia Example

We talked about the Wikipedia example ( https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Example ) during the meeting.
The example uses a joint probability factorization into conditional probabilities and usage of the likelihood to compute one "iteration".
This has nothing to do with the actual Baum-Welch algorithm described on this very same page. It is confusing and misleading. And it cost me DAYS.

The example is still used as "test_case_ghmm_0" and "test_case_ghmm_1", though obviously compared against Baum-Welch implementations.
And as it can be seen, not only in the verifications, but also in misc/BaumWelchWikipediaExample.xlsx
and the Matlab implementation ( https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/ ) that corresponds to the Tutorial 
( https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf ) linked on the ASL website for Baum-Welch project:

Our approach is absolutely correct and thoroughly verified for the project's scope and purpose!

### Optimizations

1. Randomly initialize K, N, M, T, observations, init_prob, trans_prob and emit_prob, by generating random numbers and normalize where needed.
2. Run the baseline implementation for max_iterations.
3. Check whether the sequence of the negative log likelihood is monotonously decreasing in each iteration, which is guaranteed by the expectation-maximization algorithm and shows correctness of the (unscaled) Baum-Welch algorithm conceptually.
4. Check whether the rows of init_prob, trans_prob and emit_prob sum to 1.0 each, as they represent (learned) probability distributions, both before and after the run. 
5. For each optimization: Do the same as 2., 3. and 4. and additionally check the resulting probability tables of init_prob, trans_prob and emit_prob directly with the corresponding ones from the baseline implementation.

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


