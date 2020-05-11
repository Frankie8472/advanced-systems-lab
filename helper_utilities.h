/*
    Helper Utilities
    Throw all useful functions that may or may not be used more than once in here

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#if !defined(__BW_HELPER_UTILITIES_H)
#define __BW_HELPER_UTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <random>

#include "common.h"

#define PRINT_PASSED(msg, ...) printf("\x1b[1;32mPASSED:\x1b[0m " msg "\n",  ##__VA_ARGS__)
#define PRINT_FAIL(msg, ...) printf("\x1b[1;31mFAIL:\x1b[0m " msg "\n",  ##__VA_ARGS__)
#define PRINT_VIOLATION(msg, num, ...) printf("\x1b[1;35m%zu VIOLATIONS:\x1b[0m " msg "\n", num,  ##__VA_ARGS__)

// initialization functions

void initialize_uar(const BWdata& bw);
void initialize_random(const BWdata& bw);

/**
 * TODO description
 *
 * returns: True if there was no error, false otherwise
 */
bool check_and_verify(const BWdata& bw);

void print_states(const BWdata& bw);

/**
 * Description
 * Useful for debugging
 * Only use for small values K, N, M, T
 * Prints all (!) contents of input BWdata& bw
 */
void print_BWdata(const BWdata& bw);

inline void print_BWdata_debug_helper(const BWdata& bw, const size_t iteration_variable, const char* message) {
    printf("\n\x1b[1;33m[i = %zu] %s\x1b[0m", iteration_variable, message);
    print_BWdata(bw);
}

/**
 * INPUT
 * two const BWdata& structs to compare against
 * OUTPUT
 * true if both structs contain the same below mentioned data up to some numerical EPSILON
 * DESCRIPTION
 * checks whether the following match up to an numerical EPSILON:
 * Initialization Probabilities, Transition Probabilities and Emission Probabilities
 * */
bool is_BWdata_equal_only_probabilities(const BWdata& bw1, const BWdata& bw2);

/**
 * INPUT
 * two const BWdata& structs to compare against
 * OUTPUT
 * true if both structs contain the same data up to some numerical EPSILON
 * DESCRIPTION
 * checks whether each single field of each variable and array match up to an EPSILON
 * */
bool is_BWdata_equal(const BWdata& bw1, const BWdata& bw2);


#endif /* __BW_HELPER_UTILITIES_H */