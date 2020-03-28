/*
    Verifications for the various Baum Welch algorithm implementations
    If you find other test cases, add them!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <time.h>
#include <unistd.h>
// custom files for the project
#include "helper_utilities.h"
// all versions
// only activate one at a time
// (may overwrite each other)
#include "baseline.h"
//#include "sota.h"
//#include "scalar_optimized.h"
//#include "vector_optimized.h"
//#include "combined_optimized.h"


void test_case_1(void);
void test_case_2(void);
void test_case_randomized(void);


int main(int argc, char **argv) {

    // randomize seed
    unsigned int tn = time(NULL);
    
    //printf("\nTest Case Custom 1 with srand(%d)\n", tn);
    //test_case_1();
    //printf("\nTest Case Custom 2 with srand(%d)\n", tn);
    //test_case_2();

    unsigned int iters = 10;
    for (int i = 0; i < iters; i++) {
        tn = time(NULL);
        srand(tn);
        printf("\nTest Case Randomized %d with srand(%d)\n", i, tn);
        test_case_randomized();
    }
    printf("\nAll Tests Done!\n\n");
}


void test_case_1(void) {

    const unsigned int K = 4;
    const unsigned int N = 4;
    const unsigned int M = 8;
    const unsigned int T = 4;
    const unsigned int max_iterations = 1000;

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* const observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* const init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* const trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* const emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);
    double* const neg_log_likelihoods = (double *)calloc(max_iterations, sizeof(double));
    if (neg_log_likelihoods == NULL) exit(1);

    initialize_random(K, N, M, T, observations, init_prob, trans_prob, emit_prob);
    printf("\nInitialized: K = %d, N = %d, M = %d, T = %d and max_iterations = %d", K, N, M, T, max_iterations);
    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, N, M, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    //print_states(N, M, T, init_prob, trans_prob, emit_prob);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(neg_log_likelihoods);
}


void test_case_2(void) {

    const unsigned int K = 4;
    const unsigned int N = 4;
    const unsigned int M = 4;
    const unsigned int T = 32;
    const unsigned int max_iterations = 1000;

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* const observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* const init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* const trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* const emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);
    double* const neg_log_likelihoods = (double *)calloc(max_iterations, sizeof(double));
    if (neg_log_likelihoods == NULL) exit(1);

    observations[0*T + 5] = 1;
    observations[0*T + 6] = 1;
    observations[1*T + 0] = 1;
    observations[1*T + 4] = 1;
    observations[1*T + 7] = 1;
    observations[2*T + 0] = 1;

    init_prob[0] = 0.25;
    init_prob[1] = 0.25;
    init_prob[2] = 0.25;
    init_prob[3] = 0.25;

    trans_prob[0*N + 0] = 0.5;
    trans_prob[0*N + 1] = 0.5;
    trans_prob[1*N + 0] = 0.3;
    trans_prob[1*N + 1] = 0.7;
    trans_prob[2*N + 0] = 0.5;
    trans_prob[2*N + 1] = 0.5;
    trans_prob[3*N + 0] = 0.3;
    trans_prob[3*N + 1] = 0.7;

    emit_prob[0*M + 0] = 0.3;
    emit_prob[0*M + 1] = 0.7;
    emit_prob[1*M + 0] = 0.8;
    emit_prob[1*M + 1] = 0.2;
    emit_prob[2*M + 0] = 0.3;
    emit_prob[2*M + 1] = 0.7;
    emit_prob[3*M + 0] = 0.8;
    emit_prob[3*M + 1] = 0.2;

    printf("\nInitialized: K = %d, N = %d, M = %d, T = %d and max_iterations = %d", K, N, M, T, max_iterations);
    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, N, M, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    print_states(N, M, T, init_prob, trans_prob, emit_prob);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(neg_log_likelihoods);
}


void test_case_randomized(void) {

    const unsigned int K = (rand() % 8)*4 + 4;
    const unsigned int N = (rand() % 9)*4 + 4;
    const unsigned int M = (rand() % 9)*4 + 4;
    const unsigned int T = (rand() % 14)*4 + 4;
    const unsigned int max_iterations = 500;

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* const observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* const init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* const trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* const emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);
    double* const neg_log_likelihoods = (double *)calloc(max_iterations, sizeof(double));
    if (neg_log_likelihoods == NULL) exit(1);

    initialize_random(K, N, M, T, observations, init_prob, trans_prob, emit_prob);
    printf("\nInitialized: K = %d, N = %d, M = %d, T = %d and max_iterations = %d", K, N, M, T, max_iterations);
    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, N, M, init_prob, trans_prob, emit_prob, neg_log_likelihoods);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(neg_log_likelihoods);
}