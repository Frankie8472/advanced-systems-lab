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


void test_case_1(void) {

    unsigned int K = 4;
    unsigned int N = 4;
    unsigned int M = 4;
    unsigned int T = 4;
    unsigned int max_iterations = 10000;
    double neg_log_likelihoods[max_iterations];

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);

    initialize_random(K, N, M, T, observations, init_prob, trans_prob, emit_prob);
    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, K, N, M, T, init_prob, trans_prob, emit_prob, neg_log_likelihoods);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
}


void test_case_2(void) {

    unsigned int K = 8;
    unsigned int N = 4;
    unsigned int M = 16;
    unsigned int T = 32;
    unsigned int max_iterations = 100;
    double neg_log_likelihoods[max_iterations];

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);

    initialize_random(K, N, M, T, observations, init_prob, trans_prob, emit_prob);
    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, K, N, M, T, init_prob, trans_prob, emit_prob, neg_log_likelihoods);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
}


void test_case_3(void) {

    unsigned int K = 16;
    unsigned int N = 64;
    unsigned int M = 32;
    unsigned int T = 32;
    unsigned int max_iterations = 100;
    double neg_log_likelihoods[max_iterations];

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);

    initialize_random(K, N, M, T, observations, init_prob, trans_prob, emit_prob);
    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, K, N, M, T, init_prob, trans_prob, emit_prob, neg_log_likelihoods);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
}


void test_case_4(void) {

    unsigned int K = 4;
    unsigned int N = 4;
    unsigned int M = 4;
    unsigned int T = 32;
    unsigned int max_iterations = 1000;
    double neg_log_likelihoods[max_iterations];

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit(1);
    double* init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit(1);
    double* trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit(1);
    double* emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit(1);

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

    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    check_and_verify(max_iterations, K, N, M, T, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    print_states(N, M, T, init_prob, trans_prob, emit_prob);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
}


int main(int argc, char **argv) {
    srand(42);
    printf("\nTest Case 1\n");
    test_case_1();
    printf("\nTest Case 2\n");
    test_case_2();
    printf("\nTest Case 3\n");
    test_case_3();
    printf("\nTest Case 4\n");
    test_case_4();
    printf("All Tests Done!\n\n");
}
