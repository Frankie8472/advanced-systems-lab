
// Only hard-coded tests here to check the various implementations for correctness!

#include <stdlib.h>
#include <stdio.h>

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
void test_case_3(void);
void test_case_4(void);
void test_case_5(void);

void print_states(unsigned int N, unsigned int M, unsigned int T,
    double* init_prob, double* trans_prob, double* emit_prob) {

    printf("\n");
    printf("\nInitialization probabilities:\n");
    for(int n = 0; n < N; n++) {
        printf("Pr[X_1 = %d] = %f\n", n+1, init_prob[n]);
    }

    printf("\nTransition probabilities:\n");
    for(int n0 = 0; n0 < N; n0++) {
        for(int n1 = 0; n1 < N; n1++) {
            printf("Pr[X_t = %d | X_(t-1) = %d ] = %f\n", n1+1, n0+1, trans_prob[n0*N + n1]);
        }
    }

    printf("\nEmission probabilities:\n");
    for(int n = 0; n < N; n++) {
        for(int m = 0; m < M; m++) {
            printf("Pr[Y_t = %d | X_t = %d] = %f\n", m+1, n+1, emit_prob[n*M + m]);
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {
    test_case_1();
}

// https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Example
void test_case_1(void) {

    unsigned int K = 4;
    unsigned int N = 4;
    unsigned int M = 4;
    unsigned int T = 32;
    unsigned int max_iterations = 1000;

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

    compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob);

    print_states(N, M, T, init_prob, trans_prob, emit_prob);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
}
