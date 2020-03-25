
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

unsigned int* observations; //  [T]       [t]       := observationservation of time_step t
double* init_prob; //           [N]       [n]       := P(X_1 = n)
double* trans_prob; //          [N][N]    [n0][n1]  := P(X_t = n1 | X_(t-1) = n0)
double* emit_prob; //           [N][M]    [n][m]    := P(Y_t = y_m | X_t = n)
double* alpha; //               [T][N]    [t][n]    := P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
double* beta; //                [T][N]    [t][n]    := P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
double* ggamma; //              [T][N]    [t][n]    := P(X_t = n | Y, theta)
double* sigma; //               [T][N][N] [t][n0][n1] := P(X_t = n0, X_(t+1) = n1 | Y, theta)
// where theta = {init_prob, trans_rpob, emit_prob} represent the model parameters we want to learn
// (given some initial configuration)

void test_case_1(void);
void test_case_2(void);
void test_case_3(void);
void test_case_4(void);
void test_case_5(void);

void print_states(unsigned int N, unsigned int M, unsigned int T) {
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

    unsigned int N = 2;
    unsigned int M = 2;
    unsigned int T = 10;
    unsigned int max_iterations = 1;

    // calloc initializes to 0.0
    observations = (unsigned int *)calloc(T, sizeof(unsigned int));
    if (observations == NULL) exit (1);
    init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit (1);
    trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit (1);
    emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit (1);
    alpha = (double *)calloc(T*N, sizeof(double));
    if (alpha == NULL) exit (1);
    beta = (double *)calloc(T*N, sizeof(double));
    if (beta == NULL) exit (1);
    ggamma = (double *)calloc(T*N, sizeof(double));
    if (ggamma == NULL) exit (1);
    sigma = (double *)calloc(T*N*N, sizeof(double));
    if (sigma == NULL) exit (1);

    init_prob[0] = 0.2;
    init_prob[1] = 0.8;

    trans_prob[0*N + 0] = 0.5;
    trans_prob[0*N + 1] = 0.5;
    trans_prob[1*N + 0] = 0.3;
    trans_prob[1*N + 1] = 0.7;

    emit_prob[0*M + 0] = 0.3;
    emit_prob[0*M + 1] = 0.7;
    emit_prob[1*M + 0] = 0.8;
    emit_prob[1*M + 1] = 0.2;

    // set of observationservations (see link)
    // 0 is "no egg"
    // 1 is "egg"
    observations[0] = 0;
    observations[1] = 0;
    observations[2] = 0;
    observations[3] = 0;
    observations[4] = 0;
    observations[5] = 1;
    observations[6] = 1;
    observations[7] = 0;
    observations[8] = 0;
    observations[9] = 0;

    compute_baum_welch(max_iterations, N, M, T, observations, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);

    print_states(N, M, T);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(alpha);
    free(beta);
    free(ggamma);
    free(sigma);
}
