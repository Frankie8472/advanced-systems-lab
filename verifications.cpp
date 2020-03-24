
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
double* init_prob; //           [N]       [i]       := P(X_1 = i)
double* trans_prob; //          [N][N]    [i][j]    := P(X_t = j | X_(t-1) = i) 
double* emit_prob; //           [M][N]    [i][j]    := P(Y_t = y_i | X_t = j)
double* alpha; //               [T][N]    [t][i]    := P(Y_1 = y_1, ..., Y_t = y_t, X_t = i)
double* beta; //                [T][N]    [t][i]    := P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = i)
double* ggamma; //              [T][N]    [t][i]    := P(X_t = i | Y)
double* sigma; //               [T][N][N] [t][i][j] := P(X_t = i, X_(t+1) = j | Y)

void test_case_1(void);
void test_case_2(void);
void test_case_3(void);
void test_case_4(void);
void test_case_5(void);

void print_states(unsigned int N, unsigned int M, unsigned int T) {
    printf("\nInitialization probabilities:\n");
    for(int i = 0; i < N; i++) {
        printf("state %d  %f\n", i+1, init_prob[i]);
    }

    printf("\nTransition probabilities:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("state %d -> state %d  %f\n", i+1, j+1, trans_prob[i*N + j]);
        }
    }

    printf("\nEmission probabilities:\n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("state %d : %d  %f\n", i+1, j, emit_prob[i*N + j]);
        }
    }
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

    observations = (unsigned int *)malloc(T*sizeof(unsigned int));
    init_prob = (double *)malloc(N*sizeof(double));
    trans_prob = (double *)malloc(N*N*sizeof(double));
    emit_prob = (double *)malloc(M*N*sizeof(double));
    alpha = (double *)malloc(T*N*sizeof(double));
    beta = (double *)malloc(T*N*sizeof(double));
    ggamma = (double *)malloc(T*N*sizeof(double));
    sigma = (double *)malloc(T*N*N*sizeof(double));

    init_prob[0] = 0.2;
    init_prob[1] = 0.8;

    // right now; sums up to 1 over columns
    // CHANGE TO ROWS!
    trans_prob[0*N + 0] = 0.5;
    trans_prob[0*N + 1] = 0.3;
    trans_prob[1*N + 0] = 0.5;
    trans_prob[1*N + 1] = 0.7;

    // right now; sums up to 1 over columns
    // CHANGE TO ROWS!
    emit_prob[0*N + 0] = 0.3;
    emit_prob[0*N + 1] = 0.8;
    emit_prob[1*N + 0] = 0.7;
    emit_prob[1*N + 1] = 0.2;

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

    for (int t = 0; t < T; t++) {
        for (int j = 0; j < N; j++) {
            alpha[t*N + j] = 0;
            beta[t*N + j] = 0;
            ggamma[t*N + j] = 0;
            for (int k = 0; k < N; k++) {
                sigma[(t*N + j)*N + k] = 0;
            }
        }
    }

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
