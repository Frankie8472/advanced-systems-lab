
// Structure code such that it produces output results we can copy-paste into latex plots (minimize overhead)

#ifndef WIN32
    #include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tsc_x86.h"

// all versions (may overwrite each other)
#include "baseline.h"
//#include "sota.h"
//#include "scalar_optimized.h"
//#include "vector_optimized.h"
//#include "combined_optimized.h"

#define NUM_RUNS 1
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 2.2e9
#define CALIBRATE 1

unsigned int N = 3; // number of hidden state variables
unsigned int M = 3; // number of observations
unsigned int T = 3; // number of time steps

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


void init() {

    // uniform
    for (int n0 = 0; n0 < N; n0++) {
        init_prob[n0] = 1.0/N;
        for (int n1 = 0; n1 < N; n1++) {
            trans_prob[n0*N + n1] = 1.0/N;
        }
    }

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            emit_prob[n*M + m] = 1.f/M;
        }
    }

    // fixed observationservation
    for (int t = 0; t < T; t++) {
        observations[t] = t % 2;
    }

    return;
}


double rdtsc(uint max_iterations) {

    int i, num_runs;
    myInt64 cycles;
    myInt64 start;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */

#ifdef CALIBRATE
    while (num_runs < (1 << 14)) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            compute_baum_welch(max_iterations, N, M, T, observations, init_prob,
                    trans_prob, emit_prob, alpha, beta, ggamma, sigma);
        }
        cycles = stop_tsc(start);
        if ( cycles >= CYCLES_REQUIRED ) break;
        num_runs *= 2;
    }
#endif
    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
        compute_baum_welch(max_iterations, N, M, T, observations, init_prob,
                trans_prob, emit_prob, alpha, beta, ggamma, sigma);
    }
    cycles = stop_tsc(start)/num_runs;
    return (double) cycles;
}


int main(int argc, char **argv) {

    if ( argc != 2 ) {
        printf("usage: FW <max_iterations>\n");
        return -1;
    } int max_iterations = atoi(argv[1]);

    unsigned int fp_cost = 0;
    fp_cost += 1*T;
    fp_cost += 1*N;
    fp_cost += 1*N*N;
    fp_cost += 1*N*M;
    fp_cost += 3*T*N;
    fp_cost += 1*T*N*N;

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

    // Fill
    init();

    // compute will be executed in here
    double r = rdtsc(max_iterations);

    printf("\n");
    printf("(%d, %lf)\n", max_iterations, fp_cost / r);
    printf("(%d, %lf)\n", N, fp_cost / r);
    printf("(%d, %lf)\n", M, fp_cost / r);
    printf("(%d, %lf)\n", T, fp_cost / r);
    printf("\n");

    print_states(N, M, T);

    // leakage is bad, mmmkay
    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(alpha);
    free(beta);
    free(ggamma);
    free(sigma);
}
