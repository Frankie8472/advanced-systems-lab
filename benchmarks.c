
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

#define N 3 // number of hidden state variables
#define M 3 // number of observations
#define T 3 // number of time steps

int* obs; //            [T]       [t]       := observation of time_step t
double* init_prob; //   [N]       [i]       := P(X_1 = i)
double* trans_prob; //  [N][N]    [i][j]    := P(X_t = j | X_(t-1) = i) 
double* emit_prob; //   [M][N]    [i][j]    := P(Y_t = y_i | X_t = j)
double* alpha; //       [T][N]    [t][i]    := P(Y_1 = y_1, ..., Y_t = y_t, X_t = i)
double* beta; //        [T][N]    [t][i]    := P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = i)
double* ggamma; //      [T][N]    [t][i]    := P(X_t = i | Y)
double* sigma; //       [T][N][N] [t][i][j] := P(X_t = i, X_(t+1) = j | Y)


void init() {

    // uniform
    for (int i = 0; i < N; i++) {
        init_prob[i] = 1.f/N;
        for (int j = 0; j < N; j++) {
            trans_prob[i*N + j] = 1.f/N;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            emit_prob[i*M + j] = 1.f/M;
        }
    }

    for (int t = 0; t < T; t++) {
        for (int j = 0; j < N; j++) {
            alpha[t*T + j] = 0;
            beta[t*T + j] = 0;
            for (uint k = 0; k < N; k++) {
                sigma[(t*T + j)*N + k] = 0;
            }
        }
    }

    // fixed observation
    for (int t = 0; t < T; t++) {
        obs[t] = t % 2;
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
            compute(max_iterations, N, M, T, obs, init_prob,
                    trans_prob, emit_prob, alpha, beta, ggamma, sigma);
        }
        cycles = stop_tsc(start);
        if ( cycles >= CYCLES_REQUIRED ) break;
        num_runs *= 2;
    }
#endif
    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
        compute(max_iterations, N, M, T, obs, init_prob,
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

    uint fp_cost = 0;
    fp_cost += 1*T;
    fp_cost += 1*N;
    fp_cost += 1*N*N;
    fp_cost += 1*M*N;
    fp_cost += 3*T*N;
    fp_cost += 1*T*N*N;

    obs = (int *)malloc(T*sizeof(int));
    init_prob = (double *)malloc(N*sizeof(double));
    trans_prob = (double *)malloc(N*N*sizeof(double));
    emit_prob = (double *)malloc(M*N*sizeof(double));
    alpha = (double *)malloc(T*N*sizeof(double));
    beta = (double *)malloc(T*N*sizeof(double));
    ggamma = (double *)malloc(T*N*sizeof(double));
    sigma = (double *)malloc(T*N*N*sizeof(double));

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

    printf("\nTransition probabilities:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("state %d -> state %d  %f\n", i+1, j+1, trans_prob[N*i + j]);
        }
    }

    printf("\nEmission probabilities:\n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf("state %d : %d  %f\n", i+1, j, emit_prob[N*i + j]);
        }
    }

    printf("\n");

    // leakage is bad, mmmkay
    free(obs);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(alpha);
    free(beta);
    free(ggamma);
    free(sigma);
}
