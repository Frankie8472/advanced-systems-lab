
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

    unsigned int K = 1; // number of observation sequences / training datasets
    unsigned int N = 3; // number of hidden state variables
    unsigned int M = 3; // number of observations
    unsigned int T = 3; // number of time steps

    if ( argc != 2 ) {
        printf("usage: FW <max_iterations>\n");
        return -1;
    } unsigned int max_iterations = atoi(argv[1]);

    unsigned int fp_cost = 0;
    fp_cost += 1*T;
    fp_cost += 1*N;
    fp_cost += 1*N*N;
    fp_cost += 1*N*M;
    fp_cost += 3*T*N;
    fp_cost += 1*T*N*N;

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    unsigned int* observations = (unsigned int *)calloc(K*T, sizeof(unsigned int));
    if (observations == NULL) exit (1);
    double* init_prob = (double *)calloc(N, sizeof(double));
    if (init_prob == NULL) exit (1);
    double* trans_prob = (double *)calloc(N*N, sizeof(double));
    if (trans_prob == NULL) exit (1);
    double* emit_prob = (double *)calloc(N*M, sizeof(double));
    if (emit_prob == NULL) exit (1);

    // uniform
    for (int n0 = 0; n0 < N; n0++) {
        init_prob[n0] = 1.0/N;
        for (int n1 = 0; n1 < N; n1++) {
            trans_prob[n0*N + n1] = 1.0/N;
        }
    }

    //uniform
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            emit_prob[n*M + m] = 1.0/M;
        }
    }

    // fixed observation (can be changed to e.g. all 1 for verification)
    for (int k = 0; k < K; k++) {
        for (int t = 0; t < T; t++) {
            observations[k*T + t] = t % 2;
        }
    }

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
            compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob);
        }
        cycles = stop_tsc(start);
        if ( cycles >= CYCLES_REQUIRED ) break;
        num_runs *= 2;
    }
#endif
    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
        compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob);
    }
    cycles = stop_tsc(start)/num_runs;

    printf("\n");
    printf("(%d, %lf)\n", max_iterations, fp_cost / cycles);
    printf("(%d, %lf)\n", N, fp_cost / cycles);
    printf("(%d, %lf)\n", M, fp_cost / cycles);
    printf("(%d, %lf)\n", T, fp_cost / cycles);
    printf("\n");

    print_states(N, M, T);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
}
