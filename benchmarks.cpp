/*
    Benchmarks for the various Baum Welch algorithm implementations
    Structure code such that it produces output results we can copy-paste into latex plots (minimize overhead)

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
#include <time.h>
// custom files for the project
#include "tsc_x86.h"
#include "helper_utilities.h"
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


int main(int argc, char **argv) {

    // randomize seed
    srand(time(NULL));

    const unsigned int K = 1; // number of observation sequences / training datasets
    const unsigned int N = 3; // number of hidden state variables
    const unsigned int M = 3; // number of observations
    const unsigned int T = 3; // number of time steps
    if ( argc != 2 ) {
        printf("usage: FW <max_iterations>\n");
        return -1;
    }
    const unsigned int max_iterations = atoi(argv[1]);

    unsigned int fp_cost = 0;
    fp_cost += 1*T;
    fp_cost += 1*N;
    fp_cost += 1*N*N;
    fp_cost += 1*N*M;
    fp_cost += 3*T*N;
    fp_cost += 1*T*N*N;

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

    initialize_uar(K, N, M, T, observations, init_prob, trans_prob, emit_prob);

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
            compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
        }
        cycles = stop_tsc(start);
        if ( cycles >= CYCLES_REQUIRED ) break;
        num_runs *= 2;
    }
#endif
    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
        compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    }
    cycles = stop_tsc(start)/num_runs;

    printf("\n");
    printf("(%d, %lf)\n", max_iterations, fp_cost / cycles);
    printf("(%d, %lf)\n", N, fp_cost / cycles);
    printf("(%d, %lf)\n", M, fp_cost / cycles);
    printf("(%d, %lf)\n", T, fp_cost / cycles);
    printf("\n");

    print_states(N, M, T, init_prob, trans_prob, emit_prob);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(neg_log_likelihoods);
}
