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
#define REP 50


/*
Cost analysis (add, mul and div is one flop)

forward: (1 add + 1 mul)*K*N²*T + (1 add + 2 mults)*K*N*T + (1 add + 2 mults)*K*N + (1 div)*K*T + (1 div)*K
backward: (1 add + 2 muls)*K*N²*(T-1) + (1 mult)*K*N*(T-1)
compute gamma: (1 div + 1 mult)*K*N*T + (1 add)*K*N*(T-1)
compute sigma: (1 add + 3 mults)*K*N²*(T-1)
update init: (1 add)*K*N + (1 div)*N
update trans: (2 adds)*K*N² + (1 div)*N²
update emit: (2 adds)*N*M*K + (1 add)*K*N*T + (1 add)*K*N + (1 div)*N*M

total: (1 add + 1 mul)*K*N²*T + (2 add + 5 muls)*K*N²*(T-1) + (2 adds)*K*N² + (1 div)*N² + (2 add + 3 mults + div)*K*N*T + (1 add + 1 muls)*K*N*(T-1)
    + (3 add + 2 mults)*K*N + (1 div)*K + (2 adds)*N*M*K + (1 div)*K*T + (1 div)*N + (1 div)*N*M
*/

int flops;

int main(int argc, char **argv) {

    // randomize seed
    srand(time(NULL));

    const unsigned int K = 4; // number of observation sequences / training datasets
    const unsigned int N = 4; // number of hidden state variables
    const unsigned int M = 4; // number of observations
    const unsigned int T = 4; // number of time steps
    if ( argc != 2 ) {
        printf("usage: FW <max_iterations>\n");
        return -1;
    }
    const unsigned int max_iterations = atoi(argv[1]);

    flops = 2*K*N*N*T + 7*K*N*N*(T-1) + 2*K*N*N + N*N + 6*K*N*T + 2*K*N*(T-1) + 5*K*N + K + 2*N*M*K + K*T + N + N*M;

    /*
    unsigned int fp_cost = 0;
    fp_cost += 1*T;
    fp_cost += 1*N;
    fp_cost += 1*N*N;
    fp_cost += 1*N*M;
    fp_cost += 3*T*N;
    fp_cost += 1*T*N*N;
    */

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


    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    double perf;
    myInt64 start, end;

#ifdef CALIBRATE
    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);
#endif

    // Actual performance measurements repeated REP times.
    double total_cycles = 0;
    int iter = 0;
    int total_iter = 0;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (size_t i = 0; i < num_runs; ++i) {
            iter += compute_baum_welch(max_iterations, K, N, M, T, observations, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;
        iter /= num_runs;
        total_iter += iter;

    }
    total_cycles /= REP;
    total_iter /= REP;


    cycles = total_cycles;
    iter = total_iter;
    perf =  round((100.0 * iter*flops) / cycles) / 100.0;

    printf("Performance: %f\n", perf);

    /*
    printf("\n");
    printf("(%d, %f)\n", max_iterations, fp_cost / (double) cycles);
    printf("(%d, %f)\n", N, fp_cost / (double) cycles);
    printf("(%d, %f)\n", M, fp_cost / (double) cycles);
    printf("(%d, %f)\n", T, fp_cost / (double) cycles);
    printf("\n");
    */

    //check_and_verify(max_iterations, N, M, init_prob, trans_prob, emit_prob, neg_log_likelihoods);
    //print_states(N, M, T, init_prob, trans_prob, emit_prob);

    free(observations);
    free(init_prob);
    free(trans_prob);
    free(emit_prob);
    free(neg_log_likelihoods);


}
