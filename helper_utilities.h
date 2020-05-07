/*
    Helper Utilities
    Throw all useful functions that may or may not be used more than once in here

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#if !defined(__BW_HELPER_UTILITIES_H)
#define __BW_HELPER_UTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <random>

#include "common.h"

#define PRINT_PASSED(msg, ...) printf("\x1b[1;32mPASSED:\x1b[0m " msg "\n",  ##__VA_ARGS__)
#define PRINT_FAIL(msg, ...) printf("\x1b[1;31mFAIL:\x1b[0m " msg "\n",  ##__VA_ARGS__)
#define PRINT_VIOLATION(msg, num, ...) printf("\x1b[1;35m%zu VIOLATIONS:\x1b[0m " msg "\n", num,  ##__VA_ARGS__)

void initialize_uar(const BWdata& bw);
void initialize_random(const BWdata& bw);
bool check_and_verify(const BWdata& bw);
void print_states(const BWdata& bw);
void print_BWdata(const BWdata& bw);
bool is_BWdata_equal_only_probabilities(const BWdata& bw1, const BWdata& bw2);
void print_BWdata_debug_helper(const BWdata& bw, const size_t iteration_variable, const char* message);


inline void initialize_uar(const BWdata& bw) {
    const size_t K = bw.K;
    const size_t N = bw.N;
    const size_t M = bw.M;
    const size_t T = bw.T;

    // uniform at random set init_prob and trans_prob
    for (size_t n0 = 0; n0 < N; n0++) {
        bw.init_prob[n0] = 1.0/N;
        for (size_t n1 = 0; n1 < N; n1++) {
            bw.trans_prob[n0*N + n1] = 1.0/N;
        }
    }

    // uniform at random set emit_prob
    for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
            bw.emit_prob[n*M + m] = 1.0/M;
        }
    }

    // uniform at random set observations
    // (well, not really u.a.r. but let's pretend)
    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            // % T would be wrong, because the observation sequence (over time 0 <= t < T)
            // represent observations (categorical random variable) in 0 <= m < M
            bw.observations[k*T + t] = t % M;
        }
    }
}


inline void initialize_random(const BWdata& bw) {
    const size_t K = bw.K;
    const size_t N = bw.N;
    const size_t M = bw.M;
    const size_t T = bw.T;

    double init_sum;
    double trans_sum;
    double emit_sum;

    // randomly initialized init_prob
    init_sum = 0.0;
    while(init_sum == 0.0){
        for (size_t n = 0; n < N; n++) {
            bw.init_prob[n] = rand();
            init_sum += bw.init_prob[n];
        }
    }

    // the array init_prob must sum to 1.0
    for (size_t n = 0; n < N; n++) {
        bw.init_prob[n] /= init_sum;
    }

    // randomly initialized trans_prob rows
    for (size_t n0 = 0; n0 < N; n0++) {
        trans_sum = 0.0;
        while(trans_sum == 0.0){
            for (size_t n1 = 0; n1 < N; n1++) {
                bw.trans_prob[n0*N + n1] = rand();
                trans_sum += bw.trans_prob[n0*N + n1];
            }
        }

        // the row trans_prob[n0*N] must sum to 1.0
        for (size_t n1 = 0; n1 < N; n1++) {
            bw.trans_prob[n0*N + n1] /= trans_sum;
        }
    }

    // randomly initialized emit_prob rows
    for (size_t n = 0; n < N; n++) {
        emit_sum = 0.0;
        while (emit_sum == 0.0) {
            for (size_t m = 0; m < M; m++) {
                bw.emit_prob[n * M + m] = rand();
                emit_sum += bw.emit_prob[n * M + m];
            }
        }

        // the row emit_prob[n*M] must sum to 1.0
        for (size_t m = 0; m < M; m++) {
            bw.emit_prob[n*M + m] /= emit_sum;
        }
    }

    // fixed observation (can be changed to e.g. all 1 for verification)
    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            // % T would be wrong, because the observation sequence (over time 0 <= t < T)
            // represent observations (categorical random variable) in 0 <= m < M
            bw.observations[k*T + t] = rand() % M;
        }
    }
}

/**
 * TODO description
 *
 * returns: True if there was no error, false otherwise
 */
inline bool check_and_verify(const BWdata& bw) {
    const size_t N = bw.N;
    const size_t M = bw.M;

    size_t errors;
    double init_sum;
    double trans_sum;
    double emit_sum;
    bool success = true;

    const double epsilon = 10-8;

    // check if the initial distribution sums to 1.0
    init_sum = 0.0;
    for (size_t n = 0; n < N; n++) init_sum += bw.init_prob[n];
    if ( abs(init_sum - 1.0) < epsilon ) PRINT_PASSED("init_prob sums to 1.0");
    else PRINT_VIOLATION("init_prob sums to %lf\n", 1L, init_sum);

    // check if the rows of the transition distribution sum to 1.0
    errors = 0;
    for (size_t n0 = 0; n0 < N; n0++) {
        trans_sum = 0.0;
        for (size_t n1 = 0; n1 < N; n1++) {
            trans_sum += bw.trans_prob[n0*N + n1];
        }
        if ( ! ( abs(trans_sum - 1.0) < epsilon ) ) { 
            errors++;
            PRINT_FAIL("trans_prob[%zu] sums to %lf", n0, trans_sum);
        }
    }
    if (errors > 0) {
        PRINT_VIOLATION("of rows in trans_prob that do not sum to 1.0", errors);
        success = false;
    } else {
        PRINT_PASSED("trans_prob rows sum to 1.0");
    }

    // check if the rows of the emission distribution sum to 1.0
    errors = 0;
    for (size_t n = 0; n < N; n++) {
        emit_sum = 0.0;
        for (size_t m = 0; m < M; m++) {
            emit_sum += bw.emit_prob[n*M + m];
        }
        if ( ! ( abs(emit_sum - 1.0) < epsilon ) ) { 
            errors++;
            PRINT_FAIL("emit_prob[%zu] sums to %lf", n, emit_sum);
        }
    }
    if (errors > 0) {
        PRINT_VIOLATION("of rows in emit_prob that do not sum to 1.0", errors);
        success = false;
    } else {
        PRINT_PASSED("emit_prob rows sum to 1.0");
    }

    // check the negative log likelihood sequence for monotonicity
    errors = 0;
    for (size_t iterations = 1; iterations < bw.max_iterations; iterations++) {
        double old_nll = bw.neg_log_likelihoods[iterations-1];
        double new_nll = bw.neg_log_likelihoods[iterations];
        // Note that we We want old_nll >= new_nll,
        // because we want to minimize the negative log likelihood.
        // Hence, we want to throw an error if and only if old_nll < new_nll.
        // Therefore, we need the epsilon here to account for numerical errors of small numbers.
        // (we always operate on the scale where -infinity < log(x) <= 0, i.e. 0 < x <= 1, due to x being a probability)
        if (old_nll < new_nll - epsilon) {
            errors++;
            printf("[%zu]\t%lf\t > \t%lf \t(old nll < new nll)\n", iterations, old_nll, new_nll);
        }
    }
    if (errors > 0){
        PRINT_VIOLATION("of the monotonicity of the negative log likelihood\n", errors);
        success = false;
    } else {
        PRINT_PASSED("monotonocity of the negative log likelihood\n");
    }

    return success;
}


inline void print_states(const BWdata& bw) {
    const size_t N = bw.N;
    const size_t M = bw.M;

    printf("\nInitialization probabilities:\n");
    for(size_t n = 0; n < N; n++) {
        printf("Pr[X_1 = %zu] = %f\n", n, bw.init_prob[n]);
    }

    printf("\nTransition probabilities:\n");
    for(size_t n0 = 0; n0 < N; n0++) {
        for(size_t n1 = 0; n1 < N; n1++) {
            printf("Pr[X_t = %zu | X_(t-1) = %zu ] = %f\n", n1, n0, bw.trans_prob[n0*N + n1]);
        }
    }

    printf("\nEmission probabilities:\n");
    for(size_t n = 0; n < N; n++) {
        for(size_t m = 0; m < M; m++) {
            printf("Pr[Y_t = %zu | X_t = %zu] = %f\n", m, n, bw.emit_prob[n*M + m]);
        }
    }
    printf("\n");
}


/**
 * Description
 * Useful for debugging
 * Only use for small values K, N, M, T
 * Prints all (!) contents of input BWdata& bw
 */
inline void print_BWdata(const BWdata& bw) {

    printf("\nK %zu, N %zu, M %zu, T %zu, max_iterations %zu\n", bw.K, bw.N, bw.M, bw.T, bw.max_iterations);

    printf("\nObservation Data (tip: shouldn't change after initialization):\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            printf("obs[k = %zu][t = %zu] = %zu\n", k, t, bw.observations[k*bw.T + t]);
        }
    }

    print_states(bw); // prints bw.init_prob, bw.trans_prob and bw.emit_prob

    printf("Negative Log Likelihoods (tip: should change once per iteration):\n");
    for (size_t it = 0; it < bw.max_iterations; it++) {
        printf("NLL[it = %zu] = %f\n", it, bw.neg_log_likelihoods[it]);
    }

    //printf("\nc_norm:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            printf("c_norm[k = %zu][t = %zu] = %f\n", k, t, bw.c_norm[k*bw.T + t]);
        }
    }

    //printf("\nalpha:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("alpha[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.alpha[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    //printf("\nbeta:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("beta[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.beta[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    //printf("\nggamma:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("ggamma[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.ggamma[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    //printf("\ngamma_sum:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            printf("gamma_sum[k = %zu][n = %zu] = %f\n", k, n, bw.gamma_sum[k*bw.N + n]);
        }
    }

    //printf("\nsigma:\n");
    for(size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                for(size_t n1 = 0; n1 < bw.N; n1++) {
                    printf("sigma[k = %zu][t = %zu][n0 = %zu][n1 = %zu] = %f\n", k, t, n0, n1, bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1]);
                }
            }
        }
    }

    //printf("\nsigma_sum:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                printf("sigma_sum[k = %zu][n0 = %zu][n1 = %zu] = %f\n", k, n0, n1, bw.sigma_sum[(k*bw.N + n0)*bw.N + n1]);
            }
        }
    }

}


/**
 * INPUT
 * two const BWdata& structs to compare against
 * OUTPUT
 * true if both structs contain the same below mentioned data up to some numerical epsilon
 * DESCRIPTION
 * checks whether the following match up to an numerical epsilon:
 * Initialization Probabilities, Transition Probabilities and Emission Probabilities
 * */
inline bool is_BWdata_equal_only_probabilities(const BWdata& bw1, const BWdata& bw2) {

    if (bw1.K != bw2.K) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m K1 = %zu is not %zu = K2\n", bw1.K, bw2.K);
        return false;
    }

    if (bw1.N != bw2.N) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m N1 = %zu is not %zu = N2\n", bw1.N, bw2.N);
        return false;
    }

    if (bw1.M != bw2.M) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m M1 = %zu is not %zu = M2\n", bw1.M, bw2.M);
        return false;
    }

    if (bw1.T != bw2.T) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m T1 = %zu is not %zu = T2\n", bw1.T, bw2.T);
        return false;
    }

    if (bw1.max_iterations != bw2.max_iterations) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m maxIterations1 = %zu is not %zu = maxIterations2\n",
            bw1.max_iterations, bw2.max_iterations
        );
        return false;
    }

    const size_t N = bw1.N;
    const size_t M = bw1.M;

    size_t errors_local = 0;
    size_t errors_total = 0;

    const double epsilon = 1e-6;

    for(size_t n = 0; n < N; n++) {
        const size_t index = n;
        const double err_abs_diff = abs(bw1.init_prob[index] - bw2.init_prob[index]);
        if ( ! ( err_abs_diff < epsilon ) ) { 
            errors_local++;
        }
    }

    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in init_prob!\n", errors_local);
    }

    errors_total += errors_local;
    errors_local = 0;

    for(size_t n0 = 0; n0 < N; n0++) {
        for(size_t n1 = 0; n1 < N; n1++) {
            const size_t index = n0*N + n1;
            const double err_abs_diff = abs(bw1.trans_prob[index] - bw2.trans_prob[index]);
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }

    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in trans_prob!\n", errors_local);
    }
      

    errors_total += errors_local;
    errors_local = 0;

    for(size_t n = 0; n < N; n++) {
        for(size_t m = 0; m < M; m++) {
            const size_t index = n*M + m;
            const double err_abs_diff = abs(bw1.emit_prob[index] - bw2.emit_prob[index]);
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }

    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in emit_prob!\n", errors_local);
    }

    errors_total += errors_local;
    errors_local = 0;

    if (errors_total > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in total!\n", errors_total);
    } else {
        printf("\x1b[1;32mBWdata IS equal:\x1b[0m All Probabilities Match!\n");
    }

    return ( !errors_total );
}


/**
 * INPUT
 * two const BWdata& structs to compare against
 * OUTPUT
 * true if both structs contain the same data up to some numerical epsilon
 * DESCRIPTION
 * checks whether each single field of each variable and array match up to an epsilon
 * */
// judge me aaaaaaaaaaaaall u want uwu
inline bool is_BWdata_equal(const BWdata& bw1, const BWdata& bw2) {

    if (bw1.K != bw2.K) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m K1 = %zu is not %zu = K2\n", bw1.K, bw2.K);
        return false;
    }

    if (bw1.N != bw2.N) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m N1 = %zu is not %zu = N2\n", bw1.N, bw2.N);
        return false;
    }

    if (bw1.M != bw2.M) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m M1 = %zu is not %zu = M2\n", bw1.M, bw2.M);
        return false;
    }

    if (bw1.T != bw2.T) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m T1 = %zu is not %zu = T2\n", bw1.T, bw2.T);
        return false;
    }

    if (bw1.max_iterations != bw2.max_iterations) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m maxIterations1 = %zu is not %zu = maxIterations2\n",
            bw1.max_iterations, bw2.max_iterations
        );
        return false;
    }

    const size_t K = bw1.K;
    const size_t N = bw1.N;
    const size_t M = bw1.M;
    const size_t T = bw1.T;
    const size_t max_iterations = bw1.max_iterations;

    size_t errors_local = 0;
    size_t errors_total = 0;

    const double epsilon = 1e-6;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            const size_t index = k*T + t;
            const double err_abs_diff = abs((double) (bw1.observations[index] - bw2.observations[index]));
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in observations!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

   for(size_t n = 0; n < N; n++) {
        const size_t index = n;
        const double err_abs_diff = abs(bw1.init_prob[index] - bw2.init_prob[index]);
        if ( ! ( err_abs_diff < epsilon ) ) { 
            errors_local++;
        }
    }

    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in init_prob!\n", errors_local);
    }

    errors_total += errors_local;
    errors_local = 0;

    for(size_t n0 = 0; n0 < N; n0++) {
        for(size_t n1 = 0; n1 < N; n1++) {
            const size_t index = n0*N + n1;
            const double err_abs_diff = abs(bw1.trans_prob[index] - bw2.trans_prob[index]);
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }

    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in trans_prob!\n", errors_local);
    }
      

    errors_total += errors_local;
    errors_local = 0;

    for(size_t n = 0; n < N; n++) {
        for(size_t m = 0; m < M; m++) {
            const size_t index = n*M + m;
            const double err_abs_diff = abs(bw1.emit_prob[index] - bw2.emit_prob[index]);
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }

    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in emit_prob!\n", errors_local);
    }

    errors_total += errors_local;
    errors_local = 0;

    for (size_t it = 0; it < max_iterations; it++) {
        const size_t index = it;
        const double err_abs_diff = abs(bw1.neg_log_likelihoods[index] - bw2.neg_log_likelihoods[index]);
        if ( ! ( err_abs_diff < epsilon ) ) { 
            errors_local++;
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in neg_log_likelihoods!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            const size_t index = k*T + t;
            const double err_abs_diff = abs(bw1.c_norm[index] - bw2.c_norm[index]);
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in c_norm!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n = 0; n < N; n++) {
                const size_t index = (k*T + t)*N + n;
                const double err_abs_diff = abs(bw1.alpha[index] - bw2.alpha[index]);
                if ( ! ( err_abs_diff < epsilon ) ) { 
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in alpha!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n = 0; n < N; n++) {
                const size_t index = (k*T + t)*N + n;
                const double err_abs_diff = abs(bw1.beta[index] - bw2.beta[index]);
                if ( ! ( err_abs_diff < epsilon ) ) { 
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in beta!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n = 0; n < N; n++) {
                const size_t index = (k*T + t)*N + n;
                const double err_abs_diff = abs(bw1.ggamma[index] - bw2.ggamma[index]);
                if ( ! ( err_abs_diff < epsilon ) ) { 
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in ggamma!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            const size_t index = k*N + n;
            const double err_abs_diff = abs(bw1.gamma_sum[index] - bw2.gamma_sum[index]);
            if ( ! ( err_abs_diff < epsilon ) ) { 
                errors_local++;
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in gamma_sum!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for(size_t k = 0; k < K; k++) {
        for (size_t t = 0; t < T; t++) {
            for (size_t n0 = 0; n0 < N; n0++) {
                for(size_t n1 = 0; n1 < N; n1++) {
                    const size_t index =((k*T + t)*N + n0)*N + n1;
                    const double err_abs_diff = abs(bw1.sigma[index] - bw2.sigma[index]);
                    if ( ! ( err_abs_diff < epsilon ) ) { 
                        errors_local++;
                    }
                }
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in sigma!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    for (size_t k = 0; k < K; k++) {
        for (size_t n0 = 0; n0 < N; n0++) {
            for (size_t n1 = 0; n1 < N; n1++) {
                const size_t index = (k*N + n0)*N + n1;
                const double err_abs_diff = abs(bw1.sigma_sum[index] - bw2.sigma_sum[index]);
                if ( ! ( err_abs_diff < epsilon ) ) { 
                    errors_local++;
                }
            }
        }
    }
    if (errors_local > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in sigma_sum!\n", errors_local);
    }
    errors_total += errors_local;
    errors_local = 0;

    if (errors_total > 0) {
        printf("\x1b[1;31mMismatch: BWdata is not equal!\x1b[0m [%zu] errors in total!\n", errors_total);
    } else {
        printf("\x1b[1;32mBWdata IS equal:\x1b[0m Everything Matches!\n");
    }

    return ( !errors_total );
}


inline void print_BWdata_debug_helper(const BWdata& bw, const size_t iteration_variable, const char* message) {
    printf("\n\x1b[1;33m[i = %zu] %s\x1b[0m", iteration_variable, message);
    print_BWdata(bw);
}


#endif /* __BW_HELPER_UTILITIES_H */