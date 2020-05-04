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
            bw.observations[k*T + t] = t;
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
            // % T would be wrong, because the observations sequence (over time 0 <= t < T)
            // represents observations (categorical random variable) in 0 <= m < M
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

    // check if the initial distribution sums to 1.0
    init_sum = 0.0;
    for (size_t n = 0; n < N; n++) init_sum += bw.init_prob[n];
    if (1.0 - 1e-12 < init_sum && init_sum < 1.0 + 1e-12) PRINT_PASSED("init_prob sums to 1.0");
    else PRINT_VIOLATION("init_prob sums to %lf\n", 1L, init_sum);

    // check if the rows of the transition distribution sum to 1.0
    errors = 0;
    for (size_t n0 = 0; n0 < N; n0++) {
        trans_sum = 0.0;
        for (size_t n1 = 0; n1 < N; n1++) {
            trans_sum += bw.trans_prob[n0*N + n1];
        }
        if ( ! ( 1.0 - 1e-12 < trans_sum && trans_sum < 1.0 + 1e-12 ) ) { 
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
        if ( ! ( 1.0 - 1e-12 < emit_sum && emit_sum < 1.0 + 1e-12 ) ) { 
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
        if (old_nll > new_nll + 1e-12) {
            errors++;
            printf("[%zu]\t%lf\t > \t%lf \t(old nll > new nll)\n", iterations, old_nll, new_nll);
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

    printf("\nObservations (tip: shouldn't change after initialization):\n");
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

    printf("\nc_norm:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            printf("c_norm[k = %zu][t = %zu] = %f\n", k, t, bw.c_norm[k*bw.T + t]);
        }
    }

    printf("\nalpha:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("alpha[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.alpha[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    printf("\nbeta:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("beta[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.beta[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    printf("\nggamma:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n = 0; n < bw.N; n++) {
                printf("ggamma[k = %zu][t = %zu][n = %zu] = %f\n", k, t, n, bw.ggamma[(k*bw.T + t)*bw.N + n]);
            }
        }
    }

    printf("\ngamma_sum:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            printf("gamma_sum[k = %zu][n = %zu] = %f\n", k, n, bw.gamma_sum[k*bw.N + n]);
        }
    }

    printf("\nsigma:\n");
    for(size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            for (size_t n0 = 0; n0 < bw.N; n0++) {
                for(size_t n1 = 0; n1 < bw.N; n1++) {
                    printf("sigma[k = %zu][t = %zu][n0 = %zu][n1 = %zu] = %f\n", k, t, n0, n1, bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1]);
                }
            }
        }
    }

    printf("\nsigma_sum:\n");
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                printf("sigma_sum[k = %zu][n0 = %zu][n1 = %zu] = %f\n", k, n0, n1, bw.sigma_sum[(k*bw.N + n0)*bw.N + n1]);
            }
        }
    }

}


#endif /* __BW_HELPER_UTILITIES_H */