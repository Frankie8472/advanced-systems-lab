/*
    Declarations for different implementations and optimizations for the algorithm. Also
    provides functionality to register functions to benchmark and test the implementations

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#if !defined(__BW_COMMON_H)
#define __BW_COMMON_H

#include <string>
#include <cstring>
#include "helper_utilities.h"

struct BWdata {
    const unsigned int K;  // number of observation sequences / training datasets
    const unsigned int N;  // number of hidden state variables
    const unsigned int M;  // number of distinct observations
    const unsigned int T;  // number of time steps

    // (for each observation/training sequence 0 <= k < K)
    unsigned int* const observations; //  [K][T]          [k][t]            := observation sequence k at time_step t
    double* const init_prob; //           [N]             [n]               := P(X_1 = n)
    double* const trans_prob; //          [N][N]          [n0][n1]          := P(X_t = n1 | X_(t-1) = n0)
    double* const emit_prob; //           [N][M]          [n][m]            := P(Y_t = y_m | X_t = n)
    double* const neg_log_likelihoods; // Array to store the neg_log_likelihood for each iteration
    double* const c_norm; //              [K][T]          [k][t]            :=  scaling/normalization factor for numerical stability
    double* const alpha; //               [K][T][N]       [k][t][n]         :=  P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
    double* const beta; //                [K][T][N]       [k][t][n]         :=  P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
    double* const ggamma; //              [K][T][N]       [k][t][n]         :=  P(X_t = n | Y, theta)
    double* const sigma; //               [K][T][N][N]    [k][t][n0][n1]    :=  P(X_t = n0, X_(t+1) = n1 | Y, theta)
    // where theta = {init_prob, trans_prob, emit_prob} represent the model parameters we want learn/refine/estimate iteratively.
    double* const gamma_sum; //           [K][N]
    double* const sigma_sum; //           [K][N][N]
};


/**
 * \brief Initiallizes the struct that holds all information needed to execute the algorithm.
 */
inline const BWdata& initialize_BWdata(
    const unsigned int K,
    const unsigned int N,
    const unsigned int M,
    const unsigned int T,
    const unsigned int max_iterations
    ) {

    if (K < 4 || K % 4 != 0) {
        printf("\nVIOLATION: K is %d, but must be >= 4 and divisible by 4", K);
        exit(1);
    }
    if (N < 4 || N % 4 != 0) {
        printf("\nVIOLATION: N is %d, but must be >= 4 and divisible by 4", N);
        exit(1);
    }
    if (M < 4 || M % 4 != 0) {
        printf("\nVIOLATION: M is %d, but must be >= 4 and divisible by 4", M);
        exit(1);
    }
    if (T < 4 || T % 4 != 0) {
        printf("\nVIOLATION: T is %d, but must be >= 4 and divisible by 4", T);
        exit(1);
    }
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

    const BWdata& bw = {
        K,
        N,
        M,
        T,
        observations,
        init_prob,
        trans_prob,
        emit_prob,
        // c_norm
        (double *)calloc(K*T, sizeof(double)),
        // alpha
        (double *)calloc(K*T*N, sizeof(double)),
        // beta
        (double *)calloc(K*T*N, sizeof(double)),
        // ggamma
        (double *)calloc(K*T*N, sizeof(double)),
        // sigma
        (double *)calloc(K*T*N*N, sizeof(double)),
        // ggamma_sum
        (double *)calloc(K*N, sizeof(double)),
        // sigma_sum
        (double *)calloc(K*N*N, sizeof(double))
    };


    if (bw.c_norm == NULL) exit(1);
    if (bw.alpha == NULL) exit(1);
    if (bw.beta == NULL) exit(1);
    if (bw.ggamma == NULL) exit(1);
    if (bw.sigma == NULL) exit(1);
    if (bw.gamma_sum == NULL) exit(1);
    if (bw.sigma_sum == NULL) exit(1);

    // I hate C++ so doing C style
                                               
    BWdata* bw_ptr = (BWdata*)malloc(sizeof(BWdata));
    if (bw_ptr == NULL) exit(1);
    memcpy(bw_ptr, &bw, sizeof(BWdata));
    
    return *bw_ptr;
}

inline void clean_BWdata(const BWdata& bw){
    free(bw.c_norm);
    free(bw.alpha);
    free(bw.beta);
    free(bw.ggamma);
    free(bw.sigma);
    free(bw.gamma_sum);
    free(bw.sigma_sum);
    free(bw.observations);
    free(bw.init_prob);
    free(bw.trans_prob);
    free(bw.emit_prob);
    free(bw.neg_log_likelihoods);
}

/**
 * Function interface for an implementation for the Baum-Welch algorithm
 */
typedef int(*compute_bw_func)(const BWdata& bw, const unsigned int max_iterations);
void add_function(compute_bw_func f, std::string name);

// Macro to register a function and a name that should be executed
#define REGISTER_FUNCTION(f, name)                                \
    static struct cls##_                                          \
    {                                                             \
        cls##_()                                                  \
        {                                                         \
            add_function(f, name);                                \
        }                                                         \
    } cls##__BW_;

#endif /* __BW_COMMON_H */