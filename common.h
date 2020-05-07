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
#include <vector>

struct BWdata {
    const size_t K;  // number of observation sequences / training datasets
    const size_t N;  // number of hidden state variables
    const size_t M;  // number of distinct observations
    const size_t T;  // number of time steps
    const size_t max_iterations; // Number of maximum iterations that should be performed

    // (for each observation/training sequence 0 <= k < K)
    size_t* const observations; //        [K][T]          [k][t]            :=  observation sequence k at time_step t
    double* const init_prob; //           [N]             [n]               :=  P(X_1 = n)
    double* const trans_prob; //          [N][N]          [n0][n1]          :=  P(X_t = n1 | X_(t-1) = n0)
    double* const emit_prob; //           [N][M]          [n][m]            :=  P(Y_t = y_m | X_t = n)
    double* const neg_log_likelihoods; // [max_iterations] [it]             :=  array to store the neg_log_likelihood for each iteration
    double* const c_norm; //              [K][T]          [k][t]            :=  scaling/normalization factor for numerical stability
    // NOTE that
    // min_{λ={init_prob, trans_prob, emit_prob}} -Σ_{k}(Σ_{t} log(Pr[obs[k][t]|λ]))
    // = min_{λ} -Σ_{k}(-Σ_{t} log(c_norm[k][t]))
    // = min_{λ} +Σ_{k}Σ_{t}log(c_norm[k][t])
    // (see the tutorial linked in baseline.cpp)
    double* const alpha; //               [K][T][N]       [k][t][n]         :=  P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
    double* const beta; //                [K][T][N]       [k][t][n]         :=  P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
    double* const ggamma; //              [K][T][N]       [k][t][n]         :=  P(X_t = n | Y, theta)
    double* const sigma; //               [K][T][N][N]    [k][t][n0][n1]    :=  P(X_t = n0, X_(t+1) = n1 | Y, theta)
    // where theta = {init_prob, trans_prob, emit_prob} represent the model parameters we want to learn/refine/estimate iteratively.
    double* const gamma_sum; //           [K][N]
    double* const sigma_sum; //           [K][N][N]
    bool full_copy;
};


/**
 * \brief Initiallizes the struct that holds all information needed to execute the algorithm.
 *        The matices allocated here are zeroed. To initialize with usefull date call the 
 *        appropriate functions.
 */
inline const BWdata& create_BWdata(
    const size_t K,
    const size_t N,
    const size_t M,
    const size_t T,
    const size_t max_iterations
    ) {

    /* NOTE:
    the wikipedia test case example in verifications.cpp needs values smaller than 4 for K, N and M

    if (K < 4 || K % 4 != 0) {
        printf("\x1b[1;31mVIOLATION:\x1b[0m K is %zu, but must be >= 4 and divisible by 4\n", K);
        exit(1);
    }
    if (N < 4 || N % 4 != 0) {
        printf("\x1b[1;31mVIOLATION:\x1b[0m N is %zu, but must be >= 4 and divisible by 4\n", N);
        exit(1);
    }
    if (M < 4 || M % 4 != 0) {
        printf("\x1b[1;31mVIOLATION:\x1b[0m M is %zu, but must be >= 4 and divisible by 4\n", M);
        exit(1);
    }
    if (T < 4 || T % 4 != 0) {
        printf("\x1b[1;31mVIOLATION:\x1b[0m T is %zu, but must be >= 4 and divisible by 4\n", T);
        exit(1);
    }
    */

    size_t* const observations = (size_t *)calloc(K*T, sizeof(size_t));
    double* const init_prob = (double *)calloc(N, sizeof(double));
    double* const trans_prob = (double *)calloc(N*N, sizeof(double));
    double* const emit_prob = (double *)calloc(N*M, sizeof(double));
    double* const neg_log_likelihoods = (double *)calloc(max_iterations, sizeof(double));
    double* const c_norm = (double *)calloc(K*T, sizeof(double));
    double* const alpha = (double *)calloc(K*T*N, sizeof(double));
    double* const beta = (double *)calloc(K*T*N, sizeof(double));
    double* const ggamma = (double *)calloc(K*T*N, sizeof(double));
    double* const sigma = (double *)calloc(K*T*N*N, sizeof(double));
    double* const ggamma_sum = (double *)calloc(K*N, sizeof(double));
    double* const sigma_sum = (double *)calloc(K*N*N, sizeof(double));

    if (observations == NULL) exit(2);
    if (init_prob == NULL) exit(2);
    if (trans_prob == NULL) exit(2);
    if (emit_prob == NULL) exit(2);
    if (neg_log_likelihoods == NULL) exit(2);
    if (c_norm == NULL) exit(2);
    if (alpha == NULL) exit(2);
    if (beta == NULL) exit(2);
    if (ggamma == NULL) exit(2);
    if (sigma == NULL) exit(2);
    if (ggamma_sum == NULL) exit(2);
    if (sigma_sum == NULL) exit(2);

    const BWdata& bw = {
        K,
        N,
        M,
        T,
        max_iterations,
        observations,
        init_prob,
        trans_prob,
        emit_prob,
        neg_log_likelihoods,
        c_norm,
        alpha,
        beta,
        ggamma,
        sigma,
        ggamma_sum,
        sigma_sum,
        true,
    };

    // I hate C++ so doing C style

    BWdata* bw_ptr = (BWdata*)calloc(1, sizeof(BWdata));
    if (bw_ptr == NULL) exit(2);
    memcpy(bw_ptr, &bw, sizeof(BWdata));

    return *bw_ptr;
}

inline const BWdata& copy_BWdata(const BWdata& bw){
    size_t N = bw.N;
    size_t M = bw.M;

    double* const init_prob = (double *)calloc(N, sizeof(double));
    double* const trans_prob = (double *)calloc(N*N, sizeof(double));
    double* const emit_prob = (double *)calloc(N*M, sizeof(double));

    if (init_prob == NULL) exit(2);
    if (trans_prob == NULL) exit(2);
    if (emit_prob == NULL) exit(2);

    const BWdata& new_bw = {
        bw.K,
        bw.N,
        bw.M,
        bw.T,
        bw.max_iterations,
        bw.observations,
        init_prob,
        trans_prob,
        emit_prob,
        bw.neg_log_likelihoods,
        bw.c_norm,
        bw.alpha,
        bw.beta,
        bw.ggamma,
        bw.sigma,
        bw.gamma_sum,
        bw.sigma_sum,
        false
    };

    // I hate C++ so doing C style

    BWdata* bw_ptr = (BWdata*)calloc(1, sizeof(BWdata));
    if (bw_ptr == NULL) exit(2);
    memcpy(bw_ptr, &new_bw, sizeof(BWdata));

    memcpy(init_prob, bw.init_prob, N * sizeof(double));
    memcpy(trans_prob, bw.trans_prob, N * N * sizeof(double));
    memcpy(emit_prob, bw.emit_prob, N * M * sizeof(double));

    return *bw_ptr;
}

inline const BWdata& full_copy_BWdata(const BWdata& bw){

    const BWdata& new_bw = copy_BWdata(bw);

    size_t K = bw.K;
    size_t N = bw.N;
    size_t T = bw.T;
    size_t* const observations = (size_t *)calloc(K*T, sizeof(size_t));
    double* const neg_log_likelihoods = (double *)calloc(bw.max_iterations, sizeof(double));
    double* const c_norm = (double *)calloc(K*T, sizeof(double));
    double* const alpha = (double *)calloc(K*T*N, sizeof(double));
    double* const beta = (double *)calloc(K*T*N, sizeof(double));
    double* const ggamma = (double *)calloc(K*T*N, sizeof(double));
    double* const sigma = (double *)calloc(K*T*N*N, sizeof(double));
    double* const ggamma_sum = (double *)calloc(K*N, sizeof(double));
    double* const sigma_sum = (double *)calloc(K*N*N, sizeof(double));

    if (observations == NULL) exit(2);
    if (neg_log_likelihoods == NULL) exit(2);
    if (c_norm == NULL) exit(2);
    if (alpha == NULL) exit(2);
    if (beta == NULL) exit(2);
    if (ggamma == NULL) exit(2);
    if (sigma == NULL) exit(2);
    if (ggamma_sum == NULL) exit(2);
    if (sigma_sum == NULL) exit(2);

    const BWdata& local_bw = {
        new_bw.K,
        new_bw.N,
        new_bw.M,
        new_bw.T,
        new_bw.max_iterations,
        observations,
        new_bw.init_prob,
        new_bw.trans_prob,
        new_bw.emit_prob,
        neg_log_likelihoods,
        c_norm,
        alpha,
        beta,
        ggamma,
        sigma,
        ggamma_sum,
        sigma_sum,
        true,
    };

    // I hate C++ so doing C style

    BWdata* bw_ptr = (BWdata*)calloc(1, sizeof(BWdata));
    if (bw_ptr == NULL) exit(2);

    memcpy(bw_ptr, &local_bw, sizeof(BWdata));

    memcpy(observations, bw.observations, K*T*sizeof(double));
    memcpy(neg_log_likelihoods, bw.neg_log_likelihoods, bw.max_iterations*sizeof(double));
    memcpy(c_norm, bw.c_norm, K*T*sizeof(double));
    memcpy(alpha, bw.alpha,  K*T*N*sizeof(double));
    memcpy(beta, bw.beta,  K*T*N*sizeof(double));
    memcpy(ggamma, bw.ggamma,  K*T*N*sizeof(double));
    memcpy(sigma, bw.sigma,  K*T*N*N*sizeof(double));
    memcpy(ggamma_sum, bw.gamma_sum,  K*N*sizeof(double));
    memcpy(sigma_sum, bw.sigma_sum,  K*N*N*sizeof(double));

    return *bw_ptr;
}

inline void clean_BWdata(const BWdata& bw){
    // Free matricies
    free(bw.init_prob);
    free(bw.trans_prob);
    free(bw.emit_prob);
    if(bw.full_copy){
        // WARNING: Possible Use after free when releasing a full_copy before all non-full_copies are released
        // We ignore this as we make sure we delete the copies first.
        free(bw.c_norm);
        free(bw.alpha);
        free(bw.beta);
        free(bw.ggamma);
        free(bw.sigma);
        free(bw.gamma_sum);
        free(bw.sigma_sum);
        free(bw.observations);
        free(bw.neg_log_likelihoods);
    }
    // Possibliy dangerous: Feeing memory of bw struct
    free((void *)&bw);
    // make bw data invalid
    //BWdata* bw_ptr = (BWdata *)&bw;
    //memset(bw_ptr, 0, sizeof(BWdata));
}

/**
 * Function interface for an implementation for the Baum-Welch algorithm
 */
typedef size_t(*compute_bw_func)(const BWdata& bw);

/**
 * Static class that handles the function registration. 
 */
class FuncRegister
{
public:

    /**
     * Set the function that is considererd the baseline and other 
     * implementations are compared against. There can only be one baseline
     */
    static void set_baseline(compute_bw_func f, std::string name);

    static void add_function(compute_bw_func f, std::string name);
    
    static void printRegisteredFuncs()
    {
        for(size_t i = 0; i < size(); i++){
            printf("%s at %p\n", (*func_names).at(i).c_str(), (*user_funcs).at(i));
        }
    }

    static size_t size()
    {
        return (*user_funcs).size();   
    }
    
    static std::vector<compute_bw_func> *user_funcs;
    static std::vector<std::string> *func_names;
    static compute_bw_func baseline_func;
    static std::string baseline_name;
};

// Macro to register a function and a name that should be executed
#define REGISTER_FUNCTION(f, name)                                \
    static struct f##_                                            \
    {                                                             \
        f##_()                                                    \
        {                                                         \
            FuncRegister::add_function(f, name);                  \
        }                                                         \
    } f##__BW_;

// Macro to register a function and a name that should be executed
#define SET_BASELINE(f, name)                                \
    static struct f##_                                            \
    {                                                             \
        f##_()                                                    \
        {                                                         \
            FuncRegister::set_baseline(f, name);                  \
        }                                                         \
    } f##__BW_;

#endif /* __BW_COMMON_H */