/*
    Baseline implementation

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------

    Make sure you understand it! Refer to
    https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf

    Assumptions: K can be 1 or more
    N, M and T must be divisible by 4

    Verified by checking monotonously decreasing
    sequence of negative log likelihoods and whether
    the rows of init_prob, trans_prob and emit_prob
    sum to 1.0, with large K, N, M and T and u.a.r.
    observations, init_prob, trans_prob and emit_prob

    Code checked against Matlab implementation of
    https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/
*/

#include <cmath>
#include <cstring>

struct BWdata {
    unsigned int K;  // number of observation sequences / training datasets
    unsigned int N;  // number of hidden state variables
    unsigned int M;  // number of bw.observations
    unsigned int T;  // number of time steps

    // (for each oberservation/training sequence 0 <= k < K)
    unsigned int* observations; //  [K][T]          [k][t]            := observation sequence k at time_step t
    double* init_prob; //           [N]             [n]               := P(X_1 = n)
    double* trans_prob; //          [N][N]          [n0][n1]          := P(X_t = n1 | X_(t-1) = n0)
    double* emit_prob; //           [N][M]          [n][m]            := P(Y_t = y_m | X_t = n)
    double* c_norm; //              [K][T]          [k][t]            :=  scaling/normalization factor for numerical stability
    double* alpha; //               [K][T][N]       [k][t][n]         :=  P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
    double* beta; //                [K][T][N]       [k][t][n]         :=  P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
    double* ggamma; //              [K][T][N]       [k][t][n]         :=  P(X_t = n | Y, theta)
    double* sigma; //               [K][T][N][N]    [k][t][n0][n1]    :=  P(X_t = n0, X_(t+1) = n1 | Y, theta)
    // where theta = {init_prob, trans_prob, emit_prob} represent the model parameters we want learn/refine/estimate iteratively.
    double* gamma_sum; //           [K][N]
    double* sigma_sum; //           [K][N][N]
};


void forward_step(const BWdata& bw);
void backward_step(const BWdata& bw);
void compute_gamma(const BWdata& bw);
void compute_sigma(const BWdata& bw);
void update_init_prob(const BWdata& bw);
void update_trans_prob(const BWdata& bw);
void update_emit_prob(const BWdata& bw);


void compute_baum_welch(
    unsigned int max_iterations,
    unsigned int K,
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* init_prob,
    double* trans_prob,
    double* emit_prob,
    double neg_log_likelihoods[]
    ) {

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

    for (int i = 0; i < max_iterations; i++) {

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        compute_sigma(bw);
        update_init_prob(bw);
        update_trans_prob(bw);
        update_emit_prob(bw);

        // negative log likelihood for convergence criterion
        // (we don't care about the actual convergence criterion)
        // (since we want a fixed number of iterations for benchmarking purposes)
        // (however, the negative log likelihood is used for verification purposes)
        // (in each iteration, the new negative log likelihood should be)
        // (less than or equal to the old negative log likelihood)
        // (=> guaranteed to find a (local) minima)
        double neg_log_likelihood_sum = 0.0;
        for (int k = 0; k < bw.K; k++) {
            for (int t = 0; t < bw.T; t++) {
                neg_log_likelihood_sum = neg_log_likelihood_sum - log(bw.c_norm[k*T + t]);
            }
        }
        neg_log_likelihoods[i] = neg_log_likelihood_sum;

        // reinitialization to 0.0 for the next iteration
        memset(bw.c_norm, 0, bw.K*bw.T*sizeof(double));
        memset(bw.alpha, 0, bw.K*bw.T*bw.N*sizeof(double));
        memset(bw.beta, 0, bw.K*bw.T*bw.N*sizeof(double));
        memset(bw.ggamma, 0, bw.K*bw.T*bw.N*sizeof(double));
        memset(bw.sigma, 0, bw.K*bw.T*bw.N*bw.N*sizeof(double));
        memset(bw.gamma_sum, 0, bw.K*bw.N*sizeof(double));
        memset(bw.sigma_sum, 0, bw.K*bw.N*bw.N*sizeof(double));
    }

    free(bw.c_norm);
    free(bw.alpha);
    free(bw.beta);
    free(bw.ggamma);
    free(bw.sigma);
    free(bw.gamma_sum);
    free(bw.sigma_sum);
}


inline void forward_step(const BWdata& bw) {
    for (int k = 0; k < bw.K; k++) {
        // t = 0, base case
        for (int n = 0; n < bw.N; n++) {
            bw.alpha[(k*bw.T + 0)*bw.N + n] = bw.init_prob[n]*bw.emit_prob[n*bw.M + bw.observations[k*bw.T + 0]];
            bw.c_norm[k*bw.T + 0] += bw.alpha[(k*bw.T + 0)*bw.N + n];
        } 

        bw.c_norm[k*bw.T + 0] = 1.0/bw.c_norm[k*bw.T + 0];
        for (int n = 0; n < bw.N; n++){
	    bw.alpha[(k*bw.T + 0)*bw.N + n] *= bw.c_norm[k*bw.T + 0];
	}

        // recursion step
        for (int t = 1; t < bw.T; t++) {
            for (int n0 = 0; n0 < bw.N; n0++) {
                double alpha_temp = 0.0;
                for (int n1 = 0; n1 < bw.N; n1++) {
                    alpha_temp += bw.alpha[(k*bw.T + (t-1))*bw.N + n1]*bw.trans_prob[n1*bw.N + n0];
                }
                bw.alpha[(k*bw.T + t)*bw.N + n0] = bw.emit_prob[n0*bw.M + bw.observations[k*bw.T + t]] * alpha_temp;
                bw.c_norm[k*bw.T + t] += bw.alpha[(k*bw.T + t)*bw.N + n0];
            }
            bw.c_norm[k*bw.T + t] = 1.0/bw.c_norm[k*bw.T + t];
            for (int n0 = 0; n0 < bw.N; n0++) {
                bw.alpha[(k*bw.T + t)*bw.N + n0] *= bw.c_norm[k*bw.T + t];
            }
        }
    }
}


inline void backward_step(const BWdata& bw) {
    for (int k = 0; k < bw.K; k++) {
        // t = bw.T, base case
        for (int n = 0; n < bw.N; n++) {
            bw.beta[(k*bw.T + (bw.T-1))*bw.N + n] = 1.0*bw.c_norm[k*bw.T + (bw.T-1)];
        }

        // recursion step
        for (int t = bw.T-2; t >= 0; t--) {
            for (int n0 = 0; n0 < bw.N; n0++) {
                double beta_temp = 0.0;
                for (int n1 = 0; n1 < bw.N; n1++) {
                    beta_temp += bw.beta[(k*bw.T + (t+1))*bw.N + n1] * bw.trans_prob[n0*bw.N + n1] * bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                }
                bw.beta[(k*bw.T + t)*bw.N + n0] = beta_temp * bw.c_norm[k*bw.T + t];
            }
        }
    }
}


inline void compute_gamma(const BWdata& bw) {
    for (int k = 0; k < bw.K; k++) {
        for (int t = 0; t < bw.T; t++) {
            for (int n = 0; n < bw.N; n++) {
                bw.ggamma[(k*bw.T + t)*bw.N + n] = bw.alpha[(k*bw.T + t)*bw.N + n] * bw.beta[(k*bw.T + t)*bw.N + n] / bw.c_norm[k*bw.T + t];
            }
        }
    }

    // sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob)
    for (int k = 0; k < bw.K; k++) {
        for (int n = 0; n < bw.N; n++) {
            double g_sum = 0.0;
            for (int t = 0; t < bw.T-1; t++) {
                g_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];
            }
            bw.gamma_sum[k*bw.N + n] = g_sum;
        }
    }
}


inline void compute_sigma(const BWdata& bw) {
    for (int k = 0; k < bw.K; k++) {
        for (int t = 0; t < bw.T-1; t++) {
            for (int n0 = 0; n0 < bw.N; n0++) {
                for (int n1 = 0; n1 < bw.N; n1++) {
                    bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1] = \
                        bw.alpha[(k*bw.T + t)*bw.N + n0]*bw.trans_prob[n0*bw.N + n1]*bw.beta[(k*bw.T + (t+1))*bw.N + n1]*bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];
                }
            }
        }

        // sum up bw.sigma (from t = 0 to bw.T-1)
        for (int n0 = 0; n0 < bw.N; n0++) {
            for (int n1 = 0; n1 < bw.N; n1++) {
                double s_sum = 0.0;
                for (int t = 0; t < bw.T-1; t++) {
                    s_sum += bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1];
                }
                bw.sigma_sum[(k*bw.N + n0)*bw.N + n1] = s_sum;
            }
        }
    }
}


inline void update_init_prob(const BWdata& bw) {
    for (int n = 0; n < bw.N; n++) {
        double g0_sum = 0.0;
        for (int k = 0; k < bw.K; k++) {
            g0_sum += bw.ggamma[(k*bw.T + 0)*bw.N + n];
        }
        bw.init_prob[n] = g0_sum/bw.K;
    }
}


inline void update_trans_prob(const BWdata& bw) {
    for (int n0 = 0; n0 < bw.N; n0++) {
        for (int n1 = 0; n1 < bw.N; n1++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (int k = 0; k < bw.K; k++) {
                numerator_sum += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1];
                denominator_sum += bw.gamma_sum[k*bw.N + n0];
            }
            bw.trans_prob[n0*bw.N + n1] = numerator_sum / denominator_sum;
        }
    }
}


inline void update_emit_prob(const BWdata& bw) {
    // add last bw.T-step to bw.gamma_sum
    for (int k = 0; k < bw.K; k++) {
        for (int n = 0; n < bw.N; n++) {
            bw.gamma_sum[k*bw.N + n] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n];
        }
    }
    // update bw.emit_prob
    for (int n = 0; n < bw.N; n++) {
        for (int m = 0; m < bw.M; m++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (int k = 0; k < bw.K; k++) {
                double ggamma_cond_sum = 0.0;
                for (int t = 0; t < bw.T; t++) {
                    if (bw.observations[k*bw.T + t] == m) {
                        ggamma_cond_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];
                    }
                }
                numerator_sum += ggamma_cond_sum;
                denominator_sum += bw.gamma_sum[k*bw.N + n];
            }
            bw.emit_prob[n*bw.M + m] = numerator_sum / denominator_sum;
        }
    }
}
