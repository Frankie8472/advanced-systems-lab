/*
    (should be correct; but still needs more verification!)

    Baseline implementation
    Make sure you understand it!
    Everything taken from here : 
    https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
*/

#include <cmath>
#include <cstring>

// (for each oberservation/training sequence 0 <= k < K)
double* c_norm; //           [K][T]       [k][t]            :=  scaling factor for numerical stability
double* alpha; //            [K][T][N]    [k][t][n]         :=  P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
double* beta; //             [K][T][N]    [k][t][n]         :=  P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
double* ggamma; //           [K][T][N]    [k][t][n]         :=  P(X_t = n | Y, theta)
double* sigma; //            [K][T][N][N] [k][t][n0][n1]    :=  P(X_t = n0, X_(t+1) = n1 | Y, theta)
// where theta = {init_prob, trans_prob, emit_prob} represent the model parameters we want learn/refine/estimate iteratively.

double* ggamma_sum; //       [K][N]
double* sigma_sum; //        [K][N][N]


void compute_baum_welch_iteration(
    unsigned int K,
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* init_prob,
    double* trans_prob,
    double* emit_prob
) {

    for (int k = 0; k < K; k++) {

        /* ------------------------------------------------------------------- */
        /* ------------------------- Forward-Step ---------------------------- */
        /* ------------------------------------------------------------------- */

        // t = 0, base case
        for (int n = 0; n < N; n++) {
            alpha[(k*T + 0)*N + n] = init_prob[n]*emit_prob[n*M + observations[k*T + 0]];
            c_norm[k*T + 0] += alpha[(k*T + 0)*N + n];
        } 

        c_norm[k*T + 0] = 1.0/c_norm[k*T + 0];
        for (int n = 0; n < N; n++) alpha[(k*T + 0)*N + n] *= c_norm[k*T + 0];

        // recursion step
        for (int t = 1; t < T; t++) {
            for (int n0 = 0; n0 < N; n0++) {
                double tmp_sum = 0.0;
                for (int n1 = 0; n1 < N; n1++) {
                    tmp_sum += alpha[(k*T + (t-1))*N + n1]*trans_prob[n1*N + n0];
                }
                alpha[(k*T + t)*N + n0] = emit_prob[n0*M + observations[k*T + t]] * tmp_sum;
                c_norm[k*T + t] += alpha[(k*T + t)*N + n0];
            }
            c_norm[k*T + t] = 1.0/c_norm[k*T + t];
            for (int n0 = 0; n0 < N; n0++) {
                alpha[(k*T + t)*N + n0] *= c_norm[k*T + t];
            }
        }

        /* ------------------------------------------------------------------- */
        /* ------------------------ Backward-Step ---------------------------- */
        /* ------------------------------------------------------------------- */

        // t = T, base case
        for (int n = 0; n < N; n++) {
            beta[(k*T + (T-1))*N + n] = 1.0*c_norm[k*T + (T-1)];
        }

        // recursion step
        for (int t = T-2; t >= 0; t--) {
            for (int n0 = 0; n0 < N; n0++) {
                double tmp_sum = 0.0;
                for (int n1 = 0; n1 < N; n1++) {
                    tmp_sum += beta[(k*T + (t+1))*N + n1] * trans_prob[n0*N + n1] * emit_prob[n1*M + observations[k*T + (t+1)]];
                }
                beta[(k*T + t)*N + n0] = tmp_sum*c_norm[k*T + t];
            }
        }

        /* ------------------------------------------------------------------- */
        /* ----------------------- Calculate Gamma --------------------------- */
        /* ------------------------------------------------------------------- */

        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                ggamma[(k*T + t)*N + n] = alpha[(k*T + t)*N + n] * beta[(k*T + t)*N + n] / c_norm[k*T + t];
            }
        }

        // sum up ggamma (from t = 0 to T-2)
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                double tmp_sum = 0.0;
                for (int t = 0; t < T-1; t++) {
                    tmp_sum += ggamma[(k*T + t)*N + n];
                }
                ggamma_sum[k*N + n] = tmp_sum;
            }
        }

        /* ------------------------------------------------------------------- */
        /* ----------------------- Calculate Sigma --------------------------- */
        /* ------------------------------------------------------------------- */

        for (int t = 0; t < T-1; t++) {
            for (int n0 = 0; n0 < N; n0++) {
                for (int n1 = 0; n1 < N; n1++) {
                    sigma[((k*T + t)*N + n0)*N + n1] = \
                        alpha[(k*T + t)*N + n0]*trans_prob[n0*N + n1]*beta[(k*T + (t+1))*N + n1]*emit_prob[n1*M + observations[k*T + (t+1)]];
                }
            }
        }

        // sum up sigma (from t = 0 to T-1)
        for (int n0 = 0; n0 < N; n0++) {
            for (int n1 = 0; n1 < N; n1++) {
                double tmp_sum = 0.0;
                for (int t = 0; t < T-1; t++) {
                    tmp_sum += sigma[((k*T + t)*N + n0)*N + n1];
                }
                sigma_sum[(k*N + n0)*N + n1] = tmp_sum;
            }
        }

    }

    /* ------------------------------------------------------------------- */
    /* ----------------- Update Initial Probabilities -------------------- */
    /* ------------------------------------------------------------------- */

    for (int n = 0; n < N; n++) {
        double tmp_sum = 0.0;
        for (int k = 0; k < K; k++) {
            tmp_sum += ggamma[(k*T + 0)*N + n];
        }
        init_prob[n] = tmp_sum/K;
    }

    /* ------------------------------------------------------------------- */
    /* ---------------- Update Transition Probabilities ------------------ */
    /* ------------------------------------------------------------------- */

    for (int n0 = 0; n0 < N; n0++) {
        for (int n1 = 0; n1 < N; n1++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (int k = 0; k < K; k++) {
                numerator_sum += sigma_sum[(k*N + n0)*N + n1];
                denominator_sum += ggamma_sum[k*N + n0];
            }
            trans_prob[n0*N + n1] = numerator_sum / denominator_sum;
        }
    }

    /* ------------------------------------------------------------------- */
    /* ----------------- Update Emission Probabilities ------------------- */
    /* ------------------------------------------------------------------- */

    // add last T-step to ggamma_sum
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            ggamma_sum[k*N + n] += ggamma[(k*T + (T-1))*N + n];
        }
    }

    // update emit_prob
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (int k = 0; k < K; k++) {
                double ggamma_cond_sum = 0.0;
                for (int t = 0; t < T; t++) {
                    if (observations[k*T + t] == m) {
                        ggamma_cond_sum += ggamma[(k*T + t)*N + n];
                    }
                }
                numerator_sum += ggamma_cond_sum;
                denominator_sum += ggamma_sum[k*N + n];
            }
            emit_prob[n*M + m] = numerator_sum / denominator_sum;
        }
    }

    return;
}


void compute_baum_welch(
    unsigned int iterations,
    unsigned int K,
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* init_prob,
    double* trans_prob,
    double* emit_prob
    ) {

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
    c_norm = (double *)calloc(K*T, sizeof(double));
    if (c_norm == NULL) exit(1);
    alpha = (double *)calloc(K*T*N, sizeof(double));
    if (alpha == NULL) exit(1);
    beta = (double *)calloc(K*T*N, sizeof(double));
    if (beta == NULL) exit(1);
    ggamma = (double *)calloc(K*T*N, sizeof(double));
    if (ggamma == NULL) exit(1);
    sigma = (double *)calloc(K*T*N*N, sizeof(double));
    if (sigma == NULL) exit(1);
    ggamma_sum = (double *)calloc(K*N, sizeof(double));
    if (ggamma_sum == NULL) exit(1);
    sigma_sum = (double *)calloc(K*N*N, sizeof(double));
    if (sigma_sum == NULL) exit(1);

    for (unsigned int i = 0; i < iterations; i++) {
        compute_baum_welch_iteration(K, N, M, T, observations, init_prob, trans_prob, emit_prob);
        memset(c_norm, 0, K*T*sizeof(double));
        memset(alpha, 0, K*T*N*sizeof(double));
        memset(beta, 0, K*T*N*sizeof(double));
        memset(ggamma, 0, K*T*N*sizeof(double));
        memset(sigma, 0, K*T*N*N*sizeof(double));
        memset(ggamma_sum, 0, K*N*sizeof(double));
        memset(sigma_sum, 0, K*N*N*sizeof(double));
    }

    free(c_norm);
    free(alpha);
    free(beta);
    free(ggamma);
    free(sigma);
    free(ggamma_sum);
    free(sigma_sum);

    return;
}