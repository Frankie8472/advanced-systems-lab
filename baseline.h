/*
    (should be correct; but still needs more verification!)

    Baseline implementation
    Make sure you understand it!
    Everything taken from here : 
    https://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
*/

#include <cmath>
#include <cstring>

unsigned int K;  // number of observation sequences / training datasets
unsigned int N;  // number of hidden state variables
unsigned int M;  // number of observations
unsigned int T;  // number of time steps

// (for each oberservation/training sequence 0 <= k < K)
unsigned int* observations; //  [K][T]       [k][t]            := observation sequence k at time_step t
double* init_prob; //           [N]          [n]               := P(X_1 = n)
double* trans_prob; //          [N][N]       [n0][n1]          := P(X_t = n1 | X_(t-1) = n0)
double* emit_prob; //           [N][M]       [n][m]            := P(Y_t = y_m | X_t = n)
double* c_norm; //              [K][T]       [k][t]            :=  scaling/normalization factor for numerical stability
double* alpha; //               [K][T][N]    [k][t][n]         :=  P(Y_1 = y_1, ..., Y_t = y_t, X_t = n, theta)
double* beta; //                [K][T][N]    [k][t][n]         :=  P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = n, theta)
double* ggamma; //              [K][T][N]    [k][t][n]         :=  P(X_t = n | Y, theta)
double* sigma; //               [K][T][N][N] [k][t][n0][n1]    :=  P(X_t = n0, X_(t+1) = n1 | Y, theta)
// where theta = {init_prob, trans_prob, emit_prob} represent the model parameters we want learn/refine/estimate iteratively.
double* ggamma_sum; //          [K][N]
double* sigma_sum; //           [K][N][N]

void forward_step(void);
void backward_step(void);
void compute_gamma(void);
void compute_sigma(void);
void update_init_prob(void);
void update_trans_prob(void);
void update_emit_prob(void);

void compute_baum_welch(
    unsigned int max_iterations,
    unsigned int K_local,
    unsigned int N_local,
    unsigned int M_local,
    unsigned int T_local,
    unsigned int* observations_local,
    double* init_prob_local,
    double* trans_prob_local,
    double* emit_prob_local
    ) {
    
    K = K_local;
    N = N_local;
    M = M_local;
    T = T_local;
    observations = observations_local;
    init_prob = init_prob_local;
    trans_prob = trans_prob_local;
    emit_prob = emit_prob_local;

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

    for (int i = 0; i < max_iterations; i++) {

        forward_step();
        backward_step();
        compute_gamma();
        compute_sigma();
        update_init_prob();
        update_trans_prob();
        update_emit_prob();

        // TODO: compute log likelihood for convergence criterion
        //memset ...

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
}


void forward_step(void) {
    for (int k = 0; k < K; k++) {
        // t = 0, base case
        for (int n = 0; n < N; n++) {
            alpha[(k*T + 0)*N + n] = init_prob[n]*emit_prob[n*M + observations[k*T + 0]];
            c_norm[k*T + 0] += alpha[(k*T + 0)*N + n];
        } 

        c_norm[k*T + 0] = 1.0/c_norm[k*T + 0];
        for (int n = 0; n < N; n++){
	    alpha[(k*T + 0)*N + n] *= c_norm[k*T + 0];
	}

        // recursion step
        for (int t = 1; t < T; t++) {
            for (int n0 = 0; n0 < N; n0++) {
                double alpha_temp = 0.0;
                for (int n1 = 0; n1 < N; n1++) {
                    alpha_temp += alpha[(k*T + (t-1))*N + n1]*trans_prob[n1*N + n0];
                }
                alpha[(k*T + t)*N + n0] = emit_prob[n0*M + observations[k*T + t]] * alpha_temp;
                c_norm[k*T + t] += alpha[(k*T + t)*N + n0];
            }
            c_norm[k*T + t] = 1.0/c_norm[k*T + t];
            for (int n0 = 0; n0 < N; n0++) {
                alpha[(k*T + t)*N + n0] *= c_norm[k*T + t];
            }
        }
    }
}


void backward_step(void) {
    for (int k = 0; k < K; k++) {
        // t = T, base case
        for (int n = 0; n < N; n++) {
            beta[(k*T + (T-1))*N + n] = 1.0*c_norm[k*T + (T-1)];
        }

        // recursion step
        for (int t = T-2; t >= 0; t--) {
            for (int n0 = 0; n0 < N; n0++) {
                double beta_temp = 0.0;
                for (int n1 = 0; n1 < N; n1++) {
                    beta_temp += beta[(k*T + (t+1))*N + n1] * trans_prob[n0*N + n1] * emit_prob[n1*M + observations[k*T + (t+1)]];
                }
                beta[(k*T + t)*N + n0] = beta_temp * c_norm[k*T + t];
            }
        }
    }
}


void compute_gamma(void) {
    for (int k = 0; k < K; k++) {
        for (int t = 0; t < T; t++) {
            for (int n = 0; n < N; n++) {
                ggamma[(k*T + t)*N + n] = alpha[(k*T + t)*N + n] * beta[(k*T + t)*N + n] / c_norm[k*T + t];
            }
        }
    }

    // sum up ggamma (from t = 0 to T-2; serve as normalizer for trans_prob)
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            double tmp_sum = 0.0;
            for (int t = 0; t < T-1; t++) {
                tmp_sum += ggamma[(k*T + t)*N + n];
            }
            ggamma_sum[k*N + n] = tmp_sum;
        }
    }
}


void compute_sigma(void) {
    for (int k = 0; k < K; k++) {
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
}


void update_init_prob(void) {
    for (int n = 0; n < N; n++) {
        double ggamma_sum = 0.0;
        for (int k = 0; k < K; k++) {
            ggamma_sum += ggamma[(k*T + 0)*N + n];
        }
        init_prob[n] = ggamma_sum/K;
    }
}


void update_trans_prob(void) {
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
}


void update_emit_prob(void) {
    // add last T-step to ggamma_sum
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            ggamma_sum[k*N + n] += ggamma[(k*T + (T-1))*N + n];
        }
    }
    // update emit_prob
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
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
}
