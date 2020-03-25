/*
    Baseline implementation
    Not verified yet!
*/

#include <cmath>

void forward(
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* init_prob,
    double* trans_prob,
    double* emit_prob,
    double* alpha
) {

    // t = 0, base case
    for (int n = 0; n < N; n++) {
        alpha[0*N + n] = init_prob[n]*emit_prob[n*M + observations[0]];
    }

    for (int t = 1; t < T; t++) {
        for (int n0 = 0; n0 < N; n0++) {

            for (int n1 = 0; n1 < N; n1++) {
                alpha[t*N + n0] += alpha[(t-1)*N + n1]*trans_prob[n1*N + n0];
            }

            alpha[t*N + n0] *= emit_prob[n0*M + observations[t]];

        }
    }

    return;
}


void backward(
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* trans_prob,
    double* emit_prob,
    double* beta
) {

    // t = T, base case
    for (int n = 0; n < N; n++) {
        beta[(T-1)*N + n] = 1.0;
    }

    for (int t = T-2; t >= 0; t--) {
        for (int n0 = 0; n0 < N; n0++) {
            for (int n1 = 0; n1 < N; n1++) {
                beta[t*N + n0] += beta[(t+1)*N + n1] * trans_prob[n0*N + n1] * emit_prob[n1*M + observations[t+1]];
            }
        }
    }

    return;
}


void update(
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* init_prob,
    double* trans_prob,
    double* emit_prob,
    double* alpha,
    double* beta,
    double* ggamma,
    double* sigma
) {

    // calculate ggamma
    for (int t = 0; t < T; t++) {

        double normalization = 0.0;

        for (int n = 0; n < N; n++) {
            normalization += alpha[t*N + n] * beta[t*N + n];
        }

        for (int n = 0; n < N; n++) {
            ggamma[t*N + n] = alpha[t*N + n] * beta[t*N + n] / normalization;
        }

    }

    // calculate sigma
    for (int t = 0; t < T-1; t++) {

        double normalization = 0.0;

        for (int n0 = 0; n0 < N; n0++) {
            for (int n1 = 0; n1 < N; n1++) {
                normalization += alpha[t*N + n0]*trans_prob[n0*N + n1]*beta[(t+1)*N + n1]*emit_prob[n1*M + observations[t+1]];
            }
        }

        for (int n0 = 0; n0 < N; n0++) {
            for (int n1 = 0; n1 < N; n1++) {
                sigma[(t*N + n0)*N + n1] = \
                    alpha[t*N + n0]*trans_prob[n0*N + n1]*beta[(t+1)*N + n1]*emit_prob[n1*M + observations[t+1]] / normalization;
            }
        }

    }

    // update init_prob
    for (int n = 0; n < N; n++) {
        init_prob[n] = ggamma[0*N + n];
    }

    // sum up sigma (from t = 0 to T-1)
    double sigma_sum[N][N];

    for (int n0 = 0; n0 < N; n0++) {
        for (int n1 = 0; n1 < N; n1++) {

            sigma_sum[n0][n1] = 0.0;

            for (int t = 0; t < T-1; t++) {
                sigma_sum[n0][n1] += sigma[(t*N + n0)*N + n1];
            }

        }
    }

    // sum up ggamma (from t = 0 to T-2)
    double ggamma_sum[N];

    for (int n = 0; n < N; n++) {

        ggamma_sum[n] = 0;

        for (int t = 0; t < T-1; t++) {
            ggamma_sum[n] += ggamma[t*N + n];
        }

    }

    // update trans_prob
    for (int n0 = 0; n0 < N; n0++) {
        for (int n1 = 0; n1 < N; n1++) {
            trans_prob[n0*N + n1] = sigma_sum[n0][n1] / ggamma_sum[n0];
        }
    }

    // add last T-step to ggamma_sum
    for (int n = 0; n < N; n++) {
        ggamma_sum[n] += ggamma[(T-1)*N + n];
    }

    // update emit_prob
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {

            double ggamma_cond_sum = 0;

            for (int t = 0; t < T; t++) {

                if (observations[t] == m) {
                    ggamma_cond_sum += ggamma[t*N + n];
                }

            }

            emit_prob[n*M + m] = ggamma_cond_sum / ggamma_sum[n];
        }
    }

    return;
}


void compute_baum_welch(
    unsigned int iterations,
    unsigned int N,
    unsigned int M,
    unsigned int T,
    unsigned int* observations,
    double* init_prob,
    double* trans_prob,
    double* emit_prob,
    double* alpha,
    double* beta,
    double* ggamma,
    double* sigma
    ) {

    for (unsigned int i = 0; i < iterations; i++) {
        forward(N, M, T, observations, init_prob, trans_prob, emit_prob, alpha);
        backward(N, M, T, observations, trans_prob, emit_prob, beta);
        update(N, M, T, observations, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
    }

    return;
}
