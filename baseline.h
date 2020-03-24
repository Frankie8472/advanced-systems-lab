/*
    Baseline implementation
    Not verified yet!
*/

#include <cmath>

void forward(unsigned int N, unsigned int M, unsigned int T,
            unsigned int* obs, double* init_prob, double* trans_prob, double* emit_prob,
            double* alpha, double* beta, double* ggamma, double* sigma) {

    for (int i = 0; i < N; i++) {
        alpha[0*T + i] = init_prob[i]*emit_prob[obs[0]*N + i];
    }

    for (int t = 1; t < T; t++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                alpha[t*N + i] += alpha[(t-1)*N + j]*trans_prob[j*N + i];
            }
            alpha[t*N + i] *= emit_prob[obs[t]*N + i];
        }
    }
    return;
}


void backward(unsigned int N, unsigned int M, unsigned int T,
            unsigned int* obs, double* init_prob, double* trans_prob, double* emit_prob,
            double* alpha, double* beta, double* ggamma, double* sigma) {

    for (int i = 0; i < N; i++) {
        beta[(T-1)*N + i] = 1.0;
    }

    for (int t = T-2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                beta[t*N + i] += beta[(t+1)*N + j] * trans_prob[i*N + j] * emit_prob[obs[t+1]*N + j];
            }
        }
    }

    return;
}


void update(unsigned int N, unsigned int M, unsigned int T,
            unsigned int* obs, double* init_prob, double* trans_prob, double* emit_prob,
            double* alpha, double* beta, double* ggamma, double* sigma) {

    double norm;

    // calculate ggamma
    for (int t = 0; t < T; t++) {

        norm = 0.0;

        for (int i = 0; i < N; i++) {
            ggamma[t*N + i] = alpha[t*N + i] * beta[t*N + i];
            norm += alpha[t*N + i] * beta[t*N + i];
        }

        for (int i = 0; i < N; i++) {
            ggamma[t*N + i] /= norm;

        }
    }

    // calculate sigma
    for (int t = 0; t < T-1; t++) {

        norm = 0.0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j ++) {
                sigma[(t*N + i)*N + j] = alpha[t*N + i] + trans_prob[i*N + j]*beta[(t+1)*N + j]*emit_prob[obs[t+1]*N + j];
                norm += alpha[t*N + i]*trans_prob[i*N + j]*beta[(t+1)*N + j]*emit_prob[obs[t+1]*N + j];
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j ++) {
                sigma[(t*N + i)*N + j] /= norm;
            }
        }
    }

    // update init_prob
    for (int i = 0; i < N; i++) {
        init_prob[i] = ggamma[0*N + i];
    }

    // sum up sigma (from t = 0 to T -1)
    double sigma_sum[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j ++) {
            sigma_sum[i][j] = 0.0;
            for (int t = 0; t < T-1; t++) {
                sigma_sum[i][j] += sigma[(t*N + i)*N + j];
            }
        }
    }

    // sum up ggamma (from t = 0 to T-2)
    double ggamma_sum[N];

    for (int i = 0; i < N; i++) {
        ggamma_sum[i] = 0;
        for (int t = 0; t < T-1; t++) {
            ggamma_sum[i] += ggamma[t*N + i];
        }
    }

    // update trans_prob
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j ++) {
            trans_prob[i*N + j] = sigma_sum[i][j] / ggamma_sum[i];
        }
    }

    // add last Tstep to ggamma_sum
    for (int i = 0; i < N; i++) {
        ggamma_sum[i] += ggamma[(T-1)*N + i];
    }

    // update emit_prob
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {

            double sum = 0;

            for (int t = 0; t < T; t++) {
                if (obs[t] = i) sum += ggamma[t*N + j];
            }
            emit_prob[i*N + j] = sum / ggamma_sum[j];
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
        forward(N, M, T, observations, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
        backward(N, M, T, observations, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
        update(N, M, T, observations, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
    }

    return;
}
