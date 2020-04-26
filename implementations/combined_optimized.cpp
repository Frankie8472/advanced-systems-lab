/*
    Best optimized implementation
    Final and best possible optimization, combination of all previous approaches!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#include <cmath>
#include <cstring>

#include "../common.h"


void forward_step(const BWdata& bw);
void backward_step(const BWdata& bw);
void compute_gamma(const BWdata& bw);
void compute_sigma(const BWdata& bw);
void update_init_prob(const BWdata& bw);
void update_trans_prob(const BWdata& bw);
void update_emit_prob(const BWdata& bw);
size_t comp_bw_combined(const BWdata& bw);


REGISTER_FUNCTION(comp_bw_combined, "TODO: Combined Optimized");


size_t comp_bw_combined(const BWdata& bw){

    size_t iter = 0;
    for (size_t i = 0; i < bw.max_iterations; i++) {
        iter++;

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        compute_sigma(bw);
        update_init_prob(bw);
        update_trans_prob(bw);
        update_emit_prob(bw);

        double neg_log_likelihood_sum = 0.0;
        double neg_log_likelihood_sum_old; // Does not have to be initialized as it will be if and only if i > 0

        for (size_t k = 0; k < bw.K; k++) {
            for (size_t t = 0; t < bw.T; t++) {
                neg_log_likelihood_sum = neg_log_likelihood_sum - log(bw.c_norm[k*bw.T + t]);
            }
        }
        bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;

        // convergence criterion
        if (i > 0 && std::abs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < 1e-3) break;

        neg_log_likelihood_sum_old = neg_log_likelihood_sum;
        //print_states(bw);
    }

    return iter;
}


inline void forward_step(const BWdata& bw) {
    for (size_t k = 0; k < bw.K; k++) {
        // t = 0, base case

        // Init
        double c_norm = 0;
        double alpha, alpha_sum, init_prob, emit_prob, trans_prob;
        size_t observations = bw.observations[k*bw.T + 0];

        for (size_t n = 0; n < bw.N; n++) {
            // Load
            init_prob = bw.init_prob[n];
            emit_prob = bw.emit_prob[n*bw.M + observations];

            // Calculate
            alpha = init_prob*emit_prob;
            c_norm += alpha;

            // Store
            bw.alpha[(k*bw.T + 0)*bw.N + n] = alpha;
        }

        // Calculate
        c_norm = 1.0/c_norm;

        for (size_t n = 0; n < bw.N; n++){
            // Load
            alpha = bw.alpha[(k*bw.T + 0)*bw.N + n];

            // Calculate
            alpha *= c_norm;

            // Store
            bw.alpha[(k*bw.T + 0)*bw.N + n] = alpha;
        }

        // Store
        bw.c_norm[k*bw.T + 0] = c_norm;

        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            c_norm = 0;
            observations = bw.observations[k*bw.T + t];

            for (size_t n0 = 0; n0 < bw.N; n0++) {
                // Load
                alpha_sum = 0.0;
                emit_prob = bw.emit_prob[n0*bw.M + observations];

                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    // Load
                    alpha = bw.alpha[(k*bw.T + (t-1))*bw.N + n1];
                    trans_prob = bw.trans_prob[n1*bw.N + n0];

                    // Calculate
                    alpha_sum += alpha*trans_prob;
                }

                // Calculate
                alpha = emit_prob * alpha_sum;
                c_norm += alpha;

                // Store
                bw.alpha[(k*bw.T + t)*bw.N + n0] = alpha;
            }

            c_norm = 1.0/c_norm;

            for (size_t n0 = 0; n0 < bw.N; n0++) {
                // Load
                alpha = bw.alpha[(k*bw.T + t)*bw.N + n0];

                // Calculate
                alpha *= c_norm;

                // Store
                bw.alpha[(k*bw.T + t)*bw.N + n0] = alpha;
            }

            // Store
            bw.c_norm[k*bw.T + t] = c_norm;
        }

    }
}


inline void backward_step(const BWdata& bw) {
    for (size_t k = 0; k < bw.K; k++) {
        // t = bw.T, base case

        // Init
        double c_norm = 0;
        double beta, beta_sum, init_prob, emit_prob, trans_prob;
        size_t observations = bw.observations[k*bw.T + 0];

        for (size_t n = 0; n < bw.N; n++) {
            bw.beta[(k*bw.T + (bw.T-1))*bw.N + n] = bw.c_norm[k*bw.T + (bw.T-1)];
        }

        // recursion step
        for (int t = bw.T-2; t >= 0; t--) {
            // Load
            observations = bw.observations[k*bw.T + (t+1)];
            c_norm = bw.c_norm[k * bw.T + t];

            for (size_t n0 = 0; n0 < bw.N; n0++) {
                // Init
                beta_sum = 0.0;

                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    // Load
                    beta = bw.beta[(k * bw.T + (t + 1)) * bw.N + n1];
                    trans_prob = bw.trans_prob[n0 * bw.N + n1];
                    emit_prob = bw.emit_prob[n1 * bw.M + observations];

                    // Calculate
                    beta_sum += beta * trans_prob * emit_prob;
                }

                // Calculate
                beta = beta_sum * c_norm;

                // Store
                bw.beta[(k*bw.T + t)*bw.N + n0] = beta;
            }
        }
    }
}


inline void compute_gamma(const BWdata& bw) {
    // Init
    double c_norm, alpha, beta, gamma;

    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T; t++) {
            // Load
            c_norm = bw.c_norm[k*bw.T + t];

            for (size_t n = 0; n < bw.N; n++) {
                // Load
                alpha = bw.alpha[(k*bw.T + t)*bw.N + n];
                beta = bw.beta[(k*bw.T + t)*bw.N + n];

                // Calculate
                gamma = alpha * beta / c_norm;

                // Store
                bw.ggamma[(k*bw.T + t)*bw.N + n] = gamma;
            }
        }
    }

    // sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob)
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            // Init
            double g_sum = 0.0;

            for (size_t t = 0; t < bw.T-1; t++) {
                // Calculate
                g_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];
            }

            // Store
            bw.gamma_sum[k*bw.N + n] = g_sum;
        }
    }
}


inline void compute_sigma(const BWdata& bw) {
    // Init
    double sigma, sigma_sum, alpha, beta, trans_prob, emit_prob;
    size_t observations;

    for (size_t k = 0; k < bw.K; k++) {
        for (size_t t = 0; t < bw.T-1; t++) {
            //Load
            observations = bw.observations[k*bw.T + (t+1)];

            for (size_t n0 = 0; n0 < bw.N; n0++) {
                // Load
                alpha = bw.alpha[(k*bw.T + t)*bw.N + n0];

                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    // Load
                    beta = bw.beta[(k*bw.T + (t+1))*bw.N + n1];
                    emit_prob = bw.emit_prob[n1*bw.M + observations];
                    trans_prob = bw.trans_prob[n0*bw.N + n1];

                    // Calculate
                    sigma = alpha*trans_prob*beta*emit_prob;

                    // Store
                    bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1] = sigma;
                }
            }
        }

        // sum up bw.sigma (from t = 0 to bw.T-1)
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                // Init
                sigma_sum = 0.0;

                for (size_t t = 0; t < bw.T-1; t++) {
                    // Calculate
                    sigma_sum += bw.sigma[((k * bw.T + t) * bw.N + n0) * bw.N + n1];
                }

                // Store
                bw.sigma_sum[(k*bw.N + n0)*bw.N + n1] = sigma_sum;
            }
        }
    }
}


inline void update_init_prob(const BWdata& bw) {
    for (size_t n = 0; n < bw.N; n++) {
        double g0_sum = 0.0;
        for (size_t k = 0; k < bw.K; k++) {
            g0_sum += bw.ggamma[(k*bw.T + 0)*bw.N + n];
        }
        bw.init_prob[n] = g0_sum/bw.K;
    }
}


inline void update_trans_prob(const BWdata& bw) {
    for (size_t n0 = 0; n0 < bw.N; n0++) {
        for (size_t n1 = 0; n1 < bw.N; n1++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                numerator_sum += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1];
                denominator_sum += bw.gamma_sum[k*bw.N + n0];
            }
            bw.trans_prob[n0*bw.N + n1] = numerator_sum / denominator_sum;
        }
    }
}


inline void update_emit_prob(const BWdata& bw) {
    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n++) {
            bw.gamma_sum[k*bw.N + n] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n];
        }
    }
    // update bw.emit_prob
    for (size_t n = 0; n < bw.N; n++) {
        for (size_t m = 0; m < bw.M; m++) {
            double numerator_sum = 0.0;
            double denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                double ggamma_cond_sum = 0.0;
                for (size_t t = 0; t < bw.T; t++) {
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
