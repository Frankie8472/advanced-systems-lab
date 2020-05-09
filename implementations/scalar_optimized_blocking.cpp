/*
    Scalar optimized implementation
    No vectorization intrinsics here, otherwise optimize as you like!

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
size_t comp_bw_scalar_blocking(const BWdata& bw);

//variable for the innermost block size; must be smaller than min(N,K,T-2)
size_t innermost_block_size = 8;

size_t innermost_block_size_minus_one = innermost_block_size - 1;


REGISTER_FUNCTION(comp_bw_scalar_blocking, "Scalar Optimized: Blocking");


size_t comp_bw_scalar_blocking(const BWdata& bw){

    size_t iter = 0;
    double neg_log_likelihood_sum_old; // Does not have to be initialized as it will be if and only if i > 0

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {
        iter++;

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        //compute_sigma(bw);
        update_init_prob(bw);
        update_trans_prob(bw);
        update_emit_prob(bw);

        double neg_log_likelihood_sum = 0.0;
        for (size_t k = 0; k < bw.K; k++) {
            for (size_t t = 0; t < bw.T; t++) {
                neg_log_likelihood_sum = neg_log_likelihood_sum + log(bw.c_norm[k*bw.T + t]);
            }
        }
        bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;
    }

    return iter;
}


inline void forward_step(const BWdata& bw) {
    //Init
    double c_norm, alpha, alpha_sum, init_prob, emit_prob, trans_prob;
    double alpha2, alpha_sum2, emit_prob2, trans_prob2;
    double alpha3, alpha_sum3, emit_prob3, trans_prob3;
    double alpha4, alpha_sum4, emit_prob4, trans_prob4;

    size_t kTN, kT, n1_n11, n0_n00, n_n0;

    size_t n01, n02, n03;

    for (size_t k = 0; k < bw.K; k++) {
        // t = 0, base case

        // Init
        c_norm = 0;
        kT = k*bw.T;
        kTN = kT*bw.N;
        size_t observations = bw.observations[kT];

        size_t n = 0;
        for (; n < bw.N-innermost_block_size_minus_one; n+=innermost_block_size){

            for (size_t n0 = 0; n0 < innermost_block_size; n0++){
                n_n0 = n + n0;

                // Load
                init_prob = bw.init_prob[n_n0];
                emit_prob = bw.emit_prob[n_n0*bw.M + observations];

                // Calculate
                alpha = init_prob * emit_prob;
                c_norm += alpha;

                // Store
                bw.alpha[kTN + n_n0] = alpha;
            }
        }

        for (; n < bw.N; n++){
            // Load
            init_prob = bw.init_prob[n];
            emit_prob = bw.emit_prob[n*bw.M + observations];

            // Calculate
            alpha = init_prob * emit_prob;
            c_norm += alpha;

            // Store
            bw.alpha[kTN + n] = alpha;
        }

        // Calculate
        c_norm = 1.0/c_norm;

        n = 0;
        for (; n < bw.N-innermost_block_size_minus_one; n+=innermost_block_size){

            for (size_t n0 = 0; n0 < innermost_block_size; n0++){
                n_n0 = n + n0;

                // Load
                alpha = bw.alpha[kTN + n_n0];

                // Calculate
                alpha *= c_norm;

                // Store
                bw.alpha[kTN + n_n0] = alpha;
            }
        }

        for (; n < bw.N; n++){
            // Load
            alpha = bw.alpha[kTN + n];

            // Calculate
            alpha *= c_norm;

            // Store
            bw.alpha[kTN + n] = alpha;
        }

        // Store
        bw.c_norm[kT] = c_norm;

        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            c_norm = 0;
            observations = bw.observations[kT + t];
            kTN = (kT + t)*bw.N;

            for (size_t n0 = 0; n0 < bw.N; n0+=4) {
                n01 = n0 + 1;
                n02 = n0 + 2;
                n03 = n0 + 3;

                // Load
                alpha_sum = 0.0;
                alpha_sum2 = 0.0;
                alpha_sum3 = 0.0;
                alpha_sum4 = 0.0;
                emit_prob = bw.emit_prob[n0*bw.M + observations];
                emit_prob2 = bw.emit_prob[n01*bw.M + observations];
                emit_prob3 = bw.emit_prob[n02*bw.M + observations];
                emit_prob4 = bw.emit_prob[n03*bw.M + observations];

                size_t n1 = 0;
                for (; n1 < bw.N-innermost_block_size_minus_one; n1+=innermost_block_size) {

                    for (size_t n11 = 0; n11 < innermost_block_size; n11++){
                        n1_n11 = n1 + n11;

                        // Load
                        alpha = bw.alpha[kTN - bw.N + n1_n11];
                        trans_prob = bw.trans_prob[n1_n11*bw.N + n0];
                        trans_prob2 = bw.trans_prob[n1_n11*bw.N + n01];
                        trans_prob3 = bw.trans_prob[n1_n11*bw.N + n02];
                        trans_prob4 = bw.trans_prob[n1_n11*bw.N + n03];

                        // Calculate
                        alpha_sum += alpha * trans_prob;
                        alpha_sum2 += alpha * trans_prob2;
                        alpha_sum3 += alpha * trans_prob3;
                        alpha_sum4 += alpha * trans_prob4;
                    }
                }

                for (; n1 < bw.N; n1++){
                    // Load
                    alpha = bw.alpha[kTN - bw.N + n1];
                    trans_prob = bw.trans_prob[n1*bw.N + n0];
                    trans_prob2 = bw.trans_prob[n1*bw.N + n01];
                    trans_prob3 = bw.trans_prob[n1*bw.N + n02];
                    trans_prob4 = bw.trans_prob[n1*bw.N + n03];

                    // Calculate
                    alpha_sum += alpha * trans_prob;
                    alpha_sum2 += alpha * trans_prob2;
                    alpha_sum3 += alpha * trans_prob3;
                    alpha_sum4 += alpha * trans_prob4;
                }

                // Calculate
                alpha = emit_prob * alpha_sum;
                alpha2 = emit_prob2 * alpha_sum2;
                alpha3 = emit_prob3 * alpha_sum3;
                alpha4 = emit_prob4 * alpha_sum4;
                c_norm += alpha + alpha2 + alpha3 + alpha4;

                // Store
                bw.alpha[kTN + n0] = alpha;
                bw.alpha[kTN + n01] = alpha2;
                bw.alpha[kTN + n02] = alpha3;
                bw.alpha[kTN + n03] = alpha4;
            }

            // Calculate
            c_norm = 1.0/c_norm;

            size_t n0 = 0;
            for (; n0 < bw.N-innermost_block_size_minus_one; n0+=innermost_block_size) {

                for (size_t n00 = 0 ; n00 < innermost_block_size; n00++){
                    n0_n00 = n0 + n00;

                    // Load
                    alpha = bw.alpha[kTN + n0_n00];

                    // Calculate
                    alpha *= c_norm;

                    // Store
                    bw.alpha[kTN + n0_n00] = alpha;
                }
            }

            for (; n0 < bw.N; n0++){
                // Load
                alpha = bw.alpha[kTN + n0];

                // Calculate
                alpha *= c_norm;

                // Store
                bw.alpha[kTN + n0] = alpha;
            }

            // Store
            bw.c_norm[kT + t] = c_norm;
        }

    }
}


inline void backward_step(const BWdata& bw) {
    // Init
    double alpha, beta, beta_sum, c_norm, gamma, sigma, emit_prob, trans_prob;
    double alpha2, beta_sum2, trans_prob2;
    double alpha3, beta_sum3, trans_prob3;
    double alpha4, beta_sum4, trans_prob4;

    size_t observations, kTN, kTNN, kT, nN, n1_n10, n_n0;
    size_t kTNN2, nN2;
    size_t kTNN3, nN3;
    size_t kTNN4, nN4;

    size_t n01, n02, n03;

    for (size_t k = 0; k < bw.K; k++) {
        // t = bw.T, base case
        kTN = (k*bw.T + (bw.T-1))*bw.N;

        // Load
        c_norm = bw.c_norm[k*bw.T + (bw.T-1)];
        size_t n = 0;
        for (; n < bw.N-innermost_block_size_minus_one; n+=innermost_block_size) {

            for (size_t n0 = 0; n0 < innermost_block_size; n0++){
                n_n0 = n + n0;

                // Load
                alpha = bw.alpha[kTN + n_n0];

                // Store
                bw.beta[kTN + n_n0] = c_norm;
                bw.ggamma[kTN + n_n0] = alpha;
            }

        }

        for (; n < bw.N; n++){
            // Load
            alpha = bw.alpha[kTN + n];

            // Store
            bw.beta[kTN + n] = c_norm;
            bw.ggamma[kTN + n] = alpha;
        }

        // Recursion step
        kT = k*bw.T;
        for (int t = bw.T-2; t >= 0; t--) {
            // Load
            observations = bw.observations[kT + (t+1)];
            c_norm = bw.c_norm[kT + t];
            kTN = (kT + t) * bw.N;

            for (size_t n0 = 0; n0 < bw.N; n0+=4) {
                n01 = n0 + 1;
                n02 = n0 + 2;
                n03 = n0 + 3;

                // Load
                beta_sum = 0.0;
                beta_sum2 = 0.0;
                beta_sum3 = 0.0;
                beta_sum4 = 0.0;
                alpha = bw.alpha[(kT + t)*bw.N + n0];
                alpha2 = bw.alpha[(kT + t)*bw.N + n01];
                alpha3 = bw.alpha[(kT + t)*bw.N + n02];
                alpha4 = bw.alpha[(kT + t)*bw.N + n03];
                nN = n0 * bw.N;
                nN2 = (n01) * bw.N;
                nN3 = (n02) * bw.N;
                nN4 = (n03) * bw.N;
                kTNN = (kTN + n0)*bw.N;
                kTNN2 = (kTN + n01)*bw.N;
                kTNN3 = (kTN + n02)*bw.N;
                kTNN4 = (kTN + n03)*bw.N;

                size_t n1 = 0;
                for (; n1 < bw.N-innermost_block_size_minus_one; n1+=innermost_block_size) {

                    for (size_t n10 = 0; n10 < innermost_block_size; n10++){
                        n1_n10 = n1 + n10;

                        // Load
                        beta = bw.beta[kTN + bw.N + n1_n10];
                        trans_prob = bw.trans_prob[nN + n1_n10];
                        trans_prob2 = bw.trans_prob[nN2 + n1_n10];
                        trans_prob3 = bw.trans_prob[nN3 + n1_n10];
                        trans_prob4 = bw.trans_prob[nN4 + n1_n10];
                        emit_prob = bw.emit_prob[n1_n10 * bw.M + observations];

                        // Calculate & store
                        beta_sum += beta * trans_prob * emit_prob;
                        beta_sum2 += beta * trans_prob2 * emit_prob;
                        beta_sum3 += beta * trans_prob3 * emit_prob;
                        beta_sum4 += beta * trans_prob4 * emit_prob;
                        bw.sigma[kTNN + n1_n10] = alpha * trans_prob * beta * emit_prob;
                        bw.sigma[kTNN2 + n1_n10] = alpha2 * trans_prob2 * beta * emit_prob;
                        bw.sigma[kTNN3 + n1_n10] = alpha3 * trans_prob3 * beta * emit_prob;
                        bw.sigma[kTNN4 + n1_n10] = alpha4 * trans_prob4 * beta * emit_prob;
                    }
                }

                for (; n1 < bw.N; n1++){
                    // Load
                    beta = bw.beta[kTN + bw.N + n1];
                    trans_prob = bw.trans_prob[nN + n1];
                    trans_prob2 = bw.trans_prob[nN2 + n1];
                    trans_prob3 = bw.trans_prob[nN3 + n1];
                    trans_prob4 = bw.trans_prob[nN4 + n1];
                    emit_prob = bw.emit_prob[n1 * bw.M + observations];

                    // Calculate & store
                    beta_sum += beta * trans_prob * emit_prob;
                    beta_sum2 += beta * trans_prob2 * emit_prob;
                    beta_sum3 += beta * trans_prob3 * emit_prob;
                    beta_sum4 += beta * trans_prob4 * emit_prob;
                    bw.sigma[kTNN + n1] = alpha * trans_prob * beta * emit_prob;
                    bw.sigma[kTNN2 + n1] = alpha2 * trans_prob2 * beta * emit_prob;
                    bw.sigma[kTNN3 + n1] = alpha3 * trans_prob3 * beta * emit_prob;
                    bw.sigma[kTNN4 + n1] = alpha4 * trans_prob4 * beta * emit_prob;
                }

                // Calculate & store
                bw.beta[kTN + n0] = beta_sum * c_norm;
                bw.beta[kTN + n01] = beta_sum2 * c_norm;
                bw.beta[kTN + n02] = beta_sum3 * c_norm;
                bw.beta[kTN + n03] = beta_sum4 * c_norm;
                bw.ggamma[kTN + n0] = alpha * beta_sum;
                bw.ggamma[kTN + n01] = alpha2 * beta_sum2;
                bw.ggamma[kTN + n02] = alpha3 * beta_sum3;
                bw.ggamma[kTN + n03] = alpha4 * beta_sum4;
            }
        }
    }
}


inline void compute_gamma(const BWdata& bw) {
    // ====== Sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob) =====
    size_t kT, kTN, kN;

    for (size_t k = 0; k < bw.K; k++) {

        // Init
        double g_sum, s_sum;
        double s_sum2;
        double s_sum3;
        double s_sum4;

        size_t n11, n12, n13, t_t1;

        kT = k*bw.T;
        kTN = kT*bw.N;
        kN = k*bw.N;

        // TODO: Loop switch+blocking if N << T or vice versa i don't know
        for (size_t n0 = 0; n0 < bw.N; n0++) {

            // Init
            g_sum = bw.ggamma[kTN + n0];

            size_t t;
            for (t = 1; t < bw.T-1-innermost_block_size_minus_one; t+=innermost_block_size) {
                for (size_t t1 = 0; t1 < innermost_block_size; t1++){
                    // Calculation
                    g_sum += bw.ggamma[(kT + t + t1)*bw.N + n0];
                }
            }

            for (; t < bw.T-1; t++){
                // Calculation
                g_sum += bw.ggamma[(kT + t)*bw.N + n0];
            }

            // Store
            bw.gamma_sum[k*bw.N + n0] = g_sum;

            // TODO: Loop switch+blocking if N << T or vice versa i don't know
            for (size_t n1 = 0; n1 < bw.N; n1+=4) {
                n11 = n1 + 1;
                n12 = n1 + 2;
                n13 = n1 + 3;

                // Init
                s_sum = bw.sigma[(kTN + n0) * bw.N + n1];
                s_sum2 = bw.sigma[(kTN + n0) * bw.N + n11];
                s_sum3 = bw.sigma[(kTN + n0) * bw.N + n12];
                s_sum4 = bw.sigma[(kTN + n0) * bw.N + n13];

                for (t = 1; t < bw.T-1-innermost_block_size_minus_one; t+=innermost_block_size) {
                    for (size_t t1 = 0; t1 < innermost_block_size; t1++){
                        t_t1 = t + t1;
                        // Calculation
                        s_sum += bw.sigma[((kT + t_t1)*bw.N + n0) * bw.N + n1];
                        s_sum2 += bw.sigma[((kT + t_t1)*bw.N + n0) * bw.N + n11];
                        s_sum3 += bw.sigma[((kT + t_t1)*bw.N + n0) * bw.N + n12];
                        s_sum4 += bw.sigma[((kT + t_t1)*bw.N + n0) * bw.N + n13];
                    }
                }

                for (; t < bw.T-1; t++){
                    // Calculation
                    s_sum += bw.sigma[((kT + t)*bw.N + n0) * bw.N + n1];
                    s_sum2 += bw.sigma[((kT + t)*bw.N + n0) * bw.N + n11];
                    s_sum3 += bw.sigma[((kT + t)*bw.N + n0) * bw.N + n12];
                    s_sum4 += bw.sigma[((kT + t)*bw.N + n0) * bw.N + n13];
                }

                // Store
                bw.sigma_sum[(kN + n0) * bw.N + n1] = s_sum;
                bw.sigma_sum[(kN + n0) * bw.N + n11] = s_sum2;
                bw.sigma_sum[(kN + n0) * bw.N + n12] = s_sum3;
                bw.sigma_sum[(kN + n0) * bw.N + n13] = s_sum4;
            }
        }
    }
}


inline void compute_sigma(const BWdata& bw) {
    // Init
    double sigma, sigma_sum, alpha, beta, trans_prob, emit_prob;
    size_t observations, kTN, nN, kTNN, kNN, kT, kN;

    for (size_t k = 0; k < bw.K; k++) {

        kT = k*bw.T;
        kN = k*bw.N;
        /*
        for (size_t t = 0; t < bw.T-1; t++) {
            //Load
            observations = bw.observations[k*bw.T + (t+1)];

            for (size_t n0 = 0; n0 < bw.N; n0++) {
                // Load
                alpha = bw.alpha[(k*bw.T + t)*bw.N + n0];
                kTN = (kT + (t+1))*bw.N;
                nN = n0*bw.N ;
                kTNN = ((kT + t)*bw.N + n0)*bw.N;

                for (size_t n1 = 0; n1 < bw.N; n1++) {
                    // Load
                    beta = bw.beta[kTN + n1];
                    emit_prob = bw.emit_prob[n1*bw.M + observations];
                    trans_prob = bw.trans_prob[nN + n1];

                    // Calculate
                    sigma = alpha*trans_prob*beta*emit_prob;

                    // Store
                    bw.sigma[kTNN + n1] = sigma;
                }
            }
        }
        */
        // sum up bw.sigma (from t = 0 to bw.T-2)
        for (size_t n0 = 0; n0 < bw.N; n0++) {
            kNN = (kN + n0)*bw.N;
            for (size_t n1 = 0; n1 < bw.N; n1++) {
                // Init
                sigma_sum = 0.0;

                for (size_t t = 0; t < bw.T-1; t++) {
                    // Calculate
                    sigma_sum += bw.sigma[((kT + t) * bw.N + n0) * bw.N + n1];
                }

                // Store
                bw.sigma_sum[kNN + n1] = sigma_sum;
            }
        }
    }
}


inline void update_init_prob(const BWdata& bw) {
    // Init
    double g0_sum;
    double g0_sum2, g0_sum3, g0_sum4;

    size_t k_k0;
    size_t n1, n2, n3;

    for (size_t n = 0; n < bw.N; n+=4) {
        n1 = n + 1;
        n2 = n + 2;
        n3 = n + 3;

        // Init
        g0_sum = 0.0;
        g0_sum2 = 0.0;
        g0_sum3 = 0.0;
        g0_sum4 = 0.0;

        size_t k = 0;
        for (; k < bw.K-innermost_block_size_minus_one; k+=innermost_block_size) {

            for (size_t k0 = 0; k0 < innermost_block_size; k0++){
                k_k0 = k + k0;

                // Calculate
                g0_sum += bw.ggamma[k_k0*bw.T*bw.N + n];
                g0_sum2 += bw.ggamma[k_k0*bw.T*bw.N + n1];
                g0_sum3 += bw.ggamma[k_k0*bw.T*bw.N + n2];
                g0_sum4 += bw.ggamma[k_k0*bw.T*bw.N + n3];
            }
        }

        for (; k < bw.K; k++){
            // Calculate
            g0_sum += bw.ggamma[k*bw.T*bw.N + n];
            g0_sum2 += bw.ggamma[k*bw.T*bw.N + n1];
            g0_sum3 += bw.ggamma[k*bw.T*bw.N + n2];
            g0_sum4 += bw.ggamma[k*bw.T*bw.N + n3];
        }

        //Store
        bw.init_prob[n] = g0_sum/bw.K;
        bw.init_prob[n1] = g0_sum2/bw.K;
        bw.init_prob[n2] = g0_sum3/bw.K;
        bw.init_prob[n3] = g0_sum4/bw.K;
    }
}


inline void update_trans_prob(const BWdata& bw) {
    //Init
    double numerator_sum, denominator_sum;
    double numerator_sum2;
    double numerator_sum3;
    double numerator_sum4;

    size_t nN, k_k0;

    size_t n11, n12, n13;

    for (size_t n0 = 0; n0 < bw.N; n0++) {
        nN = n0*bw.N;

        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            n11 = n1 + 1;
            n12 = n1 + 2;
            n13 = n1 + 3;

            // Init
            numerator_sum = 0.0;
            numerator_sum2 = 0.0;
            numerator_sum3 = 0.0;
            numerator_sum4 = 0.0;

            denominator_sum = 0.0;

            size_t k = 0;
            for (; k < bw.K-innermost_block_size_minus_one; k+=innermost_block_size) {

                for (size_t k0 = 0; k0 < innermost_block_size; k0++){
                    k_k0 = k + k0;

                    // Calculate
                    numerator_sum += bw.sigma_sum[(k_k0*bw.N + n0)*bw.N + n1];
                    numerator_sum2 += bw.sigma_sum[(k_k0*bw.N + n0)*bw.N + n11];
                    numerator_sum3 += bw.sigma_sum[(k_k0*bw.N + n0)*bw.N + n12];
                    numerator_sum4 += bw.sigma_sum[(k_k0*bw.N + n0)*bw.N + n13];

                    denominator_sum += bw.gamma_sum[k_k0*bw.N + n0];
                }
            }

            for (; k < bw.K; k++){
                // Calculate
                numerator_sum += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1];
                numerator_sum2 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n11];
                numerator_sum3 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n12];
                numerator_sum4 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n13];

                denominator_sum += bw.gamma_sum[k*bw.N + n0];
            }

            // Store
            bw.trans_prob[nN + n1] = numerator_sum / denominator_sum;
            bw.trans_prob[nN + n11] = numerator_sum2 / denominator_sum;
            bw.trans_prob[nN + n12] = numerator_sum3 / denominator_sum;
            bw.trans_prob[nN + n13] = numerator_sum4 / denominator_sum;
        }
    }
}


inline void update_emit_prob(const BWdata& bw) {
    // Init
    double numerator_sum, denominator_sum, ggamma_cond_sum;
    size_t kTN, kN;
    size_t kTN2, kN2;
    size_t kTN3, kN3;
    size_t kTN4, kN4;

    size_t n_n0;
    size_t k1, k2, k3;

    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k+=4) {
        k1 = k + 1;
        k2 = k + 2;
        k3 = k + 3;

        kTN = (k*bw.T + (bw.T-1))*bw.N;
        kTN2 = (k1*bw.T + (bw.T-1))*bw.N;
        kTN3 = (k2*bw.T + (bw.T-1))*bw.N;
        kTN4 = (k3*bw.T + (bw.T-1))*bw.N;

        kN = k*bw.N;
        kN2 = k1*bw.N;
        kN3 = k2*bw.N;
        kN4 = k3*bw.N;

        size_t n = 0;
        for (; n < bw.N-innermost_block_size_minus_one; n+=innermost_block_size) {

            for (size_t n0 = 0; n0 < innermost_block_size; n0++){
                n_n0 = n + n0;

                bw.gamma_sum[kN + n_n0] += bw.ggamma[kTN + n_n0];
                bw.gamma_sum[kN2 + n_n0] += bw.ggamma[kTN2 + n_n0];
                bw.gamma_sum[kN3 + n_n0] += bw.ggamma[kTN3 + n_n0];
                bw.gamma_sum[kN4 + n_n0] += bw.ggamma[kTN4 + n_n0];
            }
        }

        for (; n < bw.N; n++){
            bw.gamma_sum[kN + n] += bw.ggamma[kTN + n];
            bw.gamma_sum[kN2 + n] += bw.ggamma[kTN2 + n];
            bw.gamma_sum[kN3 + n] += bw.ggamma[kTN3 + n];
            bw.gamma_sum[kN4 + n] += bw.ggamma[kTN4 + n];
        }
    }

    // update bw.emit_prob
    for (size_t m = 0; m < bw.M; m++) {
        for (size_t n = 0; n < bw.N; n++) {
            numerator_sum = 0.0;
            denominator_sum = 0.0;
            for (size_t k = 0; k < bw.K; k++) {
                ggamma_cond_sum = 0.0;
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
