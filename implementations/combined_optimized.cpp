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
void backward_step(const BWdata& bw, const size_t& k);
void compute_gamma(const BWdata& bw, const size_t& k);
void compute_sigma(const BWdata& bw);
void update_init_prob(const BWdata& bw);
void update_trans_prob(const BWdata& bw);
void update_emit_prob(const BWdata& bw);
size_t comp_bw_combined(const BWdata& bw);


REGISTER_FUNCTION_TRANSPOSE_EMIT_PROB(comp_bw_combined, "combined", "Combined Optimized");

static double* denominator_sum = nullptr;
static double* numerator_sum = nullptr;

size_t comp_bw_combined(const BWdata& bw){
    size_t res = 0;
    double neg_log_likelihood_sum, neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;

    denominator_sum = (double *)aligned_alloc(32,bw.N * sizeof(double));
    numerator_sum = (double *)aligned_alloc(32,bw.N*bw.M * sizeof(double));
    assert(denominator_sum != nullptr && "Failed to allocate memory");
    assert(numerator_sum != nullptr && "Failed to allocate memory");

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {
        neg_log_likelihood_sum = 0.0;
        
        forward_step(bw);
        for (size_t k = 0; k < bw.K; k++) {
            backward_step(bw, k);
            compute_gamma(bw, k);
        }
        
        for (size_t k = 0; k < bw.K; k++) {
            for (size_t t = 0; t < bw.T; t++) {
                neg_log_likelihood_sum = neg_log_likelihood_sum + log(bw.c_norm[k*bw.T + t]);
            }
        }
        bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;

        if (first && i > 0 && fabs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < EPSILON){
            first = false;
            res = i;
        }

        neg_log_likelihood_sum_old = neg_log_likelihood_sum;
        
        
        update_trans_prob(bw);
        update_emit_prob(bw);
    }

    free(denominator_sum);
    free(numerator_sum);

    return res;
}


inline void forward_step(const BWdata& bw) {
    //Init
    __m256d init_prob, emit_prob, alpha, alpha_sum, c_norm_v, trans_prob;
    //__m256d emit_prob0, alpha0, c_norm_v0, alpha_sum0, trans_prob0;
    //__m256d emit_prob1, alpha1, c_norm_v1, alpha_sum1, trans_prob1;
    //__m256d emit_prob2, alpha2, c_norm_v2, alpha_sum2, trans_prob2;
    //__m256d emit_prob3, alpha3, c_norm_v3, alpha_sum3, trans_prob3;
    //__m256d sum_01, sum_23, blended, permuted;
    __m128d vlow, vhigh;
    double c_norm;

    size_t observations0, observations1, observations2, observations3;

    size_t kTN, kT;
    // t = 0, base case

    // Init
    // TODO: Unroll k
    for(size_t k=0; k < bw.K; k++){
        c_norm_v = _mm256_setzero_pd();
        kT = k*bw.T;
        
        size_t observations = bw.observations[k*bw.T];
    
        for (size_t n = 0; n < bw.N; n+=4){
            // Load
            init_prob = _mm256_load_pd(bw.init_prob + n);
            emit_prob = _mm256_load_pd(bw.emit_prob + observations*bw.N + n);
    
            // Calculate
            alpha = _mm256_mul_pd(init_prob, emit_prob);
            c_norm_v = _mm256_fmadd_pd(init_prob, emit_prob, c_norm_v);
    
            // Store
            _mm256_store_pd(bw.alpha + k*bw.T*bw.N + n, alpha);
        }
    
        // Calculate
        vlow  = _mm256_castpd256_pd128(c_norm_v);
        vhigh = _mm256_extractf128_pd(c_norm_v, 1); // high 128
        vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
//
        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        c_norm = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
        c_norm = 1.0/c_norm;
        c_norm_v = _mm256_set1_pd(c_norm);

        for (size_t n = 0; n < bw.N; n+=4){
            // Load
            alpha = _mm256_load_pd(bw.alpha + k*bw.T*bw.N + n);
    
            // Calculate
            alpha = _mm256_mul_pd(alpha, c_norm_v);
    
            // Store
            _mm256_store_pd(bw.alpha + k*bw.T*bw.N + n, alpha);
        }
    
        // Store
        bw.c_norm[kT] = c_norm;
        
        // recursion step
        for (size_t t = 1; t < bw.T; t++) {
            c_norm_v = _mm256_setzero_pd();
            observations = bw.observations[kT + t];
            kTN = (kT + t)*bw.N;
    
            for (size_t n0 = 0; n0 < bw.N; n0+=4) {
    
                // Load
                alpha_sum = _mm256_setzero_pd();
    
                for (size_t n1 = 0; n1 < bw.N; n1++) {
    
                    // Load
                    alpha = _mm256_broadcast_sd(bw.alpha + kTN - bw.N + n1);
                    trans_prob = _mm256_load_pd(bw.trans_prob + n1*bw.N + n0);
    
                    // Calculate
                    alpha_sum = _mm256_fmadd_pd(alpha, trans_prob, alpha_sum);
                }
    
                emit_prob = _mm256_load_pd(bw.emit_prob + observations*bw.N + n0);
                // Calculate
                alpha = _mm256_mul_pd(alpha_sum, emit_prob);
                c_norm_v = _mm256_fmadd_pd(alpha_sum, emit_prob, c_norm_v);
    
                // Store
                _mm256_store_pd(bw.alpha + kTN + n0, alpha);
            }
    
            // Calculate
            vlow  = _mm256_castpd256_pd128(c_norm_v);
            vhigh = _mm256_extractf128_pd(c_norm_v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
    
            __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
            c_norm = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
            c_norm = 1.0/c_norm;
            c_norm_v = _mm256_set1_pd(c_norm);
    
            for (size_t n = 0; n < bw.N; n+=4){
                // Load
                alpha = _mm256_load_pd(bw.alpha + kTN + n);
        
                // Calculate
                alpha = _mm256_mul_pd(alpha, c_norm_v);
        
                // Store
                _mm256_store_pd(bw.alpha + kTN + n, alpha);
            }
    
            // Store
            bw.c_norm[kT + t] = c_norm;
        }
    }
}


inline void backward_step(const BWdata& bw, const size_t& k) {
    // Init
    __m256d c_norm, beta, gamma, emit_prob, beta_emit_prob, alpha;
    __m256d beta_sum0, beta_temp0, trans_prob0, alpha0;
    __m256d beta_sum1, beta_temp1, trans_prob1, alpha1;
    __m256d beta_sum2, beta_temp2, trans_prob2, alpha2;
    __m256d beta_sum3, beta_temp3, trans_prob3, alpha3;

    size_t observations, kTN, kT, nN;
    // t = bw.T, base case
    kTN = (k*bw.T + (bw.T-1))*bw.N;

    // Load
    memcpy(bw.ggamma + kTN, bw.alpha + kTN, bw.N * sizeof(double));
    c_norm = _mm256_broadcast_sd(bw.c_norm + k*bw.T + (bw.T-1));
    for (size_t n = 0; n < bw.N; n+=4) {
        // Store
        _mm256_store_pd(bw.beta + kTN + n, c_norm);
    }

    // Recursion step
    kT = k*bw.T;
    for (int t = bw.T-2; t >= 0; t--) {
        // Load
        observations = bw.observations[kT + (t+1)];
        c_norm = _mm256_broadcast_sd(bw.c_norm + kT + t);
        kTN = (kT+t)*bw.N;

        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            beta = _mm256_load_pd(bw.beta + (kT+t)*bw.N + bw.N + n1);
            emit_prob = _mm256_load_pd(bw.emit_prob + observations*bw.N + n1);
            beta_emit_prob = _mm256_mul_pd(beta, emit_prob);
            _mm256_store_pd(denominator_sum + n1, beta_emit_prob);
        }

        for (size_t n0 = 0; n0 < bw.N; n0+=4) {

            // Load
            beta_sum0 = _mm256_setzero_pd();
            beta_sum1 = _mm256_setzero_pd();
            beta_sum2 = _mm256_setzero_pd();
            beta_sum3 = _mm256_setzero_pd();
            alpha0 = _mm256_broadcast_sd(bw.alpha + (kT + t)*bw.N + n0 + 0);
            alpha1 = _mm256_broadcast_sd(bw.alpha + (kT + t)*bw.N + n0 + 1);
            alpha2 = _mm256_broadcast_sd(bw.alpha + (kT + t)*bw.N + n0 + 2);
            alpha3 = _mm256_broadcast_sd(bw.alpha + (kT + t)*bw.N + n0 + 3);

            for (size_t n1 = 0; n1 < bw.N; n1+=4) {
                // Load
                beta_emit_prob = _mm256_load_pd(denominator_sum + n1);

                trans_prob0 = _mm256_load_pd(bw.trans_prob + (n0+0) * bw.N + n1);
                trans_prob1 = _mm256_load_pd(bw.trans_prob + (n0+1) * bw.N + n1);
                trans_prob2 = _mm256_load_pd(bw.trans_prob + (n0+2) * bw.N + n1);
                trans_prob3 = _mm256_load_pd(bw.trans_prob + (n0+3) * bw.N + n1);

                beta_temp0 = _mm256_mul_pd(beta_emit_prob, trans_prob0);
                beta_temp1 = _mm256_mul_pd(beta_emit_prob, trans_prob1);
                beta_temp2 = _mm256_mul_pd(beta_emit_prob, trans_prob2);
                beta_temp3 = _mm256_mul_pd(beta_emit_prob, trans_prob3);

                beta_sum0 = _mm256_fmadd_pd(beta_emit_prob, trans_prob0, beta_sum0);
                beta_sum1 = _mm256_fmadd_pd(beta_emit_prob, trans_prob1, beta_sum1);
                beta_sum2 = _mm256_fmadd_pd(beta_emit_prob, trans_prob2, beta_sum2);
                beta_sum3 = _mm256_fmadd_pd(beta_emit_prob, trans_prob3, beta_sum3);

                beta_temp0 = _mm256_mul_pd(alpha0, beta_temp0);
                beta_temp1 = _mm256_mul_pd(alpha1, beta_temp1);
                beta_temp2 = _mm256_mul_pd(alpha2, beta_temp2);
                beta_temp3 = _mm256_mul_pd(alpha3, beta_temp3);

                _mm256_store_pd(bw.sigma + (kTN + n0+0)*bw.N + n1, beta_temp0);
                _mm256_store_pd(bw.sigma + (kTN + n0+1)*bw.N + n1, beta_temp1);
                _mm256_store_pd(bw.sigma + (kTN + n0+2)*bw.N + n1, beta_temp2);
                _mm256_store_pd(bw.sigma + (kTN + n0+3)*bw.N + n1, beta_temp3);
            }

            // Calculate & store
            __m256d sum_01 = _mm256_hadd_pd(beta_sum0, beta_sum1);
            __m256d sum_23 = _mm256_hadd_pd(beta_sum2, beta_sum3);

            __m256d blended = _mm256_blend_pd(sum_01, sum_23, 0b1100);

            __m256d permuted = _mm256_permute2f128_pd(sum_01, sum_23, 0b00100001);

            __m256d tmp = _mm256_add_pd(blended, permuted);
            beta = _mm256_mul_pd(tmp, c_norm);
            alpha = _mm256_load_pd(bw.alpha + (kT + t)*bw.N + n0);
            gamma = _mm256_mul_pd(tmp, alpha);
            _mm256_store_pd(bw.beta + kTN + n0, beta);
            _mm256_store_pd(bw.ggamma + kTN + n0, gamma);
        }
    }
}

inline void compute_gamma(const BWdata& bw, const size_t& k) {
    __m256d gamma, g_sum;
    __m256d sigma0, sigma1, sigma2, sigma3, s_sum0, s_sum1, s_sum2, s_sum3, s_sum4;
    
    for (size_t n0 = 0; n0 < bw.N; n0+=4) {
        // blocking here if you want to include n1 in this loop instead of after this loop
        g_sum = _mm256_load_pd(bw.ggamma + (k*bw.T + 0)*bw.N + n0);
        for (size_t t = 1; t < bw.T-1; t++) {
            gamma = _mm256_load_pd(bw.ggamma + (k*bw.T + t)*bw.N + n0);
            g_sum = _mm256_add_pd(g_sum, gamma);
        }
        // Store
        _mm256_store_pd(bw.gamma_sum + k*bw.N + n0, g_sum);

        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            s_sum0 = _mm256_load_pd(bw.sigma + ((k*bw.T + 0)*bw.N + n0+0) * bw.N + n1);
            s_sum1 = _mm256_load_pd(bw.sigma + ((k*bw.T + 0)*bw.N + n0+1) * bw.N + n1);
            s_sum2 = _mm256_load_pd(bw.sigma + ((k*bw.T + 0)*bw.N + n0+2) * bw.N + n1);
            s_sum3 = _mm256_load_pd(bw.sigma + ((k*bw.T + 0)*bw.N + n0+3) * bw.N + n1);

            for (size_t t = 1; t < bw.T-1; t++) {
                // Calculation
                sigma0 = _mm256_load_pd(bw.sigma + ((k*bw.T + t)*bw.N + n0+0) * bw.N + n1);
                sigma1 = _mm256_load_pd(bw.sigma + ((k*bw.T + t)*bw.N + n0+1) * bw.N + n1);
                sigma2 = _mm256_load_pd(bw.sigma + ((k*bw.T + t)*bw.N + n0+2) * bw.N + n1);
                sigma3 = _mm256_load_pd(bw.sigma + ((k*bw.T + t)*bw.N + n0+3) * bw.N + n1);
                
                s_sum0 = _mm256_add_pd(s_sum0, sigma0);
                s_sum1 = _mm256_add_pd(s_sum1, sigma1);
                s_sum2 = _mm256_add_pd(s_sum2, sigma2);
                s_sum3 = _mm256_add_pd(s_sum3, sigma3);
            }

            // Store
            _mm256_store_pd(bw.sigma_sum + (k*bw.N + n0+0) * bw.N + n1, s_sum0);
            _mm256_store_pd(bw.sigma_sum + (k*bw.N + n0+1) * bw.N + n1, s_sum1);
            _mm256_store_pd(bw.sigma_sum + (k*bw.N + n0+2) * bw.N + n1, s_sum2);
            _mm256_store_pd(bw.sigma_sum + (k*bw.N + n0+3) * bw.N + n1, s_sum3);
        }
    }
}

inline void update_trans_prob(const BWdata& bw) {
    //Init (init_prob)
    __m256d ones, gamma_sum, gamma, g0_sum, denominator_sum_n, K_inv, numerator_sum, sigma_sum, denominator_sum_n0;

    K_inv = _mm256_set1_pd(1.0/bw.K);
    ones = _mm256_set1_pd(1.0);

    for (size_t n = 0; n < bw.N; n+=4) {
        denominator_sum_n = _mm256_load_pd(bw.gamma_sum + 0*bw.N + n);
        g0_sum = _mm256_load_pd(bw.ggamma + (0*bw.T)*bw.N + n);

        for (size_t k = 1; k < bw.K; k++) {
            gamma_sum = _mm256_load_pd(bw.gamma_sum + k*bw.N + n);
            denominator_sum_n = _mm256_add_pd(denominator_sum_n, gamma_sum);

            gamma = _mm256_load_pd(bw.ggamma + (k*bw.T)*bw.N + n);
            g0_sum = _mm256_add_pd(gamma, g0_sum);
        }

        denominator_sum_n = _mm256_div_pd(ones, denominator_sum_n);
        _mm256_store_pd(denominator_sum + n, denominator_sum_n);

        g0_sum = _mm256_mul_pd(g0_sum, K_inv);
        _mm256_store_pd(bw.init_prob + n, g0_sum);
    }

    for (size_t n0 = 0; n0 < bw.N; n0++) {
        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            // Init (trans_prob)
            numerator_sum = _mm256_load_pd(bw.sigma_sum + (0*bw.N + n0)*bw.N + n1);

            for (size_t k = 1; k < bw.K; k++) {
                // Calculate (trans_prob)
                sigma_sum = _mm256_load_pd(bw.sigma_sum + (k*bw.N + n0)*bw.N + n1);
                numerator_sum = _mm256_add_pd(numerator_sum, sigma_sum);
            }

            // Store (trans_prob)
            denominator_sum_n0 = _mm256_broadcast_sd(denominator_sum + n0);
            numerator_sum = _mm256_mul_pd(numerator_sum, denominator_sum_n0);
            _mm256_store_pd(bw.trans_prob + n0*bw.N + n1, numerator_sum);
        }
    }
}

inline void update_emit_prob(const BWdata& bw) {
    // Init
    __m256d ggamma, gamma_sum, gamma_sum0, gamma_sum1, denominator_sum0, denominator_sum1;
    __m256d ones, ggamma_cond_sum_tot, ggamma_cond_sum, denominator_sum_inv;
    __m256d denominator_sum_n0, numerator_sum_n0;
    __m256d denominator_sum_n1, numerator_sum_n1;
    ones = _mm256_set1_pd(1.0);

    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n+=4) {
            ggamma = _mm256_load_pd(bw.ggamma + (k*bw.T + (bw.T-1))*bw.N + n);
            gamma_sum = _mm256_load_pd(bw.gamma_sum + k*bw.N + n);
            gamma_sum = _mm256_add_pd(gamma_sum, ggamma);
            _mm256_store_pd(bw.gamma_sum + k*bw.N + n, gamma_sum);
        }
    }

    // denominator_sum (top-down)
    for (size_t n = 0; n < bw.N; n += 8){
        denominator_sum0 = _mm256_load_pd(bw.gamma_sum + 0*bw.N + n+0);
        denominator_sum1 = _mm256_load_pd(bw.gamma_sum + 0*bw.N + n+4);

        for (size_t k = 1; k < bw.K; k++) {
            gamma_sum0 = _mm256_load_pd(bw.gamma_sum + k*bw.N + n+0);
            gamma_sum1 = _mm256_load_pd(bw.gamma_sum + k*bw.N + n+4);

            denominator_sum0 = _mm256_add_pd(denominator_sum0, gamma_sum0);
            denominator_sum1 = _mm256_add_pd(denominator_sum1, gamma_sum1);
        }
        denominator_sum0 = _mm256_div_pd(ones, denominator_sum0);
        denominator_sum1 = _mm256_div_pd(ones, denominator_sum1);
        _mm256_store_pd(denominator_sum + n+0, denominator_sum0);
        _mm256_store_pd(denominator_sum + n+4, denominator_sum1);
    }

    // numerator_sum
    for (size_t m = 0; m < bw.M; m++) {
        for (size_t n = 0; n < bw.N; n+=4) {
            ggamma_cond_sum_tot = _mm256_setzero_pd();

            for (size_t k = 0; k < bw.K; k++) {
                ggamma_cond_sum = _mm256_setzero_pd();

                for (size_t t = 0; t < bw.T; t+=4) {
                    if (bw.observations[k*bw.T + t] == m) {
                        ggamma = _mm256_load_pd(bw.ggamma + (k*bw.T + t)*bw.N + n);
                        ggamma_cond_sum = _mm256_add_pd(ggamma_cond_sum, ggamma);
                    }
                    if (bw.observations[k*bw.T + t+1] == m) {
                        ggamma = _mm256_load_pd(bw.ggamma + (k*bw.T + t+1)*bw.N + n);
                        ggamma_cond_sum = _mm256_add_pd(ggamma_cond_sum, ggamma);
                    }
                    if (bw.observations[k*bw.T + t+2] == m) {
                        ggamma = _mm256_load_pd(bw.ggamma + (k*bw.T + t+2)*bw.N + n);
                        ggamma_cond_sum = _mm256_add_pd(ggamma_cond_sum, ggamma);
                    }

                    if (bw.observations[k*bw.T + t+3] == m) {
                        ggamma = _mm256_load_pd(bw.ggamma + (k*bw.T + t+3)*bw.N + n);
                        ggamma_cond_sum = _mm256_add_pd(ggamma_cond_sum, ggamma);
                    }
                }
                ggamma_cond_sum_tot = _mm256_add_pd(ggamma_cond_sum_tot, ggamma_cond_sum);

            }
            _mm256_store_pd(numerator_sum + m*bw.N + n, ggamma_cond_sum_tot);
        }
    }

    // emit_prob
    for (size_t n = 0; n < bw.N; n+=8) {
        denominator_sum_n0 = _mm256_load_pd(denominator_sum + n);
        denominator_sum_n1 = _mm256_load_pd(denominator_sum + n + 4);
        for (size_t m = 0; m < bw.M; m++) {
            numerator_sum_n0 = _mm256_load_pd(numerator_sum + m*bw.N + n);
            numerator_sum_n1 = _mm256_load_pd(numerator_sum + m*bw.N + n + 4);
            numerator_sum_n0 = _mm256_mul_pd(numerator_sum_n0, denominator_sum_n0);
            numerator_sum_n1 = _mm256_mul_pd(numerator_sum_n1, denominator_sum_n1);
            _mm256_store_pd(bw.emit_prob + m*bw.N + n, numerator_sum_n0);
            _mm256_store_pd(bw.emit_prob + m*bw.N + n + 4, numerator_sum_n1);
        }
    }
}