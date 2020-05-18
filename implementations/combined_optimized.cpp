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


void forward_step(const BWdata& bw, double& neg_log_likelihood_sum);
void backward_step(const BWdata& bw, const size_t& k);
void compute_gamma(const BWdata& bw, const size_t& k);
void compute_sigma(const BWdata& bw);
void update_init_prob(const BWdata& bw);
void update_trans_prob(const BWdata& bw);
void update_emit_prob(const BWdata& bw);
size_t comp_bw_combined(const BWdata& bw);


REGISTER_FUNCTION_TRANSPOSE_EMIT_PROB(comp_bw_combined, "combined", "Combined Optimized");


size_t comp_bw_combined(const BWdata& bw){
    size_t res = 0;
    double neg_log_likelihood_sum, neg_log_likelihood_sum_old = 0; // Does not have to be initialized as it will be if and only if i > 0
    bool first = true;

    // run for all iterations
    for (size_t i = 0; i < bw.max_iterations; i++) {
        neg_log_likelihood_sum = 0.0;
        
        forward_step(bw, neg_log_likelihood_sum);
        for (size_t k = 0; k < bw.K; k++) {
            backward_step(bw, k);
            compute_gamma(bw, k);
        }
        
        
        bw.neg_log_likelihoods[i] = neg_log_likelihood_sum;

        if (first && i > 0 && abs(neg_log_likelihood_sum - neg_log_likelihood_sum_old) < 1e-12){
            first = false;
            res = i;
        }

        neg_log_likelihood_sum_old = neg_log_likelihood_sum;
        
        
        update_trans_prob(bw);
        update_emit_prob(bw);
    }
    return res;
}


inline void forward_step(const BWdata& bw, double& neg_log_likelihood_sum) {
    //Init
    __m256d init_prob, emit_prob, alpha, c_norm_v, alpha_sum, trans_prob;
    __m128d vlow, vhigh;
    double c_norm;
    double c_norm0, c_norm1, c_norm2, c_norm3;

    size_t kTN, kT;
    // t = 0, base case

    // Init
    for(size_t k=0; k < bw.K; k++){
        c_norm_v = _mm256_setzero_pd();
        kTN = k*bw.T*bw.N;
        kT = k*bw.T;
        
        size_t observations = bw.observations[k*bw.T];
    
        for (size_t n = 0; n < bw.N; n+=4){
            // Load
            init_prob = _mm256_load_pd(bw.init_prob + n);
            emit_prob = _mm256_load_pd(bw.emit_prob + observations*bw.N + n);
    
            // Calculate
            alpha = _mm256_mul_pd(init_prob, emit_prob);
            c_norm_v = _mm256_fmadd_pd(init_prob, emit_prob, c_norm_v);
            //c_norm0 = init_prob * emit_prob + c_norm0;
            //c_norm1 = init_prob * emit_prob + c_norm1;
            //c_norm2 = init_prob * emit_prob + c_norm2;
            //c_norm3 = init_prob * emit_prob + c_norm3;
    
            // Store
            _mm256_store_pd(bw.alpha + kTN + n, alpha);
        }
    
        // Calculate
        vlow  = _mm256_castpd256_pd128(c_norm_v);
        vhigh = _mm256_extractf128_pd(c_norm_v, 1); // high 128
        vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
//
        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        c_norm = _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
        c_norm = 1.0/c_norm;
        //c_norm = 1.0/(c_norm0 + c_norm1 + c_norm2 + c_norm3)
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
        bw.c_norm[kT] = c_norm;
        neg_log_likelihood_sum += log(c_norm);
        
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
            neg_log_likelihood_sum += log(c_norm);
        }
    }

}


inline void backward_step(const BWdata& bw, const size_t& k) {
    // Init
    double alpha, c_norm, gamma, sigma;
    double beta_sum0, beta_temp0, beta0, emit_prob0, trans_prob0;
    double beta_sum1, beta_temp1, beta1, emit_prob1, trans_prob1;
    double beta_sum2, beta_temp2, beta2, emit_prob2, trans_prob2;
    double beta_sum3, beta_temp3, beta3, emit_prob3, trans_prob3;

    size_t observations, kTN, kTNN, kT, nN;
    // t = bw.T, base case
    kTN = (k*bw.T + (bw.T-1))*bw.N;

    // Load
    memcpy(bw.ggamma + kTN, bw.alpha + kTN, bw.N * sizeof(double));
    c_norm = bw.c_norm[k*bw.T + (bw.T-1)];
    for (size_t n = 0; n < bw.N; n++) {
        // Store
        bw.beta[kTN + n] = c_norm;
    }

    // Recursion step
    kT = k*bw.T;
    for (int t = bw.T-2; t >= 0; t--) {
        // Load
        observations = bw.observations[kT + (t+1)];
        c_norm = bw.c_norm[kT + t];
        kTN = (kT + t) * bw.N;

        for (size_t n0 = 0; n0 < bw.N; n0++) {

            // Load
            beta_sum0 = 0.0;
            beta_sum1 = 0.0;
            beta_sum2 = 0.0;
            beta_sum3 = 0.0;
            alpha = bw.alpha[(kT + t)*bw.N + n0];
            kTNN = (kTN + n0)*bw.N;

            for (size_t n1 = 0; n1 < bw.N; n1+=4) {
                // Load
                beta0 = bw.beta[kTN + bw.N + n1 + 0]; 
                beta1 = bw.beta[kTN + bw.N + n1 + 1];
                beta2 = bw.beta[kTN + bw.N + n1 + 2];
                beta3 = bw.beta[kTN + bw.N + n1 + 3]; 
                trans_prob0 = bw.trans_prob[n0 * bw.N + n1 + 0];
                trans_prob1 = bw.trans_prob[n0 * bw.N + n1 + 1];
                trans_prob2 = bw.trans_prob[n0 * bw.N + n1 + 2];
                trans_prob3 = bw.trans_prob[n0 * bw.N + n1 + 3];
                emit_prob0 = bw.emit_prob[observations*bw.N + (n1 + 0)]; 
                emit_prob1 = bw.emit_prob[observations*bw.N + (n1 + 1)];
                emit_prob2 = bw.emit_prob[observations*bw.N + (n1 + 2)];
                emit_prob3 = bw.emit_prob[observations*bw.N + (n1 + 3)];

                // Calculate & store
                beta_temp0 = beta0 * trans_prob0 * emit_prob0;
                beta_temp1 = beta1 * trans_prob1 * emit_prob1;
                beta_temp2 = beta2 * trans_prob2 * emit_prob2;
                beta_temp3 = beta3 * trans_prob3 * emit_prob3;
                beta_sum0 += beta_temp0;
                beta_sum1 += beta_temp1;
                beta_sum2 += beta_temp2;
                beta_sum3 += beta_temp3;
                bw.sigma[kTNN + n1 + 0] = alpha * beta_temp0;
                bw.sigma[kTNN + n1 + 1] = alpha * beta_temp1;
                bw.sigma[kTNN + n1 + 2] = alpha * beta_temp2;
                bw.sigma[kTNN + n1 + 3] = alpha * beta_temp3;
            }

            // Calculate & store
            bw.beta[kTN + n0] = (beta_sum0 + beta_sum1 + beta_sum2 + beta_sum3) * c_norm;
            bw.ggamma[kTN + n0] = alpha * (beta_sum0 + beta_sum1 + beta_sum2 + beta_sum3);
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
    double g0_sum, denominator_sum_n, denominator_sum_inv;
    double numerator_sum0, numerator_sum1, numerator_sum2, numerator_sum3;
    double* denominator_sum = (double *)aligned_alloc(32, bw.N*sizeof(double));
    double K_inv = 1.0/bw.K;
    
    for (size_t n = 0; n < bw.N; n++) {
        denominator_sum_n = 0;
        g0_sum = 0;
        
        for (size_t k = 0; k < bw.K; k++) {
            denominator_sum_n += bw.gamma_sum[k*bw.N + n];
            g0_sum += bw.ggamma[(k*bw.T)*bw.N + n];
        }
        
        denominator_sum[n] = 1.0/denominator_sum_n;
        bw.init_prob[n] = g0_sum*K_inv;
    }
    
    for (size_t n0 = 0; n0 < bw.N; n0++) {
        for (size_t n1 = 0; n1 < bw.N; n1+=4) {
            // Init (trans_prob)
            numerator_sum0 = 0.0;
            numerator_sum1 = 0.0;
            numerator_sum2 = 0.0;
            numerator_sum3 = 0.0;

            for (size_t k = 0; k < bw.K; k++) {
                // Calculate (trans_prob)
                numerator_sum0 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+0];
                numerator_sum1 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+1];
                numerator_sum2 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+2];
                numerator_sum3 += bw.sigma_sum[(k*bw.N + n0)*bw.N + n1+3];
            }

            // Store (trans_prob)
            bw.trans_prob[n0*bw.N + n1+0] = numerator_sum0*denominator_sum[n0];
            bw.trans_prob[n0*bw.N + n1+1] = numerator_sum1*denominator_sum[n0];
            bw.trans_prob[n0*bw.N + n1+2] = numerator_sum2*denominator_sum[n0];
            bw.trans_prob[n0*bw.N + n1+3] = numerator_sum3*denominator_sum[n0];
        }
    }
    free(denominator_sum);
}


inline void update_emit_prob(const BWdata& bw) {
    // Init
    double denominator_sum0, denominator_sum1, denominator_sum2, denominator_sum3, denominator_sum4, denominator_sum5, denominator_sum6, denominator_sum7;
    double ggamma_cond_sum_tot0, ggamma_cond_sum_tot1, ggamma_cond_sum_tot2, ggamma_cond_sum_tot3;
    double ggamma_cond_sum0, ggamma_cond_sum1, ggamma_cond_sum2, ggamma_cond_sum3;
    double* denominator_sum = (double *)aligned_alloc(32,bw.N * sizeof(double));
    double* numerator_sum = (double *)aligned_alloc(32,bw.N*bw.M * sizeof(double));
    
    // add last bw.T-step to bw.gamma_sum
    for (size_t k = 0; k < bw.K; k++) {
        for (size_t n = 0; n < bw.N; n+=4) {
            bw.gamma_sum[k*bw.N + n+0] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+0];
            bw.gamma_sum[k*bw.N + n+1] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+1];
            bw.gamma_sum[k*bw.N + n+2] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+2];
            bw.gamma_sum[k*bw.N + n+3] += bw.ggamma[(k*bw.T + (bw.T-1))*bw.N + n+3];
        }
    }

    // denominator_sum (top-down)
    for (size_t n = 0; n < bw.N; n += 8){
        denominator_sum0 = bw.gamma_sum[0*bw.N + n+0];
        denominator_sum1 = bw.gamma_sum[0*bw.N + n+1];
        denominator_sum2 = bw.gamma_sum[0*bw.N + n+2];
        denominator_sum3 = bw.gamma_sum[0*bw.N + n+3];
        denominator_sum4 = bw.gamma_sum[0*bw.N + n+4];
        denominator_sum5 = bw.gamma_sum[0*bw.N + n+5];
        denominator_sum6 = bw.gamma_sum[0*bw.N + n+6];
        denominator_sum7 = bw.gamma_sum[0*bw.N + n+7];

        for (size_t k = 1; k < bw.K; k++) {
            denominator_sum0 += bw.gamma_sum[k*bw.N + n+0];
            denominator_sum1 += bw.gamma_sum[k*bw.N + n+1];
            denominator_sum2 += bw.gamma_sum[k*bw.N + n+2];
            denominator_sum3 += bw.gamma_sum[k*bw.N + n+3];
            denominator_sum4 += bw.gamma_sum[k*bw.N + n+4];
            denominator_sum5 += bw.gamma_sum[k*bw.N + n+5];
            denominator_sum6 += bw.gamma_sum[k*bw.N + n+6];
            denominator_sum7 += bw.gamma_sum[k*bw.N + n+7];
        }
        
        denominator_sum[n+0] = denominator_sum0;
        denominator_sum[n+1] = denominator_sum1;
        denominator_sum[n+2] = denominator_sum2;
        denominator_sum[n+3] = denominator_sum3;
        denominator_sum[n+4] = denominator_sum4;
        denominator_sum[n+5] = denominator_sum5;
        denominator_sum[n+6] = denominator_sum6;
        denominator_sum[n+7] = denominator_sum7;
    }

    // numerator_sum
    for (size_t m = 0; m < bw.M; m++) {
        for (size_t n = 0; n < bw.N; n+=4) {
            ggamma_cond_sum_tot0 = 0.0;
            ggamma_cond_sum_tot1 = 0.0;
            ggamma_cond_sum_tot2 = 0.0;
            ggamma_cond_sum_tot3 = 0.0;
            
            for (size_t k = 0; k < bw.K; k++) {
                ggamma_cond_sum0 = 0.0;
                ggamma_cond_sum1 = 0.0;
                ggamma_cond_sum2 = 0.0;
                ggamma_cond_sum3 = 0.0;

                for (size_t t = 0; t < bw.T; t+=4) {
                    if (bw.observations[k*bw.T + t] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t)*bw.N + n+3];
                    }

                    if (bw.observations[k*bw.T + t+1] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t+1)*bw.N + n+3];
                    }

                    if (bw.observations[k*bw.T + t+2] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t+2)*bw.N + n+3];
                    }

                    if (bw.observations[k*bw.T + t+3] == m) {
                        ggamma_cond_sum0 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+0];
                        ggamma_cond_sum1 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+1];
                        ggamma_cond_sum2 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+2];
                        ggamma_cond_sum3 += bw.ggamma[(k*bw.T + t+3)*bw.N + n+3];
                    }
                }
                ggamma_cond_sum_tot0 += ggamma_cond_sum0;
                ggamma_cond_sum_tot1 += ggamma_cond_sum1;
                ggamma_cond_sum_tot2 += ggamma_cond_sum2;
                ggamma_cond_sum_tot3 += ggamma_cond_sum3;
            }
            numerator_sum[(n+0)*bw.M + m] = ggamma_cond_sum_tot0;
            numerator_sum[(n+1)*bw.M + m] = ggamma_cond_sum_tot1;
            numerator_sum[(n+2)*bw.M + m] = ggamma_cond_sum_tot2;
            numerator_sum[(n+3)*bw.M + m] = ggamma_cond_sum_tot3;
        }
    }

    // emit_prob
    for (size_t n = 0; n < bw.N; n++) {
        double denominator_sum_inv = 1.0/denominator_sum[n]; 
        for (size_t m = 0; m < bw.M; m+=4) {
            bw.emit_prob[(m+0)*bw.N + n] = numerator_sum[n*bw.M + m+0] * denominator_sum_inv;
            bw.emit_prob[(m+1)*bw.N + n] = numerator_sum[n*bw.M + m+1] * denominator_sum_inv;
            bw.emit_prob[(m+2)*bw.N + n] = numerator_sum[n*bw.M + m+2] * denominator_sum_inv;
            bw.emit_prob[(m+3)*bw.N + n] = numerator_sum[n*bw.M + m+3] * denominator_sum_inv;
        }
    }
    free(denominator_sum);
    free(numerator_sum);
}
