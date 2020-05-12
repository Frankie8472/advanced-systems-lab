/*
    Vectorized implementation
    Using AVX2 and FMA intrinsics, optimize as best as possible!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

// nice printf template
//printf("\nggamma[(k*bw.T + 0)*bw.N + n] = %f = %f [Address: %p]\n",
//    *(bw.ggamma + index + 0), bw.ggamma[index + 0], (bw.ggamma + index + 0)
//); fflush(0);

#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "../common.h"
#include "../helper_utilities.h"

void forward_step(const BWdata& bw);
void backward_step(const BWdata& bw);
void compute_gamma(const BWdata& bw);
void compute_sigma(const BWdata& bw);
void update_init_prob(const BWdata& bw);
void update_trans_prob(const BWdata& bw);
void update_emit_prob(const BWdata& bw);
size_t comp_bw_vectOwOrized(const BWdata& bw);

// helper functions
void transpose_matrix(const double* input, double* output, const size_t N, const size_t M);

// forward and backward pass are recursively dependent on T
#define STRIDE_LAYER_T_RECURSIVE 1
// note, though, the computation of trans_prob requires
// shifting and the computation of emit_prob requires masking
#define STRIDE_LAYER_T_NON_RECURSIVE 4
// all other loops are fully independent
#define STRIDE_LAYER_N 4
#define STRIDE_LAYER_M 4
#define STRIDE_LAYER_K 4

REGISTER_FUNCTION(comp_bw_vectOwOrized, "vectOwOrized", "Vector Optimized: AVX2 & FMA");

// local globals (heh)
double* helper_4_doubles;
double* emit_prob_transpose;
double* trans_prob_transpose;
const __m256d ones = _mm256_set1_pd(1.0);
const __m256d zeros = _mm256_setzero_pd();


size_t comp_bw_vectOwOrized(const BWdata& bw){

    // helpers
    helper_4_doubles = (double *)aligned_alloc(32, 4*sizeof(double));
    emit_prob_transpose = (double *)aligned_alloc(32, bw.N*bw.M*sizeof(double));
    trans_prob_transpose = (double *)aligned_alloc(32, bw.N*bw.N*sizeof(double));

    // run for all iterations
    size_t iter = 0;
    for (iter = 0; iter < bw.max_iterations; iter++) {

        transpose_matrix(bw.emit_prob, emit_prob_transpose, bw.N, bw.M); // actually worth
        transpose_matrix(bw.trans_prob, trans_prob_transpose, bw.N, bw.N); // actually worth

        forward_step(bw);
        backward_step(bw);
        compute_gamma(bw);
        compute_sigma(bw);
        update_init_prob(bw);
        update_trans_prob(bw);
        update_emit_prob(bw);

        double neg_log_likelihood_sum = 0.0;
        for (size_t k = 0; k < bw.K; k++) {
            for (size_t t = 0; t < bw.T; t++) {
                neg_log_likelihood_sum = neg_log_likelihood_sum + log(bw.c_norm[k*bw.T + t]);
            }
        }
        bw.neg_log_likelihoods[iter] = neg_log_likelihood_sum;

    }

    free(helper_4_doubles);
    free(emit_prob_transpose);
    free(trans_prob_transpose);

    return iter;
}


inline void transpose_matrix(const double* input, double* output, const size_t N, const size_t M) {
    for (size_t n = 0; n < N; n++) {
        for (size_t m = 0; m < M; m++) {
            output[m*N + n] = input[n*M + m];
        }
    }
}


inline void forward_step(const BWdata& bw) {

    // very tedious to vectorize; T is recursively dependent
    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        __m256d vec_c_norm = zeros;

        // t = 0, base case
        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {

            const __m256d vec_init_prob = _mm256_load_pd(bw.init_prob + n);

            const __m256d vec_emit_prob_kp0 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 0)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 0)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 0)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 0)*bw.T + 0]]
            );

            const __m256d vec_emit_prob_kp1 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 1)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 1)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 1)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 1)*bw.T + 0]]
            );

            const __m256d vec_emit_prob_kp2 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 2)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 2)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 2)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 2)*bw.T + 0]]
            );

            const __m256d vec_emit_prob_kp3 = _mm256_set_pd(
                bw.emit_prob[(n + 3)*bw.M + bw.observations[(k + 3)*bw.T + 0]],
                bw.emit_prob[(n + 2)*bw.M + bw.observations[(k + 3)*bw.T + 0]],
                bw.emit_prob[(n + 1)*bw.M + bw.observations[(k + 3)*bw.T + 0]],
                bw.emit_prob[(n + 0)*bw.M + bw.observations[(k + 3)*bw.T + 0]]
            );

            const __m256d vec_kp0 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp0);
            const __m256d vec_kp1 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp1);
            const __m256d vec_kp2 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp2);
            const __m256d vec_kp3 = _mm256_mul_pd(vec_init_prob, vec_emit_prob_kp3);

            const __m256d sum_0_0 = _mm256_hadd_pd(vec_kp0, vec_kp1);
            const __m256d sum_0_1 = _mm256_hadd_pd(vec_kp2, vec_kp3);
            const __m256d sum_0_2 = _mm256_blend_pd(sum_0_0, sum_0_1, 0b1100);
            const __m256d sum_0_3 = _mm256_permute2f128_pd(sum_0_0, sum_0_1, 0b00100001);
            const __m256d sum_0_4 = _mm256_add_pd(sum_0_2, sum_0_3);
            vec_c_norm = _mm256_add_pd(vec_c_norm, sum_0_4);

            _mm256_store_pd((bw.alpha + ((k + 0)*bw.T + 0)*bw.N + n), vec_kp0);
            _mm256_store_pd((bw.alpha + ((k + 1)*bw.T + 0)*bw.N + n), vec_kp1);
            _mm256_store_pd((bw.alpha + ((k + 2)*bw.T + 0)*bw.N + n), vec_kp2);
            _mm256_store_pd((bw.alpha + ((k + 3)*bw.T + 0)*bw.N + n), vec_kp3);
        }

        vec_c_norm = _mm256_div_pd(ones, vec_c_norm);
        _mm256_store_pd(helper_4_doubles, vec_c_norm);

        const __m256d vec_c_norm_kp0 = _mm256_set1_pd(helper_4_doubles[0]);
        const __m256d vec_c_norm_kp1 = _mm256_set1_pd(helper_4_doubles[1]);
        const __m256d vec_c_norm_kp2 = _mm256_set1_pd(helper_4_doubles[2]);
        const __m256d vec_c_norm_kp3 = _mm256_set1_pd(helper_4_doubles[3]);

        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N){

            double* index_kp0 = bw.alpha + (((k + 0)*bw.T + 0)*bw.N + n);
            double* index_kp1 = bw.alpha + (((k + 1)*bw.T + 0)*bw.N + n);
            double* index_kp2 = bw.alpha + (((k + 2)*bw.T + 0)*bw.N + n);
            double* index_kp3 = bw.alpha + (((k + 3)*bw.T + 0)*bw.N + n);

            const __m256d vec_alpha_kp0 = _mm256_load_pd(index_kp0);
            const __m256d vec_alpha_kp1 = _mm256_load_pd(index_kp1);
            const __m256d vec_alpha_kp2 = _mm256_load_pd(index_kp2);
            const __m256d vec_alpha_kp3 = _mm256_load_pd(index_kp3);

            _mm256_store_pd(index_kp0, _mm256_mul_pd(vec_alpha_kp0, vec_c_norm_kp0));
            _mm256_store_pd(index_kp1, _mm256_mul_pd(vec_alpha_kp1, vec_c_norm_kp1));
            _mm256_store_pd(index_kp2, _mm256_mul_pd(vec_alpha_kp2, vec_c_norm_kp2));
            _mm256_store_pd(index_kp3, _mm256_mul_pd(vec_alpha_kp3, vec_c_norm_kp3));
        }

        bw.c_norm[(k + 0)*bw.T + 0] = helper_4_doubles[0];
        bw.c_norm[(k + 1)*bw.T + 0] = helper_4_doubles[1];
        bw.c_norm[(k + 2)*bw.T + 0] = helper_4_doubles[2];
        bw.c_norm[(k + 3)*bw.T + 0] = helper_4_doubles[3];

        // recursion step
        for (size_t t = 1; t < bw.T; t += STRIDE_LAYER_T_RECURSIVE) {

            vec_c_norm = zeros;

            for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

                const __m256d vec_emit_prob_kp0 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 0)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 0)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 0)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 0)*bw.T + t]]
                );

                const __m256d vec_emit_prob_kp1 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 1)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 1)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 1)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 1)*bw.T + t]]
                );

                const __m256d vec_emit_prob_kp2 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 2)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 2)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 2)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 2)*bw.T + t]]
                );

                const __m256d vec_emit_prob_kp3 = _mm256_set_pd(
                    bw.emit_prob[(n0 + 3)*bw.M + bw.observations[(k + 3)*bw.T + t]],
                    bw.emit_prob[(n0 + 2)*bw.M + bw.observations[(k + 3)*bw.T + t]],
                    bw.emit_prob[(n0 + 1)*bw.M + bw.observations[(k + 3)*bw.T + t]],
                    bw.emit_prob[(n0 + 0)*bw.M + bw.observations[(k + 3)*bw.T + t]]
                );

                __m256d vec_trans_prob_sum_np0_kp0 = zeros;
                __m256d vec_trans_prob_sum_np1_kp0 = zeros;
                __m256d vec_trans_prob_sum_np2_kp0 = zeros;
                __m256d vec_trans_prob_sum_np3_kp0 = zeros;

                __m256d vec_trans_prob_sum_np0_kp1 = zeros;
                __m256d vec_trans_prob_sum_np1_kp1 = zeros;
                __m256d vec_trans_prob_sum_np2_kp1 = zeros;
                __m256d vec_trans_prob_sum_np3_kp1 = zeros;

                __m256d vec_trans_prob_sum_np0_kp2 = zeros;
                __m256d vec_trans_prob_sum_np1_kp2 = zeros;
                __m256d vec_trans_prob_sum_np2_kp2 = zeros;
                __m256d vec_trans_prob_sum_np3_kp2 = zeros;

                __m256d vec_trans_prob_sum_np0_kp3 = zeros;
                __m256d vec_trans_prob_sum_np1_kp3 = zeros;
                __m256d vec_trans_prob_sum_np2_kp3 = zeros;
                __m256d vec_trans_prob_sum_np3_kp3 = zeros;

                for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                    const double* index_alpha_kp0 = bw.alpha + ((k + 0)*bw.T + (t-1))*bw.N + n1;
                    const double* index_alpha_kp1 = bw.alpha + ((k + 1)*bw.T + (t-1))*bw.N + n1;
                    const double* index_alpha_kp2 = bw.alpha + ((k + 2)*bw.T + (t-1))*bw.N + n1;
                    const double* index_alpha_kp3 = bw.alpha + ((k + 3)*bw.T + (t-1))*bw.N + n1;

                    const __m256d vec_trans_np0 = _mm256_load_pd(trans_prob_transpose + ((n0 + 0)*bw.N + n1));
                    const __m256d vec_trans_np1 = _mm256_load_pd(trans_prob_transpose + ((n0 + 1)*bw.N + n1));
                    const __m256d vec_trans_np2 = _mm256_load_pd(trans_prob_transpose + ((n0 + 2)*bw.N + n1));
                    const __m256d vec_trans_np3 = _mm256_load_pd(trans_prob_transpose + ((n0 + 3)*bw.N + n1));

                    const __m256d vec_alpha_kp0 = _mm256_load_pd(index_alpha_kp0);
                    const __m256d vec_alpha_kp1 = _mm256_load_pd(index_alpha_kp1);
                    const __m256d vec_alpha_kp2 = _mm256_load_pd(index_alpha_kp2);
                    const __m256d vec_alpha_kp3 = _mm256_load_pd(index_alpha_kp3);

                    vec_trans_prob_sum_np0_kp0 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp0, vec_trans_prob_sum_np0_kp0);
                    vec_trans_prob_sum_np1_kp0 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp0, vec_trans_prob_sum_np1_kp0);
                    vec_trans_prob_sum_np2_kp0 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp0, vec_trans_prob_sum_np2_kp0);
                    vec_trans_prob_sum_np3_kp0 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp0, vec_trans_prob_sum_np3_kp0);

                    vec_trans_prob_sum_np0_kp1 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp1, vec_trans_prob_sum_np0_kp1);
                    vec_trans_prob_sum_np1_kp1 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp1, vec_trans_prob_sum_np1_kp1);
                    vec_trans_prob_sum_np2_kp1 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp1, vec_trans_prob_sum_np2_kp1);
                    vec_trans_prob_sum_np3_kp1 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp1, vec_trans_prob_sum_np3_kp1);

                    vec_trans_prob_sum_np0_kp2 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp2, vec_trans_prob_sum_np0_kp2);
                    vec_trans_prob_sum_np1_kp2 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp2, vec_trans_prob_sum_np1_kp2);
                    vec_trans_prob_sum_np2_kp2 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp2, vec_trans_prob_sum_np2_kp2);
                    vec_trans_prob_sum_np3_kp2 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp2, vec_trans_prob_sum_np3_kp2);

                    vec_trans_prob_sum_np0_kp3 = _mm256_fmadd_pd(vec_trans_np0, vec_alpha_kp3, vec_trans_prob_sum_np0_kp3);
                    vec_trans_prob_sum_np1_kp3 = _mm256_fmadd_pd(vec_trans_np1, vec_alpha_kp3, vec_trans_prob_sum_np1_kp3);
                    vec_trans_prob_sum_np2_kp3 = _mm256_fmadd_pd(vec_trans_np2, vec_alpha_kp3, vec_trans_prob_sum_np2_kp3);
                    vec_trans_prob_sum_np3_kp3 = _mm256_fmadd_pd(vec_trans_np3, vec_alpha_kp3, vec_trans_prob_sum_np3_kp3);

                }

                const __m256d a0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp0, vec_trans_prob_sum_np1_kp0);
                const __m256d a1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp0, vec_trans_prob_sum_np3_kp0);
                const __m256d a2 = _mm256_blend_pd(a0, a1, 0b1100);
                const __m256d a3 = _mm256_permute2f128_pd(a0, a1, 0b00100001);
                const __m256d a4 = _mm256_add_pd(a2, a3);

                const __m256d b0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp1, vec_trans_prob_sum_np1_kp1);
                const __m256d b1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp1, vec_trans_prob_sum_np3_kp1);
                const __m256d b2 = _mm256_blend_pd(b0, b1, 0b1100);
                const __m256d b3 = _mm256_permute2f128_pd(b0, b1, 0b00100001);
                const __m256d b4 = _mm256_add_pd(b2, b3);

                const __m256d c0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp2, vec_trans_prob_sum_np1_kp2);
                const __m256d c1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp2, vec_trans_prob_sum_np3_kp2);
                const __m256d c2 = _mm256_blend_pd(c0, c1, 0b1100);
                const __m256d c3 = _mm256_permute2f128_pd(c0, c1, 0b00100001);
                const __m256d c4 = _mm256_add_pd(c2, c3);

                const __m256d d0 = _mm256_hadd_pd(vec_trans_prob_sum_np0_kp3, vec_trans_prob_sum_np1_kp3);
                const __m256d d1 = _mm256_hadd_pd(vec_trans_prob_sum_np2_kp3, vec_trans_prob_sum_np3_kp3);
                const __m256d d2 = _mm256_blend_pd(d0, d1, 0b1100);
                const __m256d d3 = _mm256_permute2f128_pd(d0, d1, 0b00100001);
                const __m256d d4 = _mm256_add_pd(d2, d3);

                const __m256d vec_kp0 = _mm256_mul_pd(a4, vec_emit_prob_kp0);
                const __m256d vec_kp1 = _mm256_mul_pd(b4, vec_emit_prob_kp1);
                const __m256d vec_kp2 = _mm256_mul_pd(c4, vec_emit_prob_kp2);
                const __m256d vec_kp3 = _mm256_mul_pd(d4, vec_emit_prob_kp3);

                const __m256d sum_0_0 = _mm256_hadd_pd(vec_kp0, vec_kp1);
                const __m256d sum_0_1 = _mm256_hadd_pd(vec_kp2, vec_kp3);
                const __m256d sum_0_2 = _mm256_blend_pd(sum_0_0, sum_0_1, 0b1100);
                const __m256d sum_0_3 = _mm256_permute2f128_pd(sum_0_0, sum_0_1, 0b00100001);
                const __m256d sum_0_4 = _mm256_add_pd(sum_0_2, sum_0_3);
                vec_c_norm = _mm256_add_pd(vec_c_norm, sum_0_4);

                _mm256_store_pd((bw.alpha + ((k + 0)*bw.T + t)*bw.N + n0), vec_kp0);
                _mm256_store_pd((bw.alpha + ((k + 1)*bw.T + t)*bw.N + n0), vec_kp1);
                _mm256_store_pd((bw.alpha + ((k + 2)*bw.T + t)*bw.N + n0), vec_kp2);
                _mm256_store_pd((bw.alpha + ((k + 3)*bw.T + t)*bw.N + n0), vec_kp3);
            }

            vec_c_norm = _mm256_div_pd(ones, vec_c_norm);
            _mm256_store_pd(helper_4_doubles, vec_c_norm);

            const __m256d vec_c_norm_kp0 = _mm256_set1_pd(helper_4_doubles[0]);
            const __m256d vec_c_norm_kp1 = _mm256_set1_pd(helper_4_doubles[1]);
            const __m256d vec_c_norm_kp2 = _mm256_set1_pd(helper_4_doubles[2]);
            const __m256d vec_c_norm_kp3 = _mm256_set1_pd(helper_4_doubles[3]);

            for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N){

                double* index_kp0 = bw.alpha + (((k + 0)*bw.T + t)*bw.N + n);
                double* index_kp1 = bw.alpha + (((k + 1)*bw.T + t)*bw.N + n);
                double* index_kp2 = bw.alpha + (((k + 2)*bw.T + t)*bw.N + n);
                double* index_kp3 = bw.alpha + (((k + 3)*bw.T + t)*bw.N + n);

                const __m256d vec_alpha_kp0 = _mm256_load_pd(index_kp0);
                const __m256d vec_alpha_kp1 = _mm256_load_pd(index_kp1);
                const __m256d vec_alpha_kp2 = _mm256_load_pd(index_kp2);
                const __m256d vec_alpha_kp3 = _mm256_load_pd(index_kp3);

                _mm256_store_pd(index_kp0, _mm256_mul_pd(vec_alpha_kp0, vec_c_norm_kp0));
                _mm256_store_pd(index_kp1, _mm256_mul_pd(vec_alpha_kp1, vec_c_norm_kp1));
                _mm256_store_pd(index_kp2, _mm256_mul_pd(vec_alpha_kp2, vec_c_norm_kp2));
                _mm256_store_pd(index_kp3, _mm256_mul_pd(vec_alpha_kp3, vec_c_norm_kp3));
            }

            bw.c_norm[(k + 0)*bw.T + t] = helper_4_doubles[0];
            bw.c_norm[(k + 1)*bw.T + t] = helper_4_doubles[1];
            bw.c_norm[(k + 2)*bw.T + t] = helper_4_doubles[2];
            bw.c_norm[(k + 3)*bw.T + t] = helper_4_doubles[3];

        }

    }
}


inline void backward_step(const BWdata& bw) {

    for (size_t k = 0; k < bw.K; k += STRIDE_LAYER_K) {

        // t = bw.T, base case
        for (size_t n = 0; n < bw.N; n += STRIDE_LAYER_N) {
            _mm256_store_pd((bw.beta + (((k + 0)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + (bw.T-1)]));
            _mm256_store_pd((bw.beta + (((k + 1)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + (bw.T-1)]));
            _mm256_store_pd((bw.beta + (((k + 2)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + (bw.T-1)]));
            _mm256_store_pd((bw.beta + (((k + 3)*bw.T + (bw.T-1))*bw.N + n)), _mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + (bw.T-1)]));
        }

        // recursion step
        for (int t = bw.T-2; t >= 0; t -= STRIDE_LAYER_T_RECURSIVE) {

            const size_t index_emitobs_kp0 = bw.observations[(k + 0)*bw.T + (t+1)];
            const size_t index_emitobs_kp1 = bw.observations[(k + 1)*bw.T + (t+1)];
            const size_t index_emitobs_kp2 = bw.observations[(k + 2)*bw.T + (t+1)];
            const size_t index_emitobs_kp3 = bw.observations[(k + 3)*bw.T + (t+1)];

            for (size_t n0 = 0; n0 < bw.N; n0 += STRIDE_LAYER_N) {

                __m256d vec_beta_tmp_np0_kp0 = zeros;
                __m256d vec_beta_tmp_np0_kp1 = zeros;
                __m256d vec_beta_tmp_np0_kp2 = zeros;
                __m256d vec_beta_tmp_np0_kp3 = zeros;

                __m256d vec_beta_tmp_np1_kp0 = zeros;
                __m256d vec_beta_tmp_np1_kp1 = zeros;
                __m256d vec_beta_tmp_np1_kp2 = zeros;
                __m256d vec_beta_tmp_np1_kp3 = zeros;

                __m256d vec_beta_tmp_np2_kp0 = zeros;
                __m256d vec_beta_tmp_np2_kp1 = zeros;
                __m256d vec_beta_tmp_np2_kp2 = zeros;
                __m256d vec_beta_tmp_np2_kp3 = zeros;

                __m256d vec_beta_tmp_np3_kp0 = zeros;
                __m256d vec_beta_tmp_np3_kp1 = zeros;
                __m256d vec_beta_tmp_np3_kp2 = zeros;
                __m256d vec_beta_tmp_np3_kp3 = zeros;

                for (size_t n1 = 0; n1 < bw.N; n1 += STRIDE_LAYER_N) {

                    const __m256d vec_beta_kp0 = _mm256_load_pd(bw.beta + (((k + 0)*bw.T + (t+1))*bw.N + n1));
                    const __m256d vec_beta_kp1 = _mm256_load_pd(bw.beta + (((k + 1)*bw.T + (t+1))*bw.N + n1));
                    const __m256d vec_beta_kp2 = _mm256_load_pd(bw.beta + (((k + 2)*bw.T + (t+1))*bw.N + n1));
                    const __m256d vec_beta_kp3 = _mm256_load_pd(bw.beta + (((k + 3)*bw.T + (t+1))*bw.N + n1));

                    const __m256d vec_trans_prob_np0 = _mm256_load_pd(bw.trans_prob + (n0 + 0)*bw.N + n1);
                    const __m256d vec_trans_prob_np1 = _mm256_load_pd(bw.trans_prob + (n0 + 1)*bw.N + n1);
                    const __m256d vec_trans_prob_np2 = _mm256_load_pd(bw.trans_prob + (n0 + 2)*bw.N + n1);
                    const __m256d vec_trans_prob_np3 = _mm256_load_pd(bw.trans_prob + (n0 + 3)*bw.N + n1);

                    const __m256d vec_emit_prob_kp0 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp0*bw.N + n1));
                    const __m256d vec_emit_prob_kp1 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp1*bw.N + n1));
                    const __m256d vec_emit_prob_kp2 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp2*bw.N + n1));
                    const __m256d vec_emit_prob_kp3 = _mm256_load_pd(emit_prob_transpose + (index_emitobs_kp3*bw.N + n1));

                    vec_beta_tmp_np0_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp0), vec_beta_tmp_np0_kp0);
                    vec_beta_tmp_np0_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp1), vec_beta_tmp_np0_kp1);
                    vec_beta_tmp_np0_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp2), vec_beta_tmp_np0_kp2);
                    vec_beta_tmp_np0_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np0, vec_emit_prob_kp3), vec_beta_tmp_np0_kp3);

                    vec_beta_tmp_np1_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp0), vec_beta_tmp_np1_kp0);
                    vec_beta_tmp_np1_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp1), vec_beta_tmp_np1_kp1);
                    vec_beta_tmp_np1_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp2), vec_beta_tmp_np1_kp2);
                    vec_beta_tmp_np1_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np1, vec_emit_prob_kp3), vec_beta_tmp_np1_kp3);

                    vec_beta_tmp_np2_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp0), vec_beta_tmp_np2_kp0);
                    vec_beta_tmp_np2_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp1), vec_beta_tmp_np2_kp1);
                    vec_beta_tmp_np2_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp2), vec_beta_tmp_np2_kp2);
                    vec_beta_tmp_np2_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np2, vec_emit_prob_kp3), vec_beta_tmp_np2_kp3);

                    vec_beta_tmp_np3_kp0 = _mm256_fmadd_pd(vec_beta_kp0, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp0), vec_beta_tmp_np3_kp0);
                    vec_beta_tmp_np3_kp1 = _mm256_fmadd_pd(vec_beta_kp1, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp1), vec_beta_tmp_np3_kp1);
                    vec_beta_tmp_np3_kp2 = _mm256_fmadd_pd(vec_beta_kp2, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp2), vec_beta_tmp_np3_kp2);
                    vec_beta_tmp_np3_kp3 = _mm256_fmadd_pd(vec_beta_kp3, _mm256_mul_pd(vec_trans_prob_np3, vec_emit_prob_kp3), vec_beta_tmp_np3_kp3);

                }

                const __m256d a0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp0, vec_beta_tmp_np1_kp0);
                const __m256d a1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp0, vec_beta_tmp_np3_kp0);
                const __m256d a2 = _mm256_blend_pd(a0, a1, 0b1100);
                const __m256d a3 = _mm256_permute2f128_pd(a0, a1, 0b00100001);
                const __m256d a4 = _mm256_add_pd(a2, a3);

                const __m256d b0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp1, vec_beta_tmp_np1_kp1);
                const __m256d b1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp1, vec_beta_tmp_np3_kp1);
                const __m256d b2 = _mm256_blend_pd(b0, b1, 0b1100);
                const __m256d b3 = _mm256_permute2f128_pd(b0, b1, 0b00100001);
                const __m256d b4 = _mm256_add_pd(b2, b3);

                const __m256d c0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp2, vec_beta_tmp_np1_kp2);
                const __m256d c1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp2, vec_beta_tmp_np3_kp2);
                const __m256d c2 = _mm256_blend_pd(c0, c1, 0b1100);
                const __m256d c3 = _mm256_permute2f128_pd(c0, c1, 0b00100001);
                const __m256d c4 = _mm256_add_pd(c2, c3);

                const __m256d d0 = _mm256_hadd_pd(vec_beta_tmp_np0_kp3, vec_beta_tmp_np1_kp3);
                const __m256d d1 = _mm256_hadd_pd(vec_beta_tmp_np2_kp3, vec_beta_tmp_np3_kp3);
                const __m256d d2 = _mm256_blend_pd(d0, d1, 0b1100);
                const __m256d d3 = _mm256_permute2f128_pd(d0, d1, 0b00100001);
                const __m256d d4 = _mm256_add_pd(d2, d3);

                _mm256_store_pd((bw.beta + (((k + 0)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 0)*bw.T + t]), a4));
                _mm256_store_pd((bw.beta + (((k + 1)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 1)*bw.T + t]), b4));
                _mm256_store_pd((bw.beta + (((k + 2)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 2)*bw.T + t]), c4));
                _mm256_store_pd((bw.beta + (((k + 3)*bw.T + t)*bw.N + n0)), _mm256_mul_pd(_mm256_set1_pd(bw.c_norm[(k + 3)*bw.T + t]), d4));

            }

        }

    }

}


inline void compute_gamma(const BWdata& bw) {


    for (size_t k = 0; k < bw.K; k++) {

        for (size_t t = 0; t < bw.T; t++) {

            for (size_t n = 0; n < bw.N; n++) {

                bw.ggamma[(k*bw.T + t)*bw.N + n] = bw.alpha[(k*bw.T + t)*bw.N + n] * bw.beta[(k*bw.T + t)*bw.N + n] / bw.c_norm[k*bw.T + t];

            }

        }

    }



    // sum up bw.ggamma (from t = 0 to bw.T-2; serve as normalizer for bw.trans_prob)

    for (size_t k = 0; k < bw.K; k++) {

        for (size_t n = 0; n < bw.N; n++) {

            double g_sum = 0.0;

            for (size_t t = 0; t < bw.T-1; t++) {

                g_sum += bw.ggamma[(k*bw.T + t)*bw.N + n];

            }

            bw.gamma_sum[k*bw.N + n] = g_sum;

        }

    }


    /*
    for (size_t k = 0; k < 4; k++) {
        printf("c_norm[k = %zu][t = %zu] = %f\n", k, (size_t) 1, bw.c_norm[k*bw.T + 1]);
        fflush(0);
    }
    for (size_t k = 0; k < 4; k++) {
        for (size_t n = 0; n < 4; n++) {
            printf("alpha[k = %zu][t = %zu][n = %zu] = %f\n", k, (size_t) 1, n, bw.alpha[(k*bw.T + 1)*bw.N + n]);
            fflush(0);
        }
    }*/
}


inline void compute_sigma(const BWdata& bw) {



    for (size_t k = 0; k < bw.K; k++) {

        for (size_t t = 0; t < bw.T-1; t++) {

            for (size_t n0 = 0; n0 < bw.N; n0++) {

                for (size_t n1 = 0; n1 < bw.N; n1++) {

                    bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1] = \
                        bw.alpha[(k*bw.T + t)*bw.N + n0]*\
                        bw.trans_prob[n0*bw.N + n1]\
                        *bw.beta[(k*bw.T + (t+1))*bw.N + n1]*\
                        bw.emit_prob[n1*bw.M + bw.observations[k*bw.T + (t+1)]];

                }

            }

        }



        // sum up bw.sigma (from t = 0 to bw.T-1)
        for (size_t n0 = 0; n0 < bw.N; n0++) {

            for (size_t n1 = 0; n1 < bw.N; n1++) {

                double s_sum = 0.0;

                for (size_t t = 0; t < bw.T-1; t++) {

                    s_sum += bw.sigma[((k*bw.T + t)*bw.N + n0)*bw.N + n1];

                }

                bw.sigma_sum[(k*bw.N + n0)*bw.N + n1] = s_sum;

            }

        }

    }

}


inline void update_init_prob(const BWdata& bw) {

    const __m256d vec_K_inv = _mm256_set1_pd(1.0/bw.K);

    for (size_t n = 0; n < bw.N; n += 4) {

        __m256d vec_g0_sum = zeros;

        for (size_t k = 0; k < bw.K; k += 4) {

            const size_t index_0 = ((k + 0)*bw.T + 0)*bw.N + n;
            const size_t index_1 = ((k + 1)*bw.T + 0)*bw.N + n;
            const size_t index_2 = ((k + 2)*bw.T + 0)*bw.N + n;
            const size_t index_3 = ((k + 3)*bw.T + 0)*bw.N + n;

            const __m256d vec_gamma_k_0_n_0 = _mm256_load_pd(bw.ggamma + index_0);
            const __m256d vec_gamma_k_1_n_0 = _mm256_load_pd(bw.ggamma + index_1);
            const __m256d vec_gamma_k_2_n_0 = _mm256_load_pd(bw.ggamma + index_2);
            const __m256d vec_gamma_k_3_n_0 = _mm256_load_pd(bw.ggamma + index_3);

            const __m256d a = _mm256_add_pd(vec_gamma_k_0_n_0, vec_gamma_k_1_n_0);
            const __m256d b = _mm256_add_pd(vec_gamma_k_2_n_0, vec_gamma_k_3_n_0);

            vec_g0_sum = _mm256_add_pd(vec_g0_sum, _mm256_add_pd(a, b));
        }

        _mm256_store_pd(bw.init_prob + n, _mm256_mul_pd(vec_g0_sum, vec_K_inv));

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
