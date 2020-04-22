/*
    Verifications for the various Baum Welch algorithm implementations
    If you find other test cases, add them!

    -----------------------------------------------------------------------------------

    Spring 2020
    Advanced Systems Lab (How to Write Fast Numerical Code)
    Semester Project: Baum-Welch algorithm

    Authors
    Josua Cantieni, Franz Knobel, Cheuk Yu Chan, Ramon Witschi
    ETH Computer Science MSc, Computer Science Department ETH Zurich

    -----------------------------------------------------------------------------------
*/

#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <time.h>
#include <unistd.h>
// custom files for the project
#include "helper_utilities.h"
// all versions
// only activate one at a time
// (may overwrite each other)
#include "common.h"
//#include "sota.h"
//#include "scalar_optimized.h"
//#include "vector_optimized.h"
//#include "combined_optimized.h"


void test_case_1(compute_bw_func func);
void test_case_2(compute_bw_func func);
void test_case_randomized(compute_bw_func func);

size_t test(__attribute__((unused)) const BWdata& bw){
    return 0;   
}

REGISTER_FUNCTION(test, "Test");

int main() {

    // randomize seed
    size_t tn = time(NULL);
    
    //printf("\nTest Case Custom 1 with srand(%zu)\n", tn);
    //test_case_1();
    //printf("\nTest Case Custom 2 with srand(%zu)\n", tn);
    //test_case_2();

    for(size_t f = 0; f < FuncRegister::size(); f++){
        printf("----------------------------------\n");
        printf("Testing: %s\n", FuncRegister::func_names->at(f).c_str());
        printf("----------------------------------\n");
        size_t iters = 10;
        for (size_t i = 0; i < iters; i++) {
            tn = time(NULL);
            srand(tn);
            printf("\nTest Case Randomized %zu with srand(%zu)\n", i, tn);
            test_case_randomized(FuncRegister::user_funcs->at(f));
        }
    }
    
    printf("\nAll Tests Done!\n\n");
}


void test_case_1(compute_bw_func func) {

    const size_t K = 4;
    const size_t N = 4;
    const size_t M = 8;
    const size_t T = 4;
    const size_t max_iterations = 1000;

    const BWdata& bw = initialize_BWdata(K, N, M, T, max_iterations);
    initialize_random(K, N, M, T, bw.observations, bw.init_prob, bw.trans_prob, bw.emit_prob);
    printf("\nInitialized: K = %zu, N = %zu, M = %zu, T = %zu and max_iterations = %zu\n", K, N, M, T, max_iterations);
    
    func(bw);
    check_and_verify(max_iterations, N, M, bw.init_prob, bw.trans_prob, bw.emit_prob, bw.neg_log_likelihoods);
    //print_states(N, M, T, init_prob, trans_prob, emit_prob);

    clean_BWdata(bw);
}


void test_case_2(compute_bw_func func) {

    const size_t K = 4;
    const size_t N = 4;
    const size_t M = 4;
    const size_t T = 32;
    const size_t max_iterations = 1000;

    const BWdata& bw = initialize_BWdata(K, N, M, T, max_iterations);

    bw.observations[0*T + 5] = 1;
    bw.observations[0*T + 6] = 1;
    bw.observations[1*T + 0] = 1;
    bw.observations[1*T + 4] = 1;
    bw.observations[1*T + 7] = 1;
    bw.observations[2*T + 0] = 1;

    bw.init_prob[0] = 0.25;
    bw.init_prob[1] = 0.25;
    bw.init_prob[2] = 0.25;
    bw.init_prob[3] = 0.25;

    bw.trans_prob[0*N + 0] = 0.5;
    bw.trans_prob[0*N + 1] = 0.5;
    bw.trans_prob[1*N + 0] = 0.3;
    bw.trans_prob[1*N + 1] = 0.7;
    bw.trans_prob[2*N + 0] = 0.5;
    bw.trans_prob[2*N + 1] = 0.5;
    bw.trans_prob[3*N + 0] = 0.3;
    bw.trans_prob[3*N + 1] = 0.7;

    bw.emit_prob[0*M + 0] = 0.3;
    bw.emit_prob[0*M + 1] = 0.7;
    bw.emit_prob[1*M + 0] = 0.8;
    bw.emit_prob[1*M + 1] = 0.2;
    bw.emit_prob[2*M + 0] = 0.3;
    bw.emit_prob[2*M + 1] = 0.7;
    bw.emit_prob[3*M + 0] = 0.8;
    bw.emit_prob[3*M + 1] = 0.2;

    printf("\nInitialized: K = %zu, N = %zu, M = %zu, T = %zu and max_iterations = %zu\n", K, N, M, T, max_iterations);
    func(bw);
    check_and_verify(max_iterations, N, M, bw.init_prob, bw.trans_prob, bw.emit_prob, bw.neg_log_likelihoods);
    print_states(N, M, T, bw.init_prob, bw.trans_prob, bw.emit_prob);

    clean_BWdata(bw);
}


void test_case_randomized(compute_bw_func func) {

    const size_t K = (rand() % 8)*4 + 4;
    const size_t N = (rand() % 9)*4 + 4;
    const size_t M = (rand() % 9)*4 + 4;
    const size_t T = (rand() % 14)*4 + 4;
    const size_t max_iterations = 500;

    // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)


    const BWdata& bw = initialize_BWdata(K, N, M, T, max_iterations);
    initialize_random(K, N, M, T, bw.observations, bw.init_prob, bw.trans_prob, bw.emit_prob);
    printf("\nInitialized: K = %zu, N = %zu, M = %zu, T = %zu and max_iterations = %zu\n", K, N, M, T, max_iterations);
    func(bw);
    check_and_verify(max_iterations, N, M, bw.init_prob, bw.trans_prob, bw.emit_prob, bw.neg_log_likelihoods);

    clean_BWdata(bw);
}