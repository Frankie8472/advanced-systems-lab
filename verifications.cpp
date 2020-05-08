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
#include <tuple>
#include <random>
#include <time.h>
#include <unistd.h>
// custom files for the project
#include "helper_utilities.h"
#include "common.h"

void check_baseline(void);
void check_user_functions(const size_t nb_random_tests);
bool test_case_0(compute_bw_func func);
bool test_case_1(compute_bw_func func);
bool test_case_2(compute_bw_func func);
bool test_case_randomized(compute_bw_func func);

int main() {
    // maybe add commandline arguments, dunno
    if ( true ) check_baseline();
    const size_t nb_random_tests = 5;
    if ( true ) check_user_functions(nb_random_tests);
}


inline void check_baseline(void) {

    const size_t baseline_random_seed = time(NULL);
    srand(baseline_random_seed);
    const size_t baseline_random_number = rand();

    // hardcoded test cases to check correctness of the baseline implementation
    printf("\x1b[1m\n-------------------------------------------------------------------------------\x1b[0m\n");
    printf("\x1b[1mBaseline Verifications with Baseline Random Number [%zu]\x1b[0m\n", baseline_random_number);
    printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
    const bool success_test_case_0 = test_case_0(FuncRegister::baseline_func);
    const bool success_test_case_1 = test_case_1(FuncRegister::baseline_func);
    const bool success_test_case_2 = test_case_2(FuncRegister::baseline_func);
    if ( success_test_case_0 ) {
        printf("\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 0\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 0\n");
    }
    if ( success_test_case_1 ) {
        printf("\n\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 1\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 1\n");
    }
    if ( success_test_case_2 ) {
        printf("\n\x1b[1;32m[SUCCEEDED]:\x1b[0m Baseline Test Case Custom 2\n");
    } else {
        printf("\n\x1b[1;31m[FAILED]:\x1b[0m Baseline Test Case Custom 2\n");
    }
    printf("-------------------------------------------------------------------------------\n");
}


inline void check_user_functions(const size_t nb_random_tests) {

    const size_t nb_user_functions = FuncRegister::size();
    bool test_results[nb_user_functions][nb_random_tests];

    // check optimizations w.r.t. the baseline using randomized tests
    // NOTE : this assumes that the Baseline is 100% correctly implemented (which it isn't as of yet uwu)
    for (size_t i = 0; i < nb_random_tests; i++) {

        // randomize seed (new for each random test case)
        const size_t baseline_random_seed = time(NULL)*i;
        srand(baseline_random_seed);
        size_t baseline_random_number = rand();

        // initialize data for random test case i
        // NOTE
        // we assume sufficiently high (>= 16) values
        // NOTE
        // we assume divisibility of (16)
        const size_t K = (rand() % 2)*16 + 16;
        const size_t N = (rand() % 3)*16 + 16;
        const size_t M = (rand() % 3)*16 + 16;
        const size_t T = (rand() % 4)*16 + 16;
        const size_t max_iterations = 1000;

        // calloc initializes each byte to 0b00000000, i.e. 0.0 (double)
        const BWdata& bw_baseline_initialized = create_BWdata(K, N, M, T, max_iterations);
        //initialize_uar(bw_baseline_initialized); // converges fast, but works now.
        initialize_random(bw_baseline_initialized);
        const BWdata& bw_baseline = full_copy_BWdata(bw_baseline_initialized);

        // run baseline and don't touch the bw_baseline data:
        // each of the users function bw_user_function data in
        // the loop below will be checked against bw_baseline data!
        printf("\x1b[1m\n-------------------------------------------------------------------------------\x1b[0m\n");
        printf("\x1b[1mTest Case Randomized [%zu] with Baseline Random Number [%zu]\x1b[0m\n", i, baseline_random_number);
        printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
        printf("Initialized: K = %zu, N = %zu, M = %zu, T = %zu and max_iterations = %zu\n", K, N, M, T, max_iterations);
        printf("-------------------------------------------------------------------------------\n");
        printf("Running \x1b[1m'Baseline'\x1b[0m\n");
        printf("-------------------------------------------------------------------------------\n");
        const size_t baseline_convergence = FuncRegister::baseline_func(bw_baseline);
        printf("It took \x1b[1m[%zu] iterations\x1b[0m to converge\n", baseline_convergence);
        printf("-------------------------------------------------------------------------------\n");
        const bool baseline_sucess = check_and_verify(bw_baseline);
        printf("-------------------------------------------------------------------------------\n");

        // run all user functions and compare against the data
        for(size_t f = 0; f < nb_user_functions; f++) {

            printf("Running User Function \x1b[1m'%s'\x1b[0m\n", FuncRegister::func_names->at(f).c_str());
            printf("-------------------------------------------------------------------------------\n");
            const BWdata& bw_user_function = full_copy_BWdata(bw_baseline_initialized);
            const size_t user_function_convergence = FuncRegister::user_funcs->at(f)(bw_user_function);
            printf("It took \x1b[1m[%zu] iterations\x1b[0m to converge\n", user_function_convergence);
            printf("-------------------------------------------------------------------------------\n");
            const bool user_function_sucess = check_and_verify(bw_user_function);
            printf("-------------------------------------------------------------------------------\n");
            const bool is_bw_baseline_equal_bw_user_function = is_BWdata_equal(bw_baseline, bw_user_function);
            //const bool is_bw_baseline_equal_bw_user_function = is_BWdata_equal_only_probabilities(bw_baseline, bw_user_function);
            printf("-------------------------------------------------------------------------------\n");
            //const bool is_bw_baseline_equal_bw_user_function = is_BWdata_equal_only_probabilities(bw_baseline, bw_user_function);

            // Okay, hear me out!
            // If baseline is correct, then that's dope and we wanna have user function also correct, right?
            // Though, if baseline is wrong, then user function being true might be some potential bug-problem!
            // NOTE 1
            // This only concerns the "check_and_verify" function; which checks e.g. whether probabilities sum to 1.
            // However, the functional correctness may still be wrong, but that wouldn't be the fault of the user_function.
            // The only job of the user_function is to match the baseline; the baseline itself has to be correct!
            // NOTE 2
            // Checking the convergence rate of the Baseline with the User Function may or may not be a good idea
            // U may change (false -> true), but no big h8sies pls uwu
            test_results[f][i] = (
                   ( false || is_bw_baseline_equal_bw_user_function )
                && ( false || user_function_sucess )
                && ( true || ( user_function_convergence == baseline_convergence ) )
                && ( false || ( user_function_sucess == baseline_sucess ) )
            );

            clean_BWdata(bw_user_function);
        }

        clean_BWdata(bw_baseline);
        clean_BWdata(bw_baseline_initialized);
    }

    printf("\nAll Tests Done!\n\n");
    printf("Results:\n");
    printf("-------------------------------------------------------------------------------\n");
    for (size_t f = 0; f < nb_user_functions; f++) {

        size_t nb_fails = 0;
        for (size_t i = 0; i < nb_random_tests; i++) {
            if (test_results[f][i] == false) nb_fails++;
        }

        printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
        if(nb_fails == 0){
            printf("\x1b[1;32mALL CASES PASSED:\x1b[0m '%s'\n", FuncRegister::func_names->at(f).c_str());
        } else {
            printf("\x1b[1;31m[%zu/%zu] CASES FAILED:\x1b[0m    '%s' \n", nb_fails, nb_random_tests, FuncRegister::func_names->at(f).c_str());
        }
        printf("\x1b[1m-------------------------------------------------------------------------------\x1b[0m\n");
        for (size_t i = 0; i < nb_random_tests; i++) {
            if(test_results[f][i]){
                printf("\x1b[1;32mPASSED\x1b[0m Test Case Randomized [%zu]\n", i);
            } else {
                printf("\x1b[1;31mFAILED:\x1b[0m Test Case Randomized [%zu]\n", i);
            }
        }
    }
    printf("-------------------------------------------------------------------------------\n");
}


/**
 * Description
 * Implements the example from
 * https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Example
 */
bool test_case_0(compute_bw_func func) {

    const size_t K = 1;
    const size_t N = 2;
    const size_t M = 2;
    const size_t T = 10;
    const size_t max_iterations = 1; 

    const BWdata& bw = create_BWdata(K, N, M, T, max_iterations);
    
    // we have 1 observation sequence with 10 observations
    // "E = eggs" is "1" and "N = no eggs" is "0"
    bw.observations[0*T + 0] = 0;
    bw.observations[0*T + 1] = 0;
    bw.observations[0*T + 2] = 0;
    bw.observations[0*T + 3] = 0;
    bw.observations[0*T + 4] = 0;
    bw.observations[0*T + 5] = 1;
    bw.observations[0*T + 6] = 1;
    bw.observations[0*T + 7] = 0;
    bw.observations[0*T + 8] = 0;
    bw.observations[0*T + 9] = 0;

    // we have 2 initial probabilities (must sum to 1)
    bw.init_prob[0] = 0.2;
    bw.init_prob[1] = 0.8;

    // we have a transition probability matrix with 2x2 entries (rows must sum to 1)
    bw.trans_prob[0*N + 0] = 0.5;
    bw.trans_prob[0*N + 1] = 0.5;
    bw.trans_prob[1*N + 0] = 0.3;
    bw.trans_prob[1*N + 1] = 0.7;

    // we have an emission/observation probability matrix with 2x2 entries (rows must sum to 1)
    bw.emit_prob[0*M + 0] = 0.3;
    bw.emit_prob[0*M + 1] = 0.7;
    bw.emit_prob[1*M + 0] = 0.8;
    bw.emit_prob[1*M + 1] = 0.2;

    //print_BWdata(bw); // to check what's wrong

    func(bw);
    //size_t nb_iter = func(bw); // run experiment
    //printf("\ntest_case_0 took %zu iterations until convergence.\n\n", nb_iter);

    // checks only conceptual stuff;
    // e.g. whether probabilities work out (sum to 1)
    bool success = check_and_verify(bw);

    size_t errors = 0;
    // must be the same (up to some numerical error margin)!
    
    /* have to figure them out first
    if ( ! ( bw.init_prob[0] == 0.0 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("init_prob[0] = %f but should be %f", bw.init_prob[0], 0.0);
    } else {
        PRINT_PASSED("init_prob[0] passed!");
    }
    if ( ! ( bw.init_prob[1] == 0.0 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("init_prob[1] = %f but should be %f", bw.init_prob[1], 0.0);
    } else {
        PRINT_PASSED("init_prob[1] passed!");
    }
    */
    if ( ! ( bw.trans_prob[0*N + 0] == 0.3973 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("trans_prob[0][0] = %f but should be %f", bw.trans_prob[0*N + 0], 0.3973);
    } else {
        PRINT_PASSED("trans_prob[0][0] passed!");
    }
    if ( ! ( bw.trans_prob[0*N + 1] == 0.6027 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("trans_prob[0][1] = %f but should be %f", bw.trans_prob[0*N + 1], 0.6027);
    } else {
        PRINT_PASSED("trans_prob[0][1] passed!");
    }
    if ( ! ( bw.trans_prob[1*N + 0] == 	0.1833 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("trans_prob[1][0] = %f but should be %f", bw.trans_prob[1*N + 0], 0.1833);
    } else {
        PRINT_PASSED("trans_prob[1][0] passed!");
    }
    if ( ! ( bw.trans_prob[1*N + 1] == 	0.8167 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("trans_prob[1][1] = %f but should be %f", bw.trans_prob[1*N + 1], 0.8167);
    } else {
        PRINT_PASSED("trans_prob[1][1] passed!");
    }
    if ( ! ( bw.emit_prob[0*M + 0] == 0.0908 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("emit_prob[0][0] = %f but should be %f", bw.emit_prob[0*M + 0], 0.0908);
    } else {
        PRINT_PASSED("emit_prob[0][0] passed!");
    }
    if ( ! ( bw.emit_prob[0*M + 1] == 0.9092 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("emit_prob[0][1] = %f but should be %f", bw.emit_prob[0*M + 1], 0.9092);
    } else {
        PRINT_PASSED("emit_prob[0][1] passed!");
    }
    if ( ! ( bw.emit_prob[1*M + 0] == 0.5752 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("emit_prob[1][0] = %f but should be %f", bw.emit_prob[1*M + 0], 0.5752);
    } else {
        PRINT_PASSED("emit_prob[1][0] passed!");
    }
    if ( ! ( bw.emit_prob[1*M + 1] == 0.4248 ) ) {
        errors += 1;
        success = false;
        PRINT_FAIL("emit_prob[1][1] = %f but should be %f", bw.emit_prob[1*M + 1], 0.4248);
    } else {
        PRINT_PASSED("emit_prob[1][1] passed!");
    }

    if(errors > 0){
        PRINT_VIOLATION("In Wikipedia Testcase", errors);
    }

    //print_BWdata(bw); // to check what's wrong

    clean_BWdata(bw);

    return success;
}


bool test_case_1(compute_bw_func func) {

    const size_t K = 4;
    const size_t N = 4;
    const size_t M = 8;
    const size_t T = 4;
    const size_t max_iterations = 1000;

    const BWdata& bw = create_BWdata(K, N, M, T, max_iterations);
    initialize_random(bw);
    printf("\nInitialized: K = %zu, N = %zu, M = %zu, T = %zu and max_iterations = %zu\n", K, N, M, T, max_iterations);
    
    func(bw);
    bool success = check_and_verify(bw);
    //print_states(bw);

    clean_BWdata(bw);

    return success;
}


bool test_case_2(compute_bw_func func) {

    const size_t K = 4;
    const size_t N = 4;
    const size_t M = 4;
    const size_t T = 32;
    const size_t max_iterations = 1000;

    const BWdata& bw = create_BWdata(K, N, M, T, max_iterations);

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
    bool success = check_and_verify(bw);
    //print_states(bw);

    clean_BWdata(bw);

    return success;
}
