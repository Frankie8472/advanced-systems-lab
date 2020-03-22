#include <stdlib.h>
#include <stdio.h>

#ifndef WIN32
#include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tsc_x86.h"

#define NUM_RUNS 1
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 2.2e9
#define CALIBRATE


#define state_size 2
#define obs_size 2
#define time 10


int obs[time]; // [i] := observation of time i
double init_prob[state_size]; //[i] := P(X_1 = i);
double trans_prob[state_size][state_size]; // [i][j] := P(X_t = j | X_(t-1) = i)
double emit_prob[obs_size][state_size]; // [i][j] := P(Y_T = y_i | X_t = j)
double alpha[time][state_size]; // [t][i] := P(Y_1 = y_1, ..., Y_t = y_t, X_t = i)
double beta[time][state_size]; // [t][i] := P(Y_(t+1) = y_(t+1), ..., Y_N = y_N | X_t = i)
double gamma[time][state_size]; // [t][i] := P(X_t = i | Y)
double sigma[time][state_size][state_size]; // [t][i][j] = P(X_t = i, X_(t+1) = j | Y)



void init(){
    //uniform
    for (int i = 0; i < state_size; i++){
        init_prob[i] = 1.f/state_size;
        for (int j = 0; j < state_size; j++){
            trans_prob[i][j] = 1.f/state_size;
        }
    }
    for (int i = 0; i < obs_size; i++){
        for (int j = 0; j < state_size; j++){
            emit_prob[i][j] = 1.f/obs_size;
        }
    }
    for (int t = 0; t < time; t++){
        for (int j = 0; j < state_size; j++){
            alpha[t][j] = 0;
            beta[t][j] = 0;
        }
    }

    //fixed observation
    for (int t = 0; t < time; t++){
        obs[t] = t % 2;
    }

    return;
}


void forward(){
    for (int i = 0; i < state_size; i++){
        alpha[0][i] = init_prob[i]*emit_prob[obs[0]][i];
    }
    for (int t = 1; t < time; t++){
        for (int i = 0; i < state_size; i++){
            for (int j = 0; j < state_size; j++){
                alpha[t][i] += alpha[t-1][j]*trans_prob[j][i];
            }

            alpha[t][i] *= emit_prob[obs[t]][i];

        }
    }

    return;
}

void backward(){
    for (int i = 0; i < state_size; i++){
        beta[time-1][i] = 1;
    }
    for (int t = time-2; t >= 0; t--){
        for (int i = 0; i < state_size; i++){
            for (int j = 0; j < state_size; j++){
                beta[t][i] += beta[t+1][j]*trans_prob[i][j]*emit_prob[obs[t+1]][j];
            }
        }
    }

    return;

}

void update(){
    double norm;

    //calculate gamma
    for (int t = 0; t < time; t++){
        norm = 0;
        for (int i = 0; i < state_size; i++){
            gamma[t][i] = alpha[t][i] * beta[t][i];
            norm += alpha[t][i] * beta[t][i];
        }

        for (int i = 0; i < state_size; i++){
            gamma[t][i] /= norm;

        }
    }

    //calculate sigma
    for (int t = 0; t < time-1; t++){
        norm = 0;
        for (int i = 0; i < state_size; i++){
            for (int j = 0; j < state_size; j ++){
                sigma[t][i][j] = alpha[t][i]+trans_prob[i][j]*beta[t+1][j]*emit_prob[obs[t+1]][j];
                norm += alpha[t][i]*trans_prob[i][j]*beta[t+1][j]*emit_prob[obs[t+1]][j];
            }
        }

        for (int i = 0; i < state_size; i++){
            for (int j = 0; j < state_size; j ++){
                sigma[t][i][j] /= norm;
            }
        }
    }

    //update init_prob
    for (int i = 0; i < state_size; i++){
        init_prob[i] = gamma[0][i];
    }

    //sum up sigma (from t = 0 to time -1)
    double sigma_sum[state_size][state_size];

    for (int i = 0; i < state_size; i++){
        for (int j = 0; j < state_size; j ++){
            sigma_sum[i][j] = 0;
            for (int t = 0; t < time-1; t++){
                sigma_sum[i][j] += sigma[t][i][j];
            }
        }
    }

    //sum up gamma (from t = 0 to time-2)
    double gamma_sum[state_size];
    for (int i = 0; i < state_size; i++){
        gamma_sum[i] = 0;
        for (int t = 0; t < time-1; t++){
            gamma_sum[i] += gamma[t][i];
        }
    }

    //update trans_prob
    for (int i = 0; i < state_size; i++){
        for (int j = 0; j < state_size; j ++){
            trans_prob[i][j] = sigma_sum[i][j] / gamma_sum[i];
        }
    }

    //add last timestep to gamma_sum
    for (int i = 0; i < state_size; i++){
        gamma_sum[i] += gamma[time-1][i];
    }

    //update emit_prob
    double sum;
    for (int j = 0; j < state_size; j++){
        for (int i = 0; i < obs_size; i++){
            sum = 0;
            for (int t = 0; t < time; t++){
                if (obs[t] = i){
                    sum += gamma[t][j];
                }
            }

            emit_prob[i][j] = sum / gamma_sum[j];
        }
    }

    return;
}

void compute(int iter){
    for (int i = 0; i < iter; i++){
        forward();
        backward();
        update();
    }

    return;
}

double rdtsc(int n) {
    int i, num_runs;
    myInt64 cycles;
    myInt64 start;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while(num_runs < (1 << 14)) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            compute(n);
        }
        cycles = stop_tsc(start);

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
        compute(n);
    }

    cycles = stop_tsc(start)/num_runs;
    return (double) cycles;
}


int main(int argc, char **argv) {

    if (argc!=2) {printf("usage: FW <n>\n"); return -1;}
    int n = atoi(argv[1]);

    init();
    compute(n);
/*
    double r = rdtsc(n);
    printf("RDTSC instruction:\n %lf cycles measured => %lf seconds, assuming frequency is %lf MHz. (change in source file if different)\n\n", r, r/(FREQUENCY), (FREQUENCY)/1e6);
*/

    printf("Transition probabilities:\n");
    for(int i = 0; i < state_size; i++){
        for(int j = 0; j < state_size; j++){
            printf("state %d -> state %d  %f\n", i+1, j+1, trans_prob[i][j]);
         }
    }

    printf("\nEmission probabilities:\n");
    for(int i = 0; i < obs_size; i++){
        for(int j = 0; j < state_size; j++){
            printf("state %d : %d  %f\n", i+1, j, emit_prob[i][j]);
        }
    }

}
