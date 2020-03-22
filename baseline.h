/*
    Baseline implementation
    Not verified yet!
*/


unsigned int STATE_SIZE;
unsigned int OBS_SIZE;
unsigned int TIME_STEPS;


void forward(int* obs, double* init_prob, double* trans_prob, double* emit_prob,
            double* alpha, double* beta, double* ggamma, double* sigma) {

    for (int i = 0; i < STATE_SIZE; i++) {
        alpha[0*TIME_STEPS + i] = init_prob[i]*emit_prob[obs[0]*OBS_SIZE + i];
    }
    for (int t = 1; t < TIME_STEPS; t++) {
        for (int i = 0; i < STATE_SIZE; i++) {
            for (int j = 0; j < STATE_SIZE; j++) {
                alpha[t*TIME_STEPS + i] += alpha[(t-1)*TIME_STEPS + j]*trans_prob[j*STATE_SIZE + i];
            }
            alpha[t*TIME_STEPS + i] *= emit_prob[obs[t]*OBS_SIZE + i];
        }
    }
    return;
}


void backward(int* obs, double* init_prob, double* trans_prob, double* emit_prob,
            double* alpha, double* beta, double* ggamma, double* sigma) {

    for (int i = 0; i < STATE_SIZE; i++) {
        beta[(TIME_STEPS-1)*TIME_STEPS + i] = 1;
    }
    for (int t = TIME_STEPS-2; t >= 0; t--) {
        for (int i = 0; i < STATE_SIZE; i++) {
            for (int j = 0; j < STATE_SIZE; j++) {
                beta[t*TIME_STEPS + i] += beta[(t+1)*TIME_STEPS + j] * trans_prob[i*STATE_SIZE + j] * emit_prob[obs[t+1]*OBS_SIZE + j];
            }
        }
    }

    return;
}


void update(int* obs, double* init_prob, double* trans_prob, double* emit_prob,
            double* alpha, double* beta, double* ggamma, double* sigma) {

    double norm;

    //calculate ggamma
    for (int t = 0; t < TIME_STEPS; t++) {
        norm = 0;
        for (int i = 0; i < STATE_SIZE; i++) {
            ggamma[t*TIME_STEPS + i] = alpha[t*TIME_STEPS + i] * beta[t*TIME_STEPS + i];
            norm += alpha[t*TIME_STEPS + i] * beta[t*TIME_STEPS + i];
        }
        for (int i = 0; i < STATE_SIZE; i++) {
            ggamma[t*TIME_STEPS + i] /= norm;

        }
    }

    //calculate sigma
    for (int t = 0; t < TIME_STEPS-1; t++) {
        norm = 0;
        for (int i = 0; i < STATE_SIZE; i++) {
            for (int j = 0; j < STATE_SIZE; j ++) {
                sigma[(t*TIME_STEPS + i)*STATE_SIZE + j] = alpha[t*TIME_STEPS + i] + trans_prob[i*STATE_SIZE + j]*beta[(t+1)*TIME_STEPS + j]*emit_prob[obs[t+1]*OBS_SIZE + j];
                norm += alpha[t*TIME_STEPS + i]*trans_prob[i*STATE_SIZE + j]*beta[(t+1)*TIME_STEPS + j]*emit_prob[obs[t+1]*OBS_SIZE + j];
            }
        }
        for (int i = 0; i < STATE_SIZE; i++) {
            for (int j = 0; j < STATE_SIZE; j ++) {
                sigma[(t*TIME_STEPS + i)*STATE_SIZE + j] /= norm;
            }
        }
    }

    // update init_prob
    for (int i = 0; i < STATE_SIZE; i++) {
        init_prob[i] = ggamma[0*TIME_STEPS + i];
    }

    // sum up sigma (from t = 0 to TIME_STEPS -1)
    double sigma_sum[STATE_SIZE][STATE_SIZE];

    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j ++) {
            sigma_sum[i][j] = 0;
            for (int t = 0; t < TIME_STEPS-1; t++) {
                sigma_sum[i][j] += sigma[(t*TIME_STEPS + i)*STATE_SIZE + j];
            }
        }
    }

    // sum up ggamma (from t = 0 to TIME_STEPS-2)
    double ggamma_sum[STATE_SIZE];

    for (int i = 0; i < STATE_SIZE; i++) {
        ggamma_sum[i] = 0;
        for (int t = 0; t < TIME_STEPS-1; t++) {
            ggamma_sum[i] += ggamma[t*TIME_STEPS + i];
        }
    }

    // update trans_prob
    for (int i = 0; i < STATE_SIZE; i++) {
        for (int j = 0; j < STATE_SIZE; j ++) {
            trans_prob[i*STATE_SIZE + j] = sigma_sum[i][j] / ggamma_sum[i];
        }
    }

    //add last TIME_STEPSstep to ggamma_sum
    for (int i = 0; i < STATE_SIZE; i++) {
        ggamma_sum[i] += ggamma[(TIME_STEPS-1)*TIME_STEPS + i];
    }

    //update emit_prob
    double sum;
    for (int j = 0; j < STATE_SIZE; j++) {
        for (int i = 0; i < OBS_SIZE; i++) {
            sum = 0;
            for (int t = 0; t < TIME_STEPS; t++) {
                if (obs[t] = i) {
                    sum += ggamma[t*TIME_STEPS + j];
                }
            }
            emit_prob[i*OBS_SIZE + j] = sum / ggamma_sum[j];
        }
    }

    return;
}


void compute(
    unsigned int iterations,
    unsigned int state_size,
    unsigned int obs_size,
    unsigned int time_steps,
    int* obs,
    double* init_prob,
    double* trans_prob,
    double* emit_prob,
    double* alpha,
    double* beta,
    double* ggamma,
    double* sigma
    ) {

    STATE_SIZE = state_size;
    OBS_SIZE = obs_size;
    TIME_STEPS = time_steps;

    for (unsigned int i = 0; i < iterations; i++) {
        forward(obs, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
        backward(obs, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
        update(obs, init_prob, trans_prob, emit_prob, alpha, beta, ggamma, sigma);
    }

    return;
}
