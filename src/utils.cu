# include "utils.h"


float init_rand(void) {
    return 0.5f - float(rand()) / float(RAND_MAX);
}

__device__ float activation_sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

__device__ float activation_relu(float x) {
    return x < 0 ? 0 : x;
}
