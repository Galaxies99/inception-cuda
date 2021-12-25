# include "activation.h"


__device__ double activation_relu(double x) {
    return x < 0 ? 0 : x;
}

double activation_relu_cpu(double x) {
    return x < 0 ? 0 : x;
}

__global__ void forward_relu(double *input, double *output, const int size) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int begin_idx = 1ll * size * thread_pos / total_threads;
    const int end_idx = 1ll * size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i)
        output[i] = activation_relu(input[i]);
}

double* cpu_relu(double *input, const int size, bool inplace) {
    double *output;
    if (inplace) output = input;
    else output = (double*) malloc (sizeof(double) * size);
    for (int i = 0; i < size; ++ i)
        output[i] = activation_relu_cpu(input[i]);
    return output;
}

double* relu(dim3 grid, dim3 block, double *input, const int size, bool inplace) {
    double *output;
    if (inplace) output = input;
    else cudaMalloc((void **)&output, sizeof(double) * size);
    forward_relu <<<grid, block>>> (input, output, size);
    return output;
}