# include "activation.h"


__device__ float activation_relu(float x) {
    return x < 0 ? 0 : x;
}

float activation_relu_cpu(float x) {
    return x < 0 ? 0 : x;
}

__global__ void forward_relu(float *input, float *output, const int size) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int begin_idx = size * thread_pos / total_threads;
    const int end_idx = size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i)
        output[i] = activation_relu(input[i]);
}

float* cpu_relu(float *input, const int size, bool inplace) {
    float *output;
    if (inplace) output = input;
    else output = (float*) malloc (sizeof(float) * size);
    for (int i = 0; i < size; ++ i)
        output[i] = activation_relu_cpu(input[i]);
    return output;
}

float* relu(dim3 grid, dim3 block, float *input, const int size, bool inplace) {
    float *output;
    if (inplace) output = input;
    else cudaMalloc((void **)&output, sizeof(float) * size);
    forward_relu <<<grid, block>>> (input, output, size);
    return output;
}