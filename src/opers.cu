# include "opers.h"


__global__ void forward_linear_transform(float *input, float *output, const int size, const float alpha, const float beta) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int begin_idx = size * thread_pos / total_threads;
    const int end_idx = size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i)
        output[i] = input[i] * alpha + beta;
}

float* cpu_linear_transform(float *input, const int size, const float alpha, const float beta, bool inplace) {
    float *output;
    if (inplace) output = input;
    else output = (float*) malloc (sizeof(float) * size);
    for (int i = 0; i < size; ++ i)
        output[i] = input[i] * alpha + beta;
    return output;
}

float* linear_transform(dim3 grid, dim3 block, float *input, const int size, const float alpha, const float beta, bool inplace) {
    float *output;
    if (inplace) output = input;
    else cudaMalloc((void **)&output, sizeof(float) * size);
    forward_linear_transform <<<grid, block>>> (input, output, size, alpha, beta);
    return output;
}
