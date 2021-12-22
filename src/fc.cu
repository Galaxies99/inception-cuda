# include "fc.h"

FullyConnectedLayer :: FullyConnectedLayer(int _in_features, int _out_features) {
    in_features = _in_features;
    out_features = _out_features;

    weight_N = in_features * out_features;
    bias_N = out_features;
    output_N = _out_features;

    cudaMalloc(&weight, sizeof(float) * weight_N);
    cudaMalloc(&bias, sizeof(float) * bias_N);
    cudaMalloc(&output, sizeof(float) * output_N);
    set_params(true, nullptr, nullptr);
}

FullyConnectedLayer :: ~FullyConnectedLayer() {
    cudaFree(weight);
    cudaFree(bias);
    cudaFree(output);
}

// FC forward (weight) (basic, no optimization)
__global__ void fc_basic_weight_forward(float *input, float *output, float *weight, const int in_features, const int out_features) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int total_N = in_features * out_features;
    const int begin_idx = total_N * thread_pos / total_threads, end_idx = total_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++i) {
        int temp = i;
        const int col = temp % in_features;
        const int row = (temp / in_features) % out_features;
        atomicAdd(&output[row], weight[row * in_features + col] * input[col]);
    }
}

// FC forward (bias) (basic, no optimization)
__global__ void fc_basic_bias_forward(float *input, float *output, float *bias, const int out_features) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int output_N = out_features;
    const int begin_idx = output_N * thread_pos / total_threads, end_idx = output_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++i) 
        output[i] = input[i] + bias[i];
}

void FullyConnectedLayer :: basic_forward(dim3 grid, dim3 block, float *input) {
    fc_basic_weight_forward <<<grid, block>>> (input, output, weight, in_features, out_features);
    fc_basic_bias_forward <<<grid, block>>> (output, output, bias, out_features);
}

void FullyConnectedLayer :: set_params(bool init = true, float *h_weight = nullptr, float *h_bias = nullptr) {
    if (init) {
        for (int i = 0; i < out_features; ++i) {
            bias[i] = init_rand();
            for (int j = 0; j < in_features; ++j)
                weight[i * in_features + j] = init_rand();
        }
    } else {
        cudaMemcpy(weight, h_weight, sizeof(float) * weight_N, cudaMemcpyHostToDevice);
        cudaMemcpy(bias, h_bias, sizeof(float) * bias_N, cudaMemcpyHostToDevice);
    }
}
