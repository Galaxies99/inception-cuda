# include "fc.h"

// Construction function of fully connected layer.
FullyConnectedLayer :: FullyConnectedLayer(int _in_features, int _out_features) {
    in_features = _in_features;
    out_features = _out_features;

    weight_N = in_features * out_features;
    bias_N = out_features;
    output_N = _out_features;

    cudaMalloc(&weight, sizeof(float) * weight_N);
    cudaMalloc(&bias, sizeof(float) * bias_N);
    set_params();
}

// Destruction function of fully connected layer.
FullyConnectedLayer :: ~FullyConnectedLayer() {
    cudaFree(weight);
    cudaFree(bias);
    if (h_weight != NULL) free(h_weight);
    if (h_bias != NULL) free(h_bias);
}

// FC forward (cpu)
void fc_forward_cpu(float *input, float *output, float *weight, float *bias, const int batch_size, const int in_features, const int out_features) {
    const int input_N = in_features;
    const int output_N = out_features;
    for (int b = 0; b < batch_size; ++b) {
        for (int row = 0; row < out_features; ++row) {
            output[b * output_N + row] = bias[row];
            for (int col = 0; col < in_features; ++col) {
                output[b * output_N + row] += weight[row * in_features + col] * input[b * input_N + col];
            }
        }
    }
}

float* FullyConnectedLayer :: cpu_forward(float *input, const int batch_size) {
    float *output;
    output = (float *) malloc (sizeof(float) * batch_size * output_N);
    fc_forward_cpu(input, output, h_weight, h_bias, batch_size, in_features, out_features);
    return output;
}

// FC forward (weight) (basic, no optimization)
__global__ void fc_basic_weight_forward(float *input, float *output, float *weight, const int batch_size, const int in_features, const int out_features) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int input_N = in_features;
    const int output_N = out_features;
    const int total_N = in_features * out_features;
    const int begin_idx = total_N * thread_pos / total_threads;
    const int end_idx = total_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++i) {
        const int col = i % in_features;
        const int row = (i / in_features) % out_features;
        atomicAdd(&output[batch_id * output_N + row], weight[row * in_features + col] * input[batch_id * input_N + col]);
    }
}

// FC forward (bias) (basic, no optimization)
__global__ void fc_basic_bias_forward(float *input, float *output, float *bias, const int batch_size, const int in_features, const int out_features) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int output_N = out_features;
    const int begin_idx = output_N * thread_pos / total_threads;
    const int end_idx = output_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++i) {
        const int idx = i % out_features;
        output[batch_id * output_N + idx] = input[batch_id * output_N + idx] + bias[idx];
    }
}

float* FullyConnectedLayer :: basic_forward(dim3 grid, dim3 block, float *input, const int batch_size) {
    float *output;
    cudaMalloc((void **)&output, sizeof(float) * batch_size * output_N);
    cudaMemset(output, 0, sizeof(float) * batch_size * output_N);
    fc_basic_weight_forward <<<grid, block>>> (input, output, weight, batch_size, in_features, out_features);
    fc_basic_bias_forward <<<grid, block>>> (output, output, bias, batch_size, in_features, out_features);
    cudaDeviceSynchronize();
    return output;
}

void FullyConnectedLayer :: set_params(float *_h_weight, float *_h_bias) {
    if (_h_weight == NULL) {
        h_weight = (float*) malloc (sizeof(float) * weight_N);
        for (int i = 0; i < weight_N; ++i)
            h_weight[i] = init_rand();
    } else {
        if (h_weight != NULL) free(h_weight);
        h_weight = _h_weight;
    }
    if (_h_bias == NULL) {
        h_bias = (float*) malloc (sizeof(float) * bias_N);
        for (int i = 0; i < bias_N; ++i)
            h_bias[i] = init_rand();
    } else {
        if (h_bias != NULL) free(h_bias);
        h_bias = _h_bias;
    }
    cudaMemcpy(weight, h_weight, sizeof(float) * weight_N, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, sizeof(float) * bias_N, cudaMemcpyHostToDevice);
}
