# include "conv.h"

ConvolutionLayer :: ConvolutionLayer(int _in_channels, int _out_channels, int _size, int _kernel_size, int _stride = 1, int _padding = 0) {
    in_channels = _in_channels;
    out_channels = _out_channels;
    kernel_size = _kernel_size;
    stride = _stride;
    padding = _padding;
    size = _size;
    out_size = (size - kernel_size + 2 * padding) / stride + 1;
    
    channel_N = in_channels * out_channels;
    kernel_N = kernel_size * kernel_size * size * size;
    output_N = out_size * out_size * out_channels;

    cudaMalloc(&weight, sizeof(float) * channel_N * kernel_N);
    cudaMalloc(&bias, sizeof(float) * channel_N);
    cudaMalloc(&output, sizeof(float) * output_N);
    cudaMalloc(&grad_weight, sizeof(float) * channel_N * kernel_N);
    cudaMalloc(&grad_bias, sizeof(float) * channel_N);
    cudaMalloc(&grad_output, sizeof(float) * output_N);
    reset_params();
}

ConvolutionLayer :: ~ConvolutionLayer() {
    cudaFree(weight);
    cudaFree(bias);
    cudaFree(output);
    cudaFree(grad_weight);
    cudaFree(grad_bias);
    cudaFree(grad_output);
    free(h_weight);
    free(h_bias);
}

// Convolution forward (weight) (basic, no optimization)
__global__ void conv_basic_weight_forward(float *input, float *output, float *weight, const int in_channels, const int out_channels, const int size, const int out_size, const int kernel_size, const int stride, const int padding) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int total_N = kernel_size * kernel_size * out_size * out_size * in_channels * out_channels;
    const int begin_idx = total_N * thread_pos / total_threads, end_idx = total_N * (thread_pos + 1) / total_threads;
    const int offset = (kernel_size - 1) / stride - 2 * padding;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int i_kernel_r = temp % kernel_size;
        const int i_kernel_c = (temp /= kernel_size) % kernel_size;
        const int i_in_channel = (temp /= kernel_size) % in_channels;
        const int i_out_channel = (temp /= in_channels) % out_channels;
        const int i_out_r = (temp /= out_channels) % out_size;
        const int i_out_c = (temp /= out_size) % out_size;
        const int input_r = i_out_r * stride + i_kernel_r + offset;
        const int input_c = i_out_c * stride + i_kernel_c + offset;
        const int i_channel = i_out_channel * out_channels + i_in_channel;
        if (input_r >= 0 && input_c < size && output_r >= 0 && output_c < size)
            atomicAdd(&output[i_out_channel * out_size * out_size + i_out_c * out_size + i_out_r], weight[(i_channel * kernel_size + i_kernel_c) * kernel_size + i_kernel_r] * input[(i_channel * size + input_c) * size + input_r]);
    }
}

// Convolution forward (bias) (basic, no optimization)
__global__ void conv_basic_bias_forward(float *input, float *output, float *bias, const int out_channels, const int out_size) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int output_N = out_size * out_size * out_channels;
    const int begin_idx = output_N * thread_pos / total_threads, end_idx = output_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int i_channel = temp % out_channels;
        const int i_out_r = (temp /= out_channels) % out_size;
        const int i_out_c = (temp /= out_size) % out_size;
        output[(i_channel * out_size + i_out_c) * size + i_out_r] = input[(i_channel * out_size + i_out_c) * size + i_out_r] + bias[i_channel];
    }
}

void ConvolutionLayer :: basic_forward(dim3 grid, dim3 block, float *input) {
    conv_basic_weight_forward <<<grid, block>>> (input, output, weight, in_channels, out_channels, size, out_size, kernel_size, stride, padding);
    conv_basic_bias_forward <<<grid, block>>> (output, output, bias, out_channels, out_size);
}

void ConvolutionLayer :: reset_params(void) {
    for (int i = 0; i < channel_N; ++ i) {
        bias[i] = init_rand();
        for (int j = 0; j < kernel_N; ++ j)
            weight[i * kernel_N + j] = init_rand();
    }
}

void ConvolutionLayer :: clear(void) {
    cudaMemset(output, 0x00, sizeof(float) * output_N);
}

void ConvolutionLayer :: clear_grad(void) {
    cudaMemset(grad_weight, 0x00, sizeof(float) * channel_N * kernel_N);
    cudaMemset(grad_bias, 0x00, sizeof(float) * channel_N);
    cudaMemset(grad_output, 0x00, sizeof(float) * output_N);
}