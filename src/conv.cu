# include "conv.h"

// Construction function of convolution layer.
ConvolutionLayer :: ConvolutionLayer(int _in_channels, int _out_channels, int _size_r, int _size_c, int _kernel_size_r = 1, int _kernel_size_c = 1, int _stride_r = 1, int _stride_c = 1, int _padding_r = 0, int _padding_c = 0) {
    in_channels = _in_channels;
    out_channels = _out_channels;
    kernel_size_r = _kernel_size_r;
    kernel_size_c = _kernel_size_c;
    stride_r = _stride_r;
    stride_c = _stride_c;
    padding_r = _padding_r;
    padding_c = _padding_c;
    size_r = _size_r;
    size_c = _size_c;
    out_size_r = (size_r - kernel_size_r + 2 * padding_r) / stride_r + 1;
    out_size_c = (size_c - kernel_size_c + 2 * padding_c) / stride_c + 1;
    
    channel_N = in_channels * out_channels;
    kernel_N = kernel_size_r * kernel_size_c * out_size_r * out_size_c;
    output_N = out_channels * out_size_r * out_size_c;
    input_N = in_channels * size_r * size_c;
    total_N = kernel_N * channel_N;

    cudaMalloc(&weight, sizeof(float) * channel_N * kernel_N);
    cudaMalloc(&bias, sizeof(float) * channel_N);
    set_params();
}

// Destruction function of convolution layer.
ConvolutionLayer :: ~ConvolutionLayer() {
    cudaFree(weight);
    cudaFree(bias);
    cudaFree(output);
}

// Convolution forward (cpu)
void conv_forward_cpu(float* input, float* output, const int batch_size, const ConvolutionLayer &layer) {
    
}

// Convolution forward (weight) (basic, no optimization)
__global__ void conv_forward_basic_weight(float *input, float *output, const int batch_size, const ConvolutionLayer &layer) {
    const int batch_id = blockIdx.x;
    const int thread_pos = blockIdx.y * blockDim.y + threadIdx.x;
    const int total_threads = blockDim.y * gridDim.x;
    const int offset_r = (layer.kernel_size_r - 1) / layer.stride_r - 2 * layer.padding_r;
    const int offset_c = (layer.kernel_size_c - 1) / layer.stride_c - 2 * layer.padding_c;
    const int begin_idx = layer.total_N * thread_pos / total_threads;
    const int end_idx = layer.total_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int i_kernel_c = temp % layer.kernel_size_c;
        const int i_kernel_r = (temp /= layer.kernel_size_c) % layer.kernel_size_r;
        const int i_in_channel = (temp /= layer.kernel_size_r) % layer.in_channels;
        const int i_out_channel = (temp /= layer.in_channels) % layer.out_channels;
        const int i_out_c = (temp /= layer.out_channels) % layer.out_size_c;
        const int i_out_r = (temp /= layer.out_size_c) % layer.out_size_r;
        const int input_c = i_out_c * layer.stride_c + i_kernel_c + offset_c;
        const int input_r = i_out_r * layer.stride_r + i_kernel_r + offset_r;
        const int i_channel = i_out_channel * layer.in_channels + i_in_channel;
        if (input_r >= 0 && input_c < size && output_r >= 0 && output_c < size)
            atomicAdd(
                &output[((batch_id * layer.output_N + i_out_channel) * layer.out_size_r + i_out_r) * layer.out_size_c + i_out_c], 
                weight[(i_channel * layer.kernel_size_r + i_kernel_r) * layer.kernel_size_c + i_kernel_c] * input[((batch_id * layer.input_N + i_channel) * layer.size_r + input_r) * layer.size_c + input_c]
            );
    }
}

// Convolution forward (bias) (basic, no optimization)
__global__ void conv_forward_basic_bias(float *input, float *output, const int batch_size, const ConvolutionLayer &layer) {
    const int batch_id = blockIdx.x;
    const int thread_pos = blockIdx.y * blockDim.y + threadIdx.x;
    const int total_threads = blockDim.y * gridDim.x;
    const int begin_idx = layer.output_N * thread_pos / total_threads;
    const int end_idx = layer.output_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int i_channel = temp % layer.out_channels;
        const int i_out_c = (temp /= layer.out_channels) % layer.out_size_c;
        const int i_out_r = (temp /= layer.out_size_c) % layer.out_size_r;
        output[((batch_id * layer.output_N + i_channel) * layer.out_size_r + i_out_r) * layer.out_size_c + i_out_c] = input[((batch_id * layer.output_N + i_channel) * layer.out_size_r + i_out_r) * layer.out_size_c + i_out_c] + bias[i_channel];
    }
}

float* ConvolutionLayer :: basic_forward(dim3 grid, dim3 block, float *input, const int batch_size) {
    float *output;
    cudaMalloc(&output, sizeof(float) * batch_size * output_N)
    cudaMemset(output, 0x00, sizeof(float) * batch_size * output_N);
    conv_forward_basic_weight <<<grid, block>>> (input, output, batch_size, *this);
    conv_forward_basic_bias <<<grid, block>>> (output, output, batch_size, *this);
    return output;
}

void ConvolutionLayer :: set_params(bool init = true, float *h_weight = nullptr, float *h_bias = nullptr) {
    if (init) {
        for (int i = 0; i < channel_N; ++ i) {
            bias[i] = init_rand();
            for (int j = 0; j < kernel_N; ++ j)
                weight[i * kernel_N + j] = init_rand();
        }
    } else {
	    cudaMemcpy(weight, h_weight, sizeof(float) * channel_N * kernel_N, cudaMemcpyHostToDevice);
        cudaMemcpy(bias, h_bias, sizeof(float) * channel_N, cudaMemcpyHostToDevice);
    }
}

