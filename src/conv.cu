# include "conv.h"

// Construction function of convolution layer.
ConvolutionLayer :: ConvolutionLayer(int _in_channels, int _out_channels, int _size_r, int _size_c, int _kernel_size_r, int _kernel_size_c, int _stride_r, int _stride_c, int _padding_r, int _padding_c) {
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

    cudaMalloc((void **)&weight, sizeof(double) * channel_N * kernel_N);
    cudaMalloc((void **)&bias, sizeof(double) * out_channels);
    set_params();
}

// Destruction function of convolution layer.
ConvolutionLayer :: ~ConvolutionLayer() {
    cudaFree(weight);
    cudaFree(bias);
    if (h_weight != NULL) free(h_weight);
    if (h_bias != NULL) free(h_bias);
}

// Convolution forward (cpu)
void conv_forward_cpu(double* input, double *output, double *weight, double *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c) {
    const int output_N = out_channels * out_size_r * out_size_c;
    const int input_N = in_channels * size_r * size_c;
    for (int b = 0; b < batch_size; ++ b) {
        for (int out_ch = 0; out_ch < out_channels; ++ out_ch)
            for (int r = 0; r < out_size_r; ++ r)
                for (int c = 0; c < out_size_c; ++ c) {
                    output[b * output_N + (out_ch * out_size_r + r) * out_size_c + c] = bias[out_ch];
                    for (int kr = 0; kr < kernel_size_r; ++ kr) {
                        for (int kc = 0; kc < kernel_size_c; ++ kc) {
                            const int input_r = r * stride_r + kr - padding_r;
                            const int input_c = c * stride_c + kc - padding_c;
                            if (input_r >= 0 && input_r < size_r && input_c >= 0 && input_c < size_c) {
                                for (int in_ch = 0; in_ch < in_channels; ++ in_ch) {
                                    output[b * output_N + (out_ch * out_size_r + r) * out_size_c + c] += weight[((out_ch * in_channels + in_ch) * kernel_size_r + kr) * kernel_size_c + kc] * input[b * input_N + (in_ch * size_r + input_r) * size_c + input_c];
                                }
                            }
                        }
                    }
                }
    }
}

double* ConvolutionLayer :: cpu_forward(double *input, const int batch_size) {
    double *output;
    output = (double *) malloc (sizeof(double) * batch_size * output_N);
    conv_forward_cpu(input, output, h_weight, h_bias, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c);
    return output;
}

// Convolution forward (weight) (basic, no optimization)
__global__ void conv_forward_basic_weight(double *input, double *output, double *weight, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int output_N = out_channels * out_size_r * out_size_c;
    const int input_N = in_channels * size_r * size_c;
    const long long total_N = 1ll * kernel_size_r * kernel_size_c * out_size_r * out_size_c * in_channels * out_channels;
    const long long begin_idx = total_N * thread_pos / total_threads;
    const long long end_idx = total_N * (thread_pos + 1) / total_threads;
    for (long long i = begin_idx; i < end_idx; ++ i) {
        long long temp = i;
        const int i_kernel_c = temp % kernel_size_c;
        const int i_kernel_r = (temp /= kernel_size_c) % kernel_size_r;
        const int i_in_channel = (temp /= kernel_size_r) % in_channels;
        const int i_out_channel = (temp /= in_channels) % out_channels;
        const int i_out_c = (temp /= out_channels) % out_size_c;
        const int i_out_r = (temp /= out_size_c) % out_size_r;
        const int input_c = i_out_c * stride_c + i_kernel_c - padding_c;
        const int input_r = i_out_r * stride_r + i_kernel_r - padding_r;
        const int i_channel = i_out_channel * in_channels + i_in_channel;
        if (input_r >= 0 && input_r < size_r && input_c >= 0 && input_c < size_c)
            atomicAdd(
                &output[batch_id * output_N + (i_out_channel * out_size_r + i_out_r) * out_size_c + i_out_c], 
                weight[(i_channel * kernel_size_r + i_kernel_r) * kernel_size_c + i_kernel_c] * input[batch_id * input_N + (i_in_channel * size_r + input_r) * size_c + input_c]
            );
    }
}

// Convolution forward (bias) (basic, no optimization)
__global__ void conv_forward_basic_bias(double *input, double *output, double *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int output_N = out_channels * out_size_r * out_size_c;
    const int begin_idx = output_N * thread_pos / total_threads;
    const int end_idx = output_N * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int i_channel = temp % out_channels;
        const int i_out_c = (temp /= out_channels) % out_size_c;
        const int i_out_r = (temp /= out_size_c) % out_size_r;
        output[batch_id * output_N + (i_channel * out_size_r + i_out_r) * out_size_c + i_out_c] = input[batch_id * output_N + (i_channel * out_size_r + i_out_r) * out_size_c + i_out_c] + bias[i_channel];
    }
}

double* ConvolutionLayer :: basic_forward(dim3 grid, dim3 block, double *input, const int batch_size) {
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_N);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_N);
    conv_forward_basic_weight <<<grid, block>>> (input, output, weight, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c);
    conv_forward_basic_bias <<<grid, block>>> (output, output, bias, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c);
    cudaDeviceSynchronize();
    return output;
}

void ConvolutionLayer :: set_params(double *_h_weight, double *_h_bias) {
    if (_h_weight == NULL) {
        h_weight = (double*) malloc (sizeof(double) * channel_N * kernel_N);
        for (int i = 0; i < channel_N * kernel_N; ++ i)
            h_weight[i] = init_rand();
    } else {
        if (h_weight != NULL) free(h_weight);
        h_weight = _h_weight;
    }
    if (_h_bias == NULL) {
        h_bias = (double*) malloc (sizeof(double) * out_channels);
        for (int i = 0; i < out_channels; ++ i)
            h_bias[i] = init_rand();
    } else{
        if (h_bias != NULL) free(h_bias);
        h_bias = _h_bias;
    }
    cudaMemcpy(weight, h_weight, sizeof(double) * channel_N * kernel_N, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, sizeof(double) * out_channels, cudaMemcpyHostToDevice);
}

void ConvolutionLayer :: get_output_size(int &output_r, int &output_c) {
    output_r = out_size_r;
    output_c = out_size_c;
}
