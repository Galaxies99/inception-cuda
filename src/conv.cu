
   
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
    kernel_N = kernel_size_r * kernel_size_c;
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
                    const int output_idx = b * output_N + (out_ch * out_size_r + r) * out_size_c + c;
                    output[output_idx] = bias[out_ch];
                    for (int kr = 0; kr < kernel_size_r; ++ kr) {
                        for (int kc = 0; kc < kernel_size_c; ++ kc) {
                            const int input_r = r * stride_r + kr - padding_r;
                            const int input_c = c * stride_c + kc - padding_c;
                            if (input_r >= 0 && input_r < size_r && input_c >= 0 && input_c < size_c) {
                                for (int in_ch = 0; in_ch < in_channels; ++ in_ch) {
                                    output[output_idx] += weight[((out_ch * in_channels + in_ch) * kernel_size_r + kr) * kernel_size_c + kc] * input[b * input_N + (in_ch * size_r + input_r) * size_c + input_c];
                                }
                            }
                        }
                    }
                }
    }
}

double* ConvolutionLayer :: cpu_im2col_forward(double *input, const int batch_size) {
    double *im2col = cpu_im2col(input, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c);
    double *output = cpu_gemm(im2col, h_weight, h_bias, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c);
    free(im2col);
    return output;
}

double* ConvolutionLayer :: cpu_basic_forward(double *input, const int batch_size) {
    double *output;
    output = (double *) malloc (sizeof(double) * batch_size * output_N);
    conv_forward_cpu(input, output, h_weight, h_bias, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c);
    return output;
}

double* ConvolutionLayer :: cpu_forward(double *input, const int batch_size) {
    const long long workspace_size = 1ll * batch_size * out_channels * (out_size_r * out_size_c) * (in_channels * kernel_size_r * kernel_size_c);
    if (workspace_size <= 100000000ll) return cpu_im2col_forward(input, batch_size);
    else return cpu_basic_forward(input, batch_size);
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

void conv_cudnn_forward(
    cudnnHandle_t& handle,
    double *input,
    double *output,
    double *weight,
    double *bias,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int size_r,
    const int size_c,
    const int kernel_size_r,
    const int kernel_size_c,
    const int stride_r,
    const int stride_c,
    const int padding_r,
    const int padding_c,
    const int out_size_r,
    const int out_size_c
) {
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(
        cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_DOUBLE,
            batch_size,
            in_channels,
            size_r,
            size_c
        )
    );
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(
        cudnnSetFilter4dDescriptor(
            kernel_descriptor,
            CUDNN_DATA_DOUBLE,
            CUDNN_TENSOR_NCHW,
            out_channels,
            in_channels,
            kernel_size_r,
            kernel_size_c
        )
    );
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(
        cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            padding_r,
            padding_c,
            stride_r,
            stride_c,
            1,
            1,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_DOUBLE
        )
    );
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(
        cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_DOUBLE,
            batch_size,
            out_channels,
            out_size_r,
            out_size_c
        )
    );
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm_perf[4];
    int returned_cnt;
    checkCUDNN(
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            4,
            &returned_cnt,
            convolution_algorithm_perf
        )
    );
    bool found_algo = false;
    for (int n = 0; n < returned_cnt; ++ n) {
        if (convolution_algorithm_perf[n].status == CUDNN_STATUS_SUCCESS && 
            convolution_algorithm_perf[n].algo != CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) {
            convolution_algorithm = convolution_algorithm_perf[n].algo;
            found_algo = true;
            break;
        }
    }
    if (! found_algo) {
        std :: cerr << "No convolution algorithm is found." << std :: endl;
        std :: exit(EXIT_FAILURE);
    }
    size_t workspace_bytes;
    checkCUDNN(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            input_descriptor,
            kernel_descriptor,
            convolution_descriptor,
            output_descriptor,
            convolution_algorithm,
            &workspace_bytes
        )
    );
    if (workspace_bytes == 0) workspace_bytes = (size_t)(1 << 23);
    void *workspace = NULL;
    cudaMalloc(&workspace, workspace_bytes);
    const double alpha = 1.0, beta = 0.0;
    const size_t output_bytes = sizeof(double) * batch_size * out_channels * out_size_r * out_size_c;
    checkCUDNN(
        cudnnConvolutionForward(
            handle,
            &alpha,
            input_descriptor,
            input,
            kernel_descriptor,
            weight,
            convolution_descriptor,
            convolution_algorithm,
            workspace,
            output_bytes,
            &beta,
            output_descriptor,
            output
        )
    );

    cudaFree(workspace);

    if (bias != NULL) {
        cudnnTensorDescriptor_t bias_descriptor;
        checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
        checkCUDNN(
            cudnnSetTensor4dDescriptor(
                bias_descriptor,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_DOUBLE,
                1,
                out_channels,
                1,
                1
            )
        );
        checkCUDNN(
            cudnnAddTensor(
                handle,
                &alpha,
                bias_descriptor,
                bias,
                &alpha,
                output_descriptor,
                output
            )
        );
        cudnnDestroyTensorDescriptor(bias_descriptor);
    }

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

double* ConvolutionLayer :: cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size) {    
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_N);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_N);
    conv_cudnn_forward(handle, input, output, weight, bias, batch_size, in_channels, out_channels, size_r, size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c, out_size_r, out_size_c);
    return output;
}

double* cpu_im2col(double *input, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c) {
    const int input_N = in_channels * size_r * size_c;
    const int im2col_row = in_channels * kernel_size_r * kernel_size_c;
    const int im2col_col = out_size_r * out_size_c;
    double *output = (double*) malloc (sizeof(double) * batch_size * im2col_row * im2col_col);
    for (int b = 0; b < batch_size; ++ b) 
        for (int in_ch = 0; in_ch < in_channels; ++ in_ch) 
            for (int kr = 0; kr < kernel_size_r; ++ kr)
                for (int kc = 0; kc < kernel_size_c; ++ kc) 
                    for (int r = 0; r < out_size_r; ++ r)
                        for (int c = 0; c < out_size_c; ++ c) {
                            const int im2col_r = in_ch * kernel_size_r * kernel_size_c + kr * kernel_size_c + kc;
                            const int im2col_c = r * out_size_c + c;
                            const int input_r = r * stride_r + kr - padding_r;
                            const int input_c = c * stride_c + kc - padding_c;
                            if (input_r >= 0 && input_r < size_r && input_c >= 0 && input_c < size_c) 
                                output[b * im2col_row * im2col_col + im2col_r * im2col_col + im2col_c] = input[b * input_N + (in_ch * size_r + input_r) * size_c + input_c];
                            else output[b * im2col_row * im2col_col + im2col_r * im2col_col + im2col_c] = 0;
                        }

    return output;
}

double *cpu_gemm(double *input, double *weight, double *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c) {
    const int output_N = out_channels * out_size_r * out_size_c;
    const int im2col_row = in_channels * kernel_size_r * kernel_size_c;
    const int im2col_col = out_size_r * out_size_c;
    double *output = (double*) malloc (sizeof(double) * batch_size * output_N);
    for (int b = 0; b < batch_size; ++ b) {
        for (int out_ch = 0; out_ch < out_channels; ++ out_ch) {
            for (int r = 0; r < out_size_r; ++ r)
                for (int c = 0; c < out_size_c; ++ c) {
                    const int im2col_c = r * out_size_c + c;
                    const int output_idx = b * output_N + (out_ch * out_size_r + r) * out_size_c + c;
                    output[output_idx] = bias[out_ch];
                    for (int im2col_r = 0; im2col_r < im2col_row; ++ im2col_r)
                        output[output_idx] += input[(b * im2col_row + im2col_r) * im2col_col + im2col_c] * weight[out_ch * im2col_row + im2col_r];
                }
        }
    }
    return output;
}

__global__ void conv_forward_implicit_im2col(double *input, double *output, double *weight, double *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c) {
    const int im2col_row = in_channels * kernel_size_r * kernel_size_c;
    const int im2col_col = out_size_r * out_size_c;
    __shared__ double _kernel_tile_[TILE_WIDTH][TILE_WIDTH];
    __shared__ double _input_tile_[TILE_WIDTH][TILE_WIDTH];
    const int batch_idx = blockIdx.z;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_ch = by * TILE_WIDTH + ty;
    const int im2col_c = bx * TILE_WIDTH + tx;
    const int out_c = im2col_c % out_size_c;
    const int out_r = (im2col_c / out_size_c) % out_size_r;
    int ito = (im2col_row / TILE_WIDTH) + (im2col_row % TILE_WIDTH != 0);
    double value = bias[out_ch];
    for (int i = 0; i < ito; ++ i) {
        const int im2col_r = i * TILE_WIDTH + ty;
        const int kc = im2col_r % kernel_size_c;
        const int kr = (im2col_r / kernel_size_c) % kernel_size_r;
        const int in_ch = (im2col_r / kernel_size_c / kernel_size_r) % in_channels;
        const int input_r = out_r * stride_r + kr - padding_r;
        const int input_c = out_c * stride_c + kc - padding_c; 
        if (i * TILE_WIDTH + tx >= im2col_row) _kernel_tile_[ty][tx] = 0;
        else _kernel_tile_[ty][tx] = weight[out_ch * im2col_row + i * TILE_WIDTH + tx];
        if (im2col_r >= im2col_row || input_r < 0 || input_r >= size_r || input_c < 0 || input_c >= size_c) _input_tile_[ty][tx] = 0;
        else _input_tile_[ty][tx] = input[((batch_idx * in_channels + in_ch) * size_r + input_r) * size_c + input_c];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++ k)
            value += _kernel_tile_[ty][k] * _input_tile_[k][tx];
        __syncthreads();
    }
    if (out_ch < out_channels && im2col_c < im2col_col)output[((batch_idx * out_channels + out_ch) * out_size_r + out_r) * out_size_c + out_c] = value;
}

double* ConvolutionLayer :: implicit_im2col_forward(double *input, const int batch_size) {
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_N);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_N);
    const int dimy = out_channels / TILE_WIDTH + (out_channels % TILE_WIDTH != 0);
    const int dimx = (out_size_r * out_size_c) / TILE_WIDTH + ((out_size_r * out_size_c) % TILE_WIDTH != 0);
    dim3 grid(dimx, dimy, batch_size), block(TILE_WIDTH, TILE_WIDTH);
    conv_forward_implicit_im2col <<<grid, block>>> (input, output, weight, bias, batch_size, in_channels, out_channels, size_r, size_c, out_size_r, out_size_c, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c);
    cudaDeviceSynchronize();
    return output;
}