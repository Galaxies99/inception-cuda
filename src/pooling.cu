# include "pooling.h"

void maxpooling_cpu(double* input, double* output, const int batch_size, const int channel, const int size, const int kernel_size, const int stride) {
    const int out_size = (size - kernel_size) / stride + 1;
    const int input_N = channel * size * size;
    const int output_N = channel * out_size * out_size;
    double s;
    for (int b = 0; b < batch_size; ++ b)
        for (int ch = 0; ch < channel; ++ ch)
            for (int r = 0; r < out_size; ++ r)
                for (int c = 0; c < out_size; ++ c) {
                    s = -1e18;
                    for (int kr = 0; kr < kernel_size; ++ kr)
                        for (int kc = 0; kc < kernel_size; ++ kc) {
                            const int input_r = r * stride + kr;
                            const int input_c = c * stride + kc;
                            if (input_r >= 0 && input_r < size && input_c >= 0 && input_c < size) 
                                s = max(s, input[b * input_N + (ch * size + input_r) * size + input_c]);
                        }
                    output[b * output_N + (ch * out_size + r) * out_size + c] = s;
                }
}

void meanpooling_cpu(double* input, double* output, const int batch_size, const int channel, const int size, const int kernel_size, const int stride, const int padding) {
    const int out_size = (size - kernel_size + 2 * padding) / stride + 1;
    const int input_N = channel * size * size;
    const int output_N = channel * out_size * out_size;
    double s;
    for (int b = 0; b < batch_size; ++ b)
        for (int ch = 0; ch < channel; ++ ch)
            for (int r = 0; r < out_size; ++ r)
                for (int c = 0; c < out_size; ++ c) {
                    s = 0;
                    for (int kr = 0; kr < kernel_size; ++ kr)
                        for (int kc = 0; kc < kernel_size; ++ kc) {
                            const int input_r = r * stride + kr - padding;
                            const int input_c = c * stride + kc - padding;
                            if (input_r >= 0 && input_r < size && input_c >= 0 && input_c < size) 
                                s += input[b * input_N + (ch * size + input_r) * size + input_c];
                        }
                    output[b * output_N + (ch * out_size + r) * out_size + c] = s / (kernel_size * kernel_size);
                }
}

// batch * channel * height * width
__global__ void maxpool_forward(double* bottom_data, double* top_data, const int size, const int kernel_size, const int channels, const int stride) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    const int input_N = channels * size * size;
    const int begin_idx = 1ll * channels * thread_pos / total_threads;
    const int end_idx = 1ll * channels * (thread_pos+1) / total_threads;

    int len = (size - kernel_size) / stride + 1;
    const int output_N = channels * len * len;

    double s;
    int index, index2 = thread_pos * len * len;
    for (int c = begin_idx; c < end_idx; c++) {
        index2 = batch_id * output_N + c * len * len;
        for (int i = 0; i < len; ++i) {
            for (int j = 0; j < len; ++j) {
                index = batch_id * input_N + c * size * size + i * stride * size + j * stride;
                s = -1e18;
                for (int u = 0; u < kernel_size && (u + stride * i) < size; ++u)
                    for (int v = 0; v < kernel_size && (v + stride * j) < size; ++v)
                        if (*(bottom_data + index + u * size + v) > s)
                            s = *(bottom_data + index + u * size + v);
                *(top_data + index2) = s;
                ++index2;
            }
        }
    }
}

__global__ void meanpool_forward(double* bottom_data, double* top_data, const int size, const int kernel_size, const int channels, const int stride, const int padding) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    const int input_N = channels * size * size;
    const int begin_idx = 1ll * channels * thread_pos / total_threads;
    const int end_idx = 1ll * channels * (thread_pos+1) / total_threads;

    int len = (size + padding * 2 - kernel_size) / stride + 1;
    const int output_N = channels * len * len;

    double s;
    int index, index2 = thread_pos * len * len;
    for(int c = begin_idx; c < end_idx; c++) {
        index2 = batch_id * output_N + c * len * len;
        for (int i = 0; i < len; ++i){
            for (int j = 0; j < len; ++j) {
                index = batch_id * input_N + c * size * size + i * stride * size + j * stride;
                s = 0;
                for (int u = -padding; u < kernel_size-padding && (u + stride * i) < size; ++u)
                    for (int v = -padding; v < kernel_size-padding && (v + stride * j) < size; ++v)
                        if (i * stride + u >= 0 && j * stride + v >= 0 && i * stride + u < size && j * stride + v < size)
                            s += *(bottom_data + index + u * size + v) ;         
                *(top_data + index2) = s / (kernel_size * kernel_size);
                ++index2;
            }
        }
    }
}

// Construction function of convolution layer.
MaxpoolingLayer :: MaxpoolingLayer(int _channels, int _size, int _kernel_size, int _stride){
    channels = _channels;
    size = _size;
    kernel_size = _kernel_size;
    stride = _stride;
    int len = (size - kernel_size) / stride + 1;
    output_size = channels * len * len;
}
// Destruction function of maxpooling layer.
MaxpoolingLayer :: ~MaxpoolingLayer() {}

double* MaxpoolingLayer :: basic_forward(dim3 grid, dim3 block, double *input, const int batch_size) {
    double* output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_size);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_size);
    maxpool_forward <<<grid, block>>> (input, output, size, kernel_size, channels, stride);
    cudaDeviceSynchronize();
    return output;
}

double* MaxpoolingLayer :: cpu_forward(double *input, const int batch_size) {
    double *output;
    output = (double *) malloc (sizeof(double) * batch_size * output_size);
    maxpooling_cpu(input, output, batch_size, channels, size, kernel_size, stride);
    return output;
}

// Construction function of convolution layer.
MeanpoolingLayer :: MeanpoolingLayer(int _channels, int _size, int _kernel_size, int _stride, int _padding){
    channels = _channels;
    size = _size;
    kernel_size = _kernel_size;
    stride = _stride;
    padding = _padding;
    int len = (size + padding * 2 - kernel_size) / stride + 1;
    output_size = channels * len * len;
}
// Destruction function of meanpooling layer.
MeanpoolingLayer :: ~MeanpoolingLayer() {}

double* MeanpoolingLayer :: basic_forward(dim3 grid, dim3 block, double *input, const int batch_size) {
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_size);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_size);

    meanpool_forward <<<grid, block>>> (input, output, size, kernel_size, channels, stride, padding);
    cudaDeviceSynchronize();
    return output;
}

double* MeanpoolingLayer :: cpu_forward(double *input, const int batch_size) {
    double *output;
    output = (double *) malloc (sizeof(double) * batch_size * output_size);
    meanpooling_cpu(input, output, batch_size, channels, size, kernel_size, stride, padding);
    return output;
}

void pooling_cudnn_forward(
    cudnnHandle_t& handle,
    double *input,
    double *output,
    const int batch_size,
    const int channels,
    const int size,
    const int kernel_size,
    const int stride,
    const int padding,
    const cudnnPoolingMode_t mode
) {
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(
        cudnnSetTensor4dDescriptor(
            input_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_DOUBLE,
            batch_size,
            channels,
            size,
            size
        )
    );
    
    const int out_size = (size - kernel_size + 2 * padding) / stride + 1;
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(
        cudnnSetTensor4dDescriptor(
            output_descriptor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_DOUBLE,
            batch_size,
            channels,
            out_size,
            out_size
        )
    );

    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(
        cudnnSetPooling2dDescriptor(
            pooling_descriptor,
            mode,
            CUDNN_PROPAGATE_NAN,
            kernel_size,
            kernel_size,
            padding,
            padding,
            stride,
            stride
        )
    );

    const double alpha = 1.0, beta = 0.0;
    checkCUDNN(
        cudnnPoolingForward(
            handle,
            pooling_descriptor,
            &alpha,
            input_descriptor,
            input,
            &beta,
            output_descriptor,
            output
        )
    );
    
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
}

double* MaxpoolingLayer :: cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size) {
    double* output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_size);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_size);
    pooling_cudnn_forward(handle, input, output, batch_size, channels, size, kernel_size, stride, 0, CUDNN_POOLING_MAX);
    return output;
}

double* MeanpoolingLayer :: cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size) {
    double* output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_size);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_size);
    pooling_cudnn_forward(handle, input, output, batch_size, channels, size, kernel_size, stride, padding, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
    return output;
}
