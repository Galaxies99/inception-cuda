# include "fc.h"

// Construction function of fully connected layer.
FullyConnectedLayer :: FullyConnectedLayer(int _in_features, int _out_features) {
    in_features = _in_features;
    out_features = _out_features;

    weight_N = in_features * out_features;
    bias_N = out_features;
    output_N = _out_features;

    cudaMalloc(&weight, sizeof(double) * weight_N);
    cudaMalloc(&bias, sizeof(double) * bias_N);
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
void fc_forward_cpu(double *input, double *output, double *weight, double *bias, const int batch_size, const int in_features, const int out_features) {
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

double* FullyConnectedLayer :: cpu_forward(double *input, const int batch_size) {
    double *output;
    output = (double *) malloc (sizeof(double) * batch_size * output_N);
    fc_forward_cpu(input, output, h_weight, h_bias, batch_size, in_features, out_features);
    return output;
}

// FC forward (weight) (basic, no optimization)
__global__ void fc_basic_weight_forward(double *input, double *output, double *weight, const int batch_size, const int in_features, const int out_features) {
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
__global__ void fc_basic_bias_forward(double *input, double *output, double *bias, const int batch_size, const int in_features, const int out_features) {
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

double* FullyConnectedLayer :: basic_forward(dim3 grid, dim3 block, double *input, const int batch_size) {
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_N);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_N);
    fc_basic_weight_forward <<<grid, block>>> (input, output, weight, batch_size, in_features, out_features);
    fc_basic_bias_forward <<<grid, block>>> (output, output, bias, batch_size, in_features, out_features);
    cudaDeviceSynchronize();
    return output;
}

void FullyConnectedLayer :: set_params(double *_h_weight, double *_h_bias) {
    if (_h_weight == NULL) {
        h_weight = (double*) malloc (sizeof(double) * weight_N);
        for (int i = 0; i < weight_N; ++i)
            h_weight[i] = init_rand();
    } else {
        if (h_weight != NULL) free(h_weight);
        h_weight = _h_weight;
    }
    if (_h_bias == NULL) {
        h_bias = (double*) malloc (sizeof(double) * bias_N);
        for (int i = 0; i < bias_N; ++i)
            h_bias[i] = init_rand();
    } else {
        if (h_bias != NULL) free(h_bias);
        h_bias = _h_bias;
    }
    cudaMemcpy(weight, h_weight, sizeof(double) * weight_N, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, sizeof(double) * bias_N, cudaMemcpyHostToDevice);
}

void fc_cudnn_forward(
    cudnnHandle_t& handle,
    double *input,
    double *output,
    double *weight,
    double *bias,
    const int batch_size,
    const int in_channels,
    const int out_channels
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
            1,
            1
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
            1,
            1
        )
    );
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(
        cudnnSetConvolution2dDescriptor(
            convolution_descriptor,
            0,
            0,
            1,
            1,
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
            1,
            1
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
    const size_t output_bytes = sizeof(double) * batch_size * out_channels;
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

double* FullyConnectedLayer :: cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size) {
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * output_N);
    cudaMemset(output, 0, sizeof(double) * batch_size * output_N);
    fc_cudnn_forward(handle, input, output, weight, bias, batch_size, in_features, out_features);
    return output;
}
