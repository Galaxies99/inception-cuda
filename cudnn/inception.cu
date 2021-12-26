# include "inception.h"


void conv_forward(
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
    const bool with_relu
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

    const int out_size_r = (size_r - kernel_size_r + 2 * padding_r) / stride_r + 1;
    const int out_size_c = (size_c - kernel_size_c + 2 * padding_c) / stride_c + 1;

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
    checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm(
            handle,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
			&convolution_algorithm
        )
    );

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
    assert (workspace_bytes > 0);
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
            kernel,
            convolution_descriptor,
            convolution_algorithm,
            workspace,
            output_bytes,
            &beta,
            output_descriptor,
            output
        )
    );

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
                &beta,
                output_descriptor,
                output
            )
        );
        cudnnDestroyTensorDescriptor(bias_descriptor);
    }

    if (with_relu) {
        cudnnActivationDescriptor_t activation_descriptor;
        checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
        checkCUDNN(
            cudnnSetActivationDescriptor(
                activation_descriptor,
                CUDNN_ACTIVATION_RELU,
                CUDNN_PROPAGATE_NAN,
                0
            )
        );
        checkCUDNN(
            cudnnActivationForward(
                handle,
                activation_descriptor,
                &alpha,
                output_descriptor,
                output,
                &beta,
                output_descriptor,
                output
            )
        );
        cudnnDestroyActivationDescriptor(activation_descriptor);
    }

    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}