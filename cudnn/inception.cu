# include "inception.h"

void conv_forward_layer(
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
    const int padding_c
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
}