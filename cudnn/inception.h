# ifndef _INCEPTION_CUDNN_H_
# define _INCEPTION_CUDNN_H_
# include <cudnn.h>
# include <stdio.h>
# include "cuda_runtime.h"

void conv_forward_layer(
    cudnnHandle_t& handle,
    double *input,
    double *output,
    double *weight,
    double *bias,
    const int batch_size,
    const int in_channels,
    const int size_r,
    const int size_c,
    const int kernel_size_r,
    const int kernel_size_c,
    const int stride_r,
    const int stride_c,
    const int padding_r,
    const int padding_c
);

# endif