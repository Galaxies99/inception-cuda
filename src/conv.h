# ifndef _CONV_H_
# define _CONV_H_
# include <cuda.h>
# include <cudnn.h>
# include <cudnn_v8.h>
# include <stdio.h>

# include "utils.h"
# include "cudnn_utils.h"
# include "cuda_runtime.h"

// Convolution
class ConvolutionLayer {
    private:
        int in_channels, out_channels, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c;
        int size_r, size_c, out_size_r, out_size_c;
        int channel_N, kernel_N, input_N, output_N, total_N;
        double *weight, *bias;
        double *h_weight, *h_bias;
    
    public:
        ConvolutionLayer(int _in_channels, int _out_channels, int _size_r, int _size_c, int _kernel_size_r = 1, int _kernel_size_c = 1, int _stride_r = 1, int _stride_c = 1, int _padding_r = 0, int _padding_c = 0);
        double* basic_forward(dim3 grid, dim3 block, double *input, const int batch_size);
        double* cpu_forward(double *input, const int batch_size);
        // double* cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size);
        void set_params(double *_h_weight = NULL, double *_h_bias = NULL);
        void get_output_size(int &output_r, int &output_c);
        ~ ConvolutionLayer();
};

void conv_forward_cpu(double *input, double *output, double *weight, double *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);

__global__ void conv_forward_basic_weight(double *input, double *output, double *weight, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);

__global__ void conv_forward_basic_bias(double *input, double *output, double *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);

// void conv_cudnn_forward(cudnnHandle_t& handle, double *input, double *output, double *weight, double *bias, const int batch_size, const int in_channels, const int size_r, const int size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);

# endif