# ifndef _LAYER_H_
# define _LAYER_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include "utils.h"
# include <iostream>
# include <stdio.h>
using namespace std;
# endif


// Convolution
class ConvolutionLayer {
    private:
        int in_channels, out_channels, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c;
        int size_r, size_c, out_size_r, out_size_c;
        int channel_N, kernel_N, input_N, output_N, total_N;
        float *weight, *bias;
        float *h_weight, *h_bias;
    
    public:
        ConvolutionLayer(int _in_channels, int _out_channels, int _size_r, int _size_c, int _kernel_size_r = 1, int _kernel_size_c = 1, int _stride_r = 1, int _stride_c = 1, int _padding_r = 0, int _padding_c = 0);
        float* basic_forward(dim3 grid, dim3 block, float *input, const int batch_size);
        float* cpu_forward(float *input, const int batch_size);
        void set_params(float *_h_weight = NULL, float *_h_bias = NULL);
        void get_output_size(int &output_r, int &output_c);
        ~ ConvolutionLayer();
};

void conv_forward_cpu(float *input, float *output, float *weight, float *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);
__global__ void conv_forward_basic_weight(float *input, float *output, float *weight, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);
__global__ void conv_forward_basic_bias(float *input, float *output, float *bias, const int batch_size, const int in_channels, const int out_channels, const int size_r, const int size_c, const int out_size_r, const int out_size_c, const int kernel_size_r, const int kernel_size_c, const int stride_r, const int stride_c, const int padding_r, const int padding_c);