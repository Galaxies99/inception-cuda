# ifndef _LAYER_H_
# define _LAYER_H_
# include "cuda_runtime.h"
# include "utils.h"
# endif


// Convolution
class ConvolutionLayer {
    private:
        int in_channels, out_channels, kernel_size_r, kernel_size_c, stride_r, stride_c, padding_r, padding_c;
        int size_r, size_c, out_size_r, out_size_c;
        int channel_N, kernel_N, input_N, output_N, total_N;
        float *weight, *bias;
    
    public:
        ConvolutionLayer(int, int, int, int, int, int, int, int, int, int);
        float* basic_forward(dim3, dim3, float*, const int);
        float* 
        void set_params(bool, float*, float*);
        ~ ConvolutionLayer();
}

void conv_forward_cpu(float*, float*, const int, const ConvolutionLayer&);
__global__ void conv_forward_basic_weight(float*, float*, const int, const ConvolutionLayer&);
__global__ void conv_forward_basic_bias(float*, float*, const int, const ConvolutionLayer&);
