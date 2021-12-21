# ifndef _LAYER_H_
# define _LAYER_H_
# include "cuda_runtime.h"
# include "utils.h"
# endif


// Convolution Layer (Square)
class ConvolutionLayer {
    private:
        int in_channels, out_channels, kernel_size, stride, padding;
        int size, out_size;
        int channel_N, kernel_N, output_N;
        float *weight, *bias, *output;
    
    public:
        ConvolutionLayer(int _in_channels, int _out_channels, int _size, int _kernel_size, int _stride, int _padding);
        void basic_forward(dim3 grid, dim3 block, float *input);
        void set_params(bool init, float *h_weight, float *h_bias);
        void clear_grad(void);
        void clear(void);
        ~ ConvolutionLayer();
}

__global__ void conv_basic_weight_forward(float*, float*, float*, const int, const int, const int, const int, const int, const int);
__global__ void conv_basic_bias_forward(float*, float*, float*, const int, const int);
