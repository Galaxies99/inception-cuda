# include <cuda.h>

# ifndef _LAYER_H_
# define _LAYER_H_
# endif


// Convolution Layer with stride 1.
class BasicConvLayer {
    private:
        int kernel_size, in_channels, out_channels;
        int size;
        float *weight, *bias;
    
    public:
        BasicConvLayer(int _in_channels, int _out_channels, int _kernel_size, int _size);
        ~ BasicConvLayer();
}

__global__ conv_forward(...);
__global__ conv_backward(...);