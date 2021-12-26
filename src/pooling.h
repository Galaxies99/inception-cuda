# ifndef _POOLING_H_
# define _POOLING_H_
# include <cuda.h>
# include <cudnn.h>
# include <stdio.h>
# include <cudnn_v8.h>

# include "utils.h"
# include "cudnn_utils.h"
# include "cuda_runtime.h"

void maxpooling_cpu(double*, double*, int*, const int, const int, const int, const int, const int);
void meanpooling_cpu(double*, double*, const int, const int, const int, const int, const int, const int);

__global__ void maxpooling_forward(double*, double*, int*, const int, const int,const int, const int);
__global__ void meanpool_forward(double*, double*, const int, const int, const int, const int, const int);

void pooling_cudnn_forward(cudnnHandle_t& handle, double *input, double *output, const int batch_size, const int channels, const int size, const int kernel_size, const int stride, const int padding, const cudnnPoolingMode_t mode);

class MaxpoolingLayer{
    private:
        int channels, size, kernel_size, stride;
        int output_size;
    public:
        MaxpoolingLayer(int _channels, int _size, int _kernel_size, int _stride);
        double* basic_forward(dim3 grid, dim3 block, double* input, const int batch_size);
        double* cpu_forward(double *input, const int batch_size);
        double* cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size);
        ~MaxpoolingLayer();
};

class MeanpoolingLayer{
    private:
        int channels, size, kernel_size, stride, padding;
        int output_size;
    public:
        MeanpoolingLayer(int _channels, int _size, int _kernel_size, int _stride, int _padding);
        double* basic_forward(dim3 grid, dim3 block, double* input, const int batch_size);
        double* cpu_forward(double *input, const int batch_size);
        double* cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size);
        ~MeanpoolingLayer();
};

# endif