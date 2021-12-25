<<<<<<< HEAD
# ifndef _POOLING_H
# define _POOLING_H
=======
# ifndef _POOL_H_
# define _POOL_H_
>>>>>>> 76bad974ef9511852c330ca6537be4bae4f09775
# include "cuda_runtime.h"
# include "utils.h"
# include <stdio.h>
# endif

void maxpooling_cpu(float*, float*, int*, const int, const int, const int, const int, const int);
void meanpooling_cpu(float*, float*, const int, const int, const int, const int, const int);

__global__ void maxpooling_forward(float*, const int, const int, float*, float*);
__global__ void meanpool_forward(float*, const int, const int, float*);

class MaxpoolingLayer{
    private:
        int channels, size, kernel_size, stride;
        int output_size;
    public:
        MaxpoolingLayer(int _channels, int _size, int _kernel_size, int _stride);
        float* basic_forward(dim3 grid, dim3 block, float* input, const int batch_size);
        float* cpu_forward(float *input, const int batch_size);
        ~MaxpoolingLayer();
};

class MeanpoolingLayer{
    private:
        int channels, size, kernel_size, stride;
        int output_size;
    public:
        MeanpoolingLayer(int _channels, int _size, int _kernel_size, int _stride);
        float* basic_forward(dim3 grid, dim3 block, float* input, const int batch_size);
        float* cpu_forward(float *input, const int batch_size);
        ~MeanpoolingLayer();
};