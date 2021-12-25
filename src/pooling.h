# ifndef _POOLING_H_
# define _POOLING_H_
# include "cuda_runtime.h"
# include "utils.h"
# include <stdio.h>
# endif

void maxpooling_cpu(double*, double*, int*, const int, const int, const int, const int, const int);
void meanpooling_cpu(double*, double*, const int, const int, const int, const int, const int, const int);

__global__ void maxpooling_forward(double*, double*, int*, const int, const int, const int);
__global__ void meanpool_forward(double*, double*, const int, const int, const int, const int);

class MaxpoolingLayer{
    private:
        int channels, size, kernel_size, stride;
        int output_size;
    public:
        MaxpoolingLayer(int _channels, int _size, int _kernel_size, int _stride);
        double* basic_forward(dim3 grid, dim3 block, double* input, const int batch_size);
        double* cpu_forward(double *input, const int batch_size);
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
        ~MeanpoolingLayer();
};