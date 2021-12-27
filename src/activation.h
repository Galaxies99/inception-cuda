# ifndef _ACTIVATION_H_
# define _ACTIVATION_H_
# include <cuda.h>
# include <cudnn.h>
# include <stdio.h>
# include <cudnn_v8.h>

# include "utils.h"
# include "cudnn_utils.h"
# include "cuda_runtime.h"

double activation_relu_cpu(double x);
__device__ double activation_relu(double x);
__global__ void forward_relu(double *input, double *output, const int size);

double* cpu_relu(double *input, const int size, bool inplace = true);
double* relu(dim3 grid, dim3 block, double *input, const int size, bool inplace = true);
double* cudnn_relu(cudnnHandle_t& handle, double *input, const int batch_size, const int channels, const int size, bool inplace = true);

# endif