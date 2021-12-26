# ifndef _ACTIVATION_H_
# define _ACTIVATION_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include <stdio.h>
double activation_relu_cpu(double x);
__device__ double activation_relu(double x);
__global__ void forward_relu(double *input, double *output, const int size);

double* cpu_relu(double *input, const int size, bool inplace = true);
double* relu(dim3 grid, dim3 block, double *input, const int size, bool inplace = true);

# endif