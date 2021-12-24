# ifndef _ACTIVATION_H_
# define _ACTIVATION_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include <stdio.h>
# endif

float activation_relu_cpu(float x);
__device__ float activation_relu(float x);
__global__ void forward_relu(float *input, float *output, const int size);

float* cpu_relu(float *input, const int size, bool inplace = true);
float* relu(dim3 grid, dim3 block, float *input, const int size, bool inplace = true);