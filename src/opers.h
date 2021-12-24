# ifndef _OPERS_H_
# define _OPERS_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include <stdio.h>
# endif


__global__ void forward_linear_transform(float *input, float *output, const int size, const float alpha, const float beta);

float* cpu_linear_transform(float *input, const int size, const float alpha, const float beta, bool inplace = true);
float* linear_transform(dim3 grid, dim3 block, float *input, const int size, const float alpha, const float beta, bool inplace = true);