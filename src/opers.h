# ifndef _OPERS_H_
# define _OPERS_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include <stdio.h>
# include <assert.h>
# endif


__global__ void forward_linear_transform(float *input, float *output, const int size, const float alpha, const float beta);

float* cpu_linear_transform(float *input, const int size, const float alpha, const float beta, bool inplace = true);
float* linear_transform(dim3 grid, dim3 block, float *input, const int size, const float alpha, const float beta, bool inplace = true);

__global__ void forward_channel_concat_2(float *input1, float *input2, float *output, const int batch_size, const int channel1, const int channel2, const int size_r, const int size_c);
__global__ void forward_channel_concat_3(float *input1, float *input2, float *input3, float *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int size_r, const int size_c);
__global__ void forward_channel_concat_4(float *input1, float *input2, float *input3, float *input4, float *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int channel4, const int size_r, const int size_c);

float* cpu_channel_concat(float *input[], const int num, const int batch_size, const int channel[], const int size_r, const int size_c);
float* channel_concat(dim3 grid, dim3 block, float **input, const int num, const int batch_size, const int channel[], const int size_r, const int size_c);