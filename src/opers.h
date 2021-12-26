# ifndef _OPERS_H_
# define _OPERS_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include <stdio.h>
# include <assert.h>

__global__ void forward_gather(double *input, double *output, const int size, const int channels, const int channel_idx);

double* cpu_gather(double *input, const int batch_size, const int size, const int channels, const int channel_idx);
double* gather(dim3 grid, dim3 block, double *input, const int batch_size, const int size, const int channels, const int channel_idx);


__global__ void forward_linear_transform(double *input, double *output, const int size, const double alpha, const double beta);

double* cpu_linear_transform(double *input, const int size, const double alpha, const double beta, bool inplace = true);
double* linear_transform(dim3 grid, dim3 block, double *input, const int size, const double alpha, const double beta, bool inplace = true);

__global__ void forward_channel_concat_2(double *input1, double *input2, double *output, const int batch_size, const int channel1, const int channel2, const int size_r, const int size_c);
__global__ void forward_channel_concat_3(double *input1, double *input2, double *input3, double *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int size_r, const int size_c);
__global__ void forward_channel_concat_4(double *input1, double *input2, double *input3, double *input4, double *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int channel4, const int size_r, const int size_c);

double* cpu_channel_concat(double *input[], const int num, const int batch_size, const int channel[], const int size_r, const int size_c);
double* channel_concat(dim3 grid, dim3 block, double **input, const int num, const int batch_size, const int channel[], const int size_r, const int size_c);

# endif