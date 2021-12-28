# ifndef _FC_H_
# define _FC_H_
# include <cuda.h>
# include <cudnn.h>
# include <stdio.h>
# include <cudnn_v8.h>

# include "utils.h"
# include "cudnn_utils.h"
# include "cuda_runtime.h"

// Fully Connected Layer
class FullyConnectedLayer {
    private:
        int in_features, out_features;
        int weight_N, bias_N, output_N;
        double *weight, *bias;
        double *h_weight, *h_bias;
    
    public:
        FullyConnectedLayer(int _in_features, int _out_features);
        double* basic_forward(dim3 grid, dim3 block, double *input, const int batch_size);
        double* opt_forward(dim3 grid, dim3 block, double *input, const int batch_size);
        double* cpu_forward(double *input, const int batch_size);
        double* cudnn_forward(cudnnHandle_t &handle, double *input, const int batch_size);
        void set_params(double *_h_weight = NULL, double *_h_bias = NULL);
        ~ FullyConnectedLayer();
};

void fc_forward_cpu(double *input, double *output, double *weight, double *bias, const int batch_size, const int in_features, const int out_features);

__global__ void fc_basic_weight_forward(double *input, double *output, double *weight, const int batch_size, const int in_features, const int out_features);

__global__ void fc_basic_bias_forward(double *input, double *output, double *bias, const int batch_size, const int in_features, const int out_features);

__global__ void fc_opt_weight_forward(double *input, double *output, double *weight, const int batch_size, const int in_features, const int out_features);

void fc_cudnn_forward(cudnnHandle_t& handle, double *input, double *output, double *weight, double *bias, const int batch_size, const int in_channels, const int out_channels);

# endif
