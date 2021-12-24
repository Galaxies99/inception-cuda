# ifndef _FC_H_
# define _FC_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include "utils.h"
# include <stdio.h>
# endif


// Fully Connected Layer
class FullyConnectedLayer {
    private:
        int in_features, out_features;
        int weight_N, bias_N, output_N;
        float *weight, *bias;
        float *h_weight, *h_bias;
    
    public:
        FullyConnectedLayer(int _in_features, int _out_features);
        float* basic_forward(dim3 grid, dim3 block, float *input, const int batch_size);
        float* cpu_forward(float *input, const int batch_size);
        void set_params(float *_h_weight = NULL, float *_h_bias = NULL);
        ~ FullyConnectedLayer();
};

void fc_forward_cpu(float *input, float *output, float *weight, float *bias, const int batch_size, const int in_features, const int out_features);
__global__ void fc_basic_weight_forward(float *input, float *output, float *weight, const int batch_size, const int in_features, const int out_features);
__global__ void fc_basic_bias_forward(float *input, float *output, float *bias, const int batch_size, const int in_features, const int out_features);
