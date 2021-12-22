# ifndef _FC_H_
# define _FC_H_
# include "cuda_runtime.h"
# include "utils.h"
# endif


// Fully Connected Layer
class FullyConnectedLayer {
    private:
        int in_features, out_features;
        int weight_N, bias_N, output_N;
        float *weight, *bias, *output;
    
    public:
        FullyConnectedLayer(int _in_features, int _out_features);
        void basic_forward(dim3 grid, dim3 block, float *input);
        void set_params(bool init, float *h_weight, float *h_bias);
        ~ FullyConnectedLayer();
};

__global__ void fc_basic_weight_forward(float*, float*, float*, const int, const int);
__global__ void fc_basic_bias_forward(float*, float*, float*, const int);
