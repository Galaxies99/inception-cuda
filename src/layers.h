# ifndef _LAYER_H_
# define _LAYER_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include "utils.h"
# include "conv.h"
# include "pooling.h"
# include "activation.h"
# include "opers.h"
# include "fc.h"
# include <stdio.h>
# include <iostream>
using namespace std;
# endif


struct InceptionLayer6params {
    double *way1_w, *way1_b;
    double *way23_1_w, *way23_1_b;
    double *way2_2_w, *way2_2_b;
    double *way3_2_w, *way3_2_b;
    double *way45_1_w, *way45_1_b;
    double *way45_2_w, *way45_2_b;
    double *way4_3_w, *way4_3_b;
    double *way5_3_w, *way5_3_b;
    double *way6_w, *way6_b;
};

class InceptionLayer6 {
    private:
        int in_channels, size;
        ConvolutionLayer way1, way23_1, way2_2, way3_2, way45_1, way45_2, way4_3, way5_3, way6;
        MeanpoolingLayer avgpool;
    public:
        InceptionLayer6(const int in_channels, const int size);
        void set_params(struct InceptionLayer6params params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionLayer6();
};