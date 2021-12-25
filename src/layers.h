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
# endif


struct InceptionLayer6params {
    float *way1_w, *way1_b;
    float *way23_1_w, *way23_1_b;
    float *way2_2_w, *way2_2_b;
    float *way3_2_w, *way3_2_b;
    float *way45_1_w, *way45_1_b;
    float *way45_2_w, *way45_2_b;
    float *way4_3_w, *way4_3_b;
    float *way5_3_w, *way5_3_b;
    float *way6_w, *way6_b;
};

class InceptionLayer6 {
    private:
        int in_channels, size;
        ConvolutionLayer way1, way23_1, way2_2, way3_2, way_45_1, way45_2, way4_3, way5_3, way6;
        MeanpoolingLayer avgpool;
    public:
        InceptionLayer6(const int in_channels, const int size);
        void set_params(struct InceptionLayer6params params);
        void cpu_forward(float *input, const int batch_size);
        ~InceptionLayer6();
}