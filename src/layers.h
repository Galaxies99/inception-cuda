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


struct InceptionLayer2_2params {
    double *way1_w, *way1_b;
    double *way2_1_w, *way2_1_b;
    double *way2_2_w, *way2_2_b;
    double *way3_1_w, *way3_1_b;
    double *way3_2_w, *way3_2_b;
    double *way3_3_w, *way3_3_b;
    double *way4_w, *way4_b;
};

class InceptionLayer2_2 {
    private:
        int in_channels, size, out_channels, out_size;
        ConvolutionLayer way1, way2_1, way2_2, way3_1, way3_2, way3_3, way4;
        MeanpoolingLayer avgpool;
    public:
        InceptionLayer2_2(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer2_2params params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionLayer2_2();
};

struct InceptionLayer5params {
    double *way1_1_w, *way1_1_b;
    double *way1_2_w, *way1_2_b;
    double *way2_1_w, *way2_1_b;
    double *way2_2_w, *way2_2_b;
    double *way2_3_w, *way2_3_b;
    double *way2_4_w, *way2_4_b;
};

class InceptionLayer5 {
    private:
        int in_channels, size, out_channels, out_size;
        ConvolutionLayer way1_1, way1_2, way2_1, way2_2, way2_3, way2_4;
        MaxpoolingLayer maxpool;
    public:
        InceptionLayer5(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer5params params);
        double *cpu_forward(double *input, const int batch_size);
        double *gpu_forward(double *input, const int batch_size);
        ~InceptionLayer5();
};

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
        int in_channels, size, out_channels, out_size;
        ConvolutionLayer way1, way23_1, way2_2, way3_2, way45_1, way45_2, way4_3, way5_3, way6;
        MeanpoolingLayer avgpool;
    public:
        InceptionLayer6(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer6params params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionLayer6();
};
