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


struct InceptionLayer1params {
    double way1_w, way1_b;
    double way2_w, way2_b;
    double way3_w, way3_b;
    double *c_1_w, *c_1_b;
    double *c_2_w, *c_2_b;
    double *c_3_w, *c_3_b;
    double *c_4_w, *c_4_b;
    double *c_5_w, *c_5_b;
};

class InceptionLayer1 {
    private:
        double way1_w, way1_b;
        double way2_w, way2_b;
        double way3_w, way3_b;
        int in_channels, size, out_channels, out_size;
        ConvolutionLayer c_1, c_2, c_3, c_4, c_5;
        MaxpoolingLayer m1, m2;
    public:
        InceptionLayer1(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer1params params);
        double *cpu_forward(double *input, const int batch_size);
        double *gpu_forward(double *input, const int batch_size);
        ~InceptionLayer1();
};


struct InceptionLayer2params {
    double *way1_w, *way1_b;
    double *way2_1_w, *way2_1_b;
    double *way2_2_w, *way2_2_b;
    double *way3_1_w, *way3_1_b;
    double *way3_2_w, *way3_2_b;
    double *way3_3_w, *way3_3_b;
    double *way4_w, *way4_b;
};

class InceptionLayer2 {
    private:
        int in_channels, size, out_channels, out_size, way4_ch;
        ConvolutionLayer way1, way2_1, way2_2, way3_1, way3_2, way3_3, way4;
        MeanpoolingLayer avgpool;
    public:
        InceptionLayer2(const int in_channels, const int size, const int way4_ch);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer2params params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionLayer2();
};

struct InceptionLayer3params {
    double *way1_w, *way1_b;
    double *way2_1_w, *way2_1_b;
    double *way2_2_w, *way2_2_b;
    double *way2_3_w, *way2_3_b;
};

class InceptionLayer3 {
    private:
        int in_channels, size, out_channels, out_size;
        ConvolutionLayer way1, way2_1, way2_2, way2_3;
        MaxpoolingLayer maxpool;
    public:
        InceptionLayer3(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer3params params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionLayer3();
};

struct InceptionLayer4params {
    double *way1_w, *way1_b;
    double *way2_1_w, *way2_1_b;
    double *way2_2_w, *way2_2_b;
    double *way2_3_w, *way2_3_b;
    double *way3_1_w, *way3_1_b;
    double *way3_2_w, *way3_2_b;
    double *way3_3_w, *way3_3_b;
    double *way3_4_w, *way3_4_b;
    double *way3_5_w, *way3_5_b;
    double *way4_w, *way4_b;
};

class InceptionLayer4 {
    private:
        int in_channels, size, out_channels, out_size;
        ConvolutionLayer way1, way2_1, way2_2, way2_3, way3_1, way3_2, way3_3, way3_4, way3_5, way4;
        MeanpoolingLayer meanpool;
    public:
        InceptionLayer4(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionLayer4params params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionLayer4();
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

struct InceptionOutputLayerparams {
    double *fc_w, *fc_b;
};

class InceptionOutputLayer {
    private:
        int in_channels, size, out_channels, out_size;
        FullyConnectedLayer fc;
        MeanpoolingLayer avgpool;
    public:
        InceptionOutputLayer(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionOutputLayerparams params);
        double* cpu_forward(double *input, const int batch_size);
        double* gpu_forward(double *input, const int batch_size);
        ~InceptionOutputLayer();
};

# endif