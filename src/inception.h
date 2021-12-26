# ifndef _INCEPTION_H_
# define _INCEPTION_H_
# include <cuda.h>
# include "cuda_runtime.h"
# include "layers.h"
# include "utils.h"
# include "conv.h"
# include "pooling.h"
# include "activation.h"
# include "opers.h"
# include "fc.h"
# include <stdio.h>

struct InceptionParams {
    InceptionLayer1params param_l1;
    InceptionLayer2params param_l2_1, param_l2_2, param_l2_3;
    InceptionLayer3params param_l3;
    InceptionLayer4params param_l4_1, param_l4_2, param_l4_3, param_l4_4;
    InceptionLayer5params param_l5;
    InceptionLayer6params param_l6_1, param_l6_2;
    InceptionOutputLayerparams param_output;
};

class Inception {
    private:
        int in_channels, size, out_channels, out_size;
        InceptionLayer1 layer1;
        InceptionLayer2 layer2_1, layer2_2, layer2_3;
        InceptionLayer3 layer3;
        InceptionLayer4 layer4_1, layer4_2, layer4_3, layer4_4;
        InceptionLayer5 layer5;
        InceptionLayer6 layer6_1, layer6_2;
        InceptionOutputLayer outputlayer;
        

    public:
        Inception(const int in_channels, const int size);
        int get_out_size() const;
        int get_out_channels() const;
        void set_params(struct InceptionParams params);
        double *cpu_forward(double *input, const int batch_size);
        double *gpu_forward(double *input, const int batch_size);
        ~Inception();
};

# endif