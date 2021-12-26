# include "inception.h"

Inception :: Inception(const int in_channels, const int size) : in_channels(in_channels), size(size), layer1(in_channels, size), layer2_1(layer1.get_out_channels(), layer1.get_out_size(), 32), layer2_2(layer2_1.get_out_channels(), layer2_1.get_out_size(), 64), layer2_3(layer2_2.get_out_channels(), layer2_2.get_out_size(), 64), layer3(layer2_3.get_out_channels(), layer2_3.get_out_size()), layer4_1(layer3.get_out_channels(), layer3.get_out_size()), layer4_2(layer4_1.get_out_channels(), layer4_1.get_out_size()), layer4_3(layer4_2.get_out_channels(), layer4_2.get_out_size()), layer4_4(layer4_3.get_out_channels(), layer4_3.get_out_size()), layer5(layer4_4.get_out_channels(), layer4_4.get_out_size()), layer6_1(layer5.get_out_channels(), layer5.get_out_size()), layer6_2(layer6_1.get_out_channels(), layer6_1.get_out_size()), outputlayer(layer6_2.get_out_channels(), layer6_2.get_out_size()){
    // out_size = 1;
    // out_channels = 1000;    
    out_size = 35;
    out_channels = 192;
}

int Inception :: get_out_size() const {
    return out_size;
}

int Inception :: get_out_channels() const {
    return out_channels;
}

void Inception :: set_params(struct InceptionParams params) {
    layer1.set_params(params.param_l1);
    layer2_1.set_params(params.param_l2_1);
    layer2_2.set_params(params.param_l2_2);
    layer2_3.set_params(params.param_l2_3);
    layer3.set_params(params.param_l3);
    layer4_1.set_params(params.param_l4_1);
    layer4_2.set_params(params.param_l4_2);
    layer4_3.set_params(params.param_l4_3);
    layer4_4.set_params(params.param_l4_4);
    layer5.set_params(params.param_l5);
    layer6_1.set_params(params.param_l6_1);
    layer6_2.set_params(params.param_l6_2);
    outputlayer.set_params(params.param_output);
}

double* Inception :: cpu_forward(double *input, const int batch_size) {
    double *layer1_o = layer1.cpu_forward(input, batch_size);
    return layer1_o;
    // double *layer2_1_o = layer2_1.cpu_forward(layer1_o, batch_size);
    // free(layer1_o);
    // double *layer2_2_o = layer2_2.cpu_forward(layer2_1_o, batch_size);
    // free(layer2_1_o);
    // double *layer2_3_o = layer2_3.cpu_forward(layer2_2_o, batch_size);
    // free(layer2_2_o);
    // double *layer3_o = layer3.cpu_forward(layer2_3_o, batch_size);
    // free(layer2_3_o);

    // return layer3_o;

    // double *layer4_1_o = layer4_1.cpu_forward(layer3_o, batch_size);
    // free(layer3_o);
    // double *layer4_2_o = layer4_2.cpu_forward(layer4_1_o, batch_size);
    // free(layer4_1_o);
    // double *layer4_3_o = layer4_3.cpu_forward(layer4_2_o, batch_size);
    // free(layer4_2_o);
    // double *layer4_4_o = layer4_4.cpu_forward(layer4_3_o, batch_size);
    // free(layer4_3_o);
    
    // double *layer5_o = layer5.cpu_forward(layer4_4_o, batch_size);
    // free(layer4_4_o);

    // double *layer6_1_o = layer6_1.cpu_forward(layer5_o, batch_size);
    // free(layer5_o);
    // double *layer6_2_o = layer6_2.cpu_forward(layer6_1_o, batch_size);
    // free(layer6_1_o);


    // double *final = outputlayer.cpu_forward(layer6_2_o, batch_size);
    // free(layer6_2_o);
    // return final;
}

double* Inception :: gpu_forward(double *input, const int batch_size) {
    
    double *layer1_o = layer1.gpu_forward(input, batch_size);
    return layer1_o;
    // double *layer2_1_o = layer2_1.gpu_forward(layer1_o, batch_size);
    // cudaFree(layer1_o);
    // double *layer2_2_o = layer2_2.gpu_forward(layer2_1_o, batch_size);
    // cudaFree(layer2_1_o);
    // double *layer2_3_o = layer2_3.gpu_forward(layer2_2_o, batch_size);
    // cudaFree(layer2_2_o);
    // double *layer3_o = layer3.gpu_forward(layer2_3_o, batch_size);
    // cudaFree(layer2_3_o);

    // return layer3_o;

    // double *layer4_1_o = layer4_1.gpu_forward(layer3_o, batch_size);
    // cudaFree(layer3_o);
    // double *layer4_2_o = layer4_2.gpu_forward(layer4_1_o, batch_size);
    // cudaFree(layer4_1_o);
    // double *layer4_3_o = layer4_3.gpu_forward(layer4_2_o, batch_size);
    // cudaFree(layer4_2_o);
    // double *layer4_4_o = layer4_4.gpu_forward(layer4_3_o, batch_size);
    // cudaFree(layer4_3_o);
    
    // double *layer5_o = layer5.gpu_forward(layer4_4_o, batch_size);
    // cudaFree(layer4_4_o);

    // double *layer6_1_o = layer6_1.gpu_forward(layer5_o, batch_size);
    // cudaFree(layer5_o);
    // double *layer6_2_o = layer6_2.gpu_forward(layer6_1_o, batch_size);
    // cudaFree(layer6_1_o);

    // double *final = outputlayer.gpu_forward(layer6_2_o, batch_size);
    // cudaFree(layer6_2_o);
    // return final;
}

Inception :: ~Inception() {}
