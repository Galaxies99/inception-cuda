# include "layers.h"

InceptionLayer1 :: InceptionLayer1(const int in_channels, const int size) : in_channels(in_channels), size(size), c_1(in_channels, 32, 299, 299, 3, 3, 2, 2, 0, 0), c_2(32, 32, 149, 149, 3, 3, 1, 1, 0, 0), c_3(32, 64, 147, 147, 3, 3, 1, 1, 1, 1), c_4(64, 80, 73, 73), c_5(80, 192, 73, 73, 3, 3, 1, 1, 0, 0), m1(64, 147, 3, 2), m2(192, 71, 3, 2) {
    way1_w = 0.46;
    way1_b = -0.02;
    way2_w = 0.45;
    way2_b = -0.09;
    way3_w = 0.45;
    way3_b = -0.19;
    out_size = 35;
    out_channels = 192;
}

int InceptionLayer1 :: get_out_size() const {
    return out_size;
}

int InceptionLayer1 :: get_out_channels() const {
    return out_channels;
}

void InceptionLayer1 :: set_params(struct InceptionLayer1params params) {    
    way1_w = params.way1_w;
    way1_b = params.way1_b;
    way2_w = params.way2_w;
    way2_b = params.way2_b;
    way3_w = params.way3_w;
    way3_b = params.way3_b;
    c_1.set_params(params.c_1_w, params.c_1_b);
    c_2.set_params(params.c_2_w, params.c_2_b);
    c_3.set_params(params.c_3_w, params.c_3_b);
    c_4.set_params(params.c_4_w, params.c_4_b);
    c_5.set_params(params.c_5_w, params.c_5_b);
}

double* InceptionLayer1 :: cpu_forward(double *input, const int batch_size) {
    double* slice0 = cpu_gather(input, batch_size, size, in_channels, 0);
    double* slice1 = cpu_gather(input, batch_size, size, in_channels, 1);
    double* slice2 = cpu_gather(input, batch_size, size, in_channels, 2);
    
    cpu_linear_transform(slice0, batch_size*size*size, way1_w, way1_b);
    cpu_linear_transform(slice1, batch_size*size*size, way2_w, way2_b);
    cpu_linear_transform(slice2, batch_size*size*size, way3_w, way3_b);

    double *concat_slice[] = {slice0, slice1, slice2};
    int channel_list[] = {1, 1, 1};
    double *input_new = cpu_channel_concat(concat_slice, 3, batch_size, channel_list, size, size);
    free(slice0);
    free(slice1);
    free(slice2);


    // center processing
    double* c_1_o = c_1.cpu_forward(input_new, batch_size);
    cpu_relu(c_1_o, batch_size * 32 * 149 * 149);
    free(input_new);
    

    double* c_2_o = c_2.cpu_forward(c_1_o, batch_size);
    cpu_relu(c_2_o, batch_size * 32 * 147 * 147);
    free(c_1_o);

    double* c_3_o = c_3.cpu_forward(c_2_o, batch_size);
    cpu_relu(c_3_o, batch_size * 64 * 147 * 147);
    free(c_2_o);

    double* maxpool_o = m1.cpu_forward(c_3_o, batch_size);
    free(c_3_o);

    double* c_4_o = c_4.cpu_forward(maxpool_o, batch_size);
    cpu_relu(c_4_o, batch_size * 80 * 73 * 73);
    free(maxpool_o);
    
    double* c_5_o = c_5.cpu_forward(c_4_o, batch_size);
    cpu_relu(c_5_o, batch_size * 192 * 71 * 71);
    free(c_4_o);

    
    double* final = m2.cpu_forward(c_5_o, batch_size);
    free(c_5_o);

    return final;
}

double* InceptionLayer1 :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);

    double* slice0 = gather(grid_conv, block_conv, input, batch_size, size, in_channels, 0);
    double* slice1 = gather(grid_conv, block_conv, input, batch_size, size, in_channels, 1);
    double* slice2 = gather(grid_conv, block_conv, input, batch_size, size, in_channels, 2);

    linear_transform(grid_act, block_act, slice0, batch_size*size*size, way1_w, way1_b);
    linear_transform(grid_act, block_act, slice1, batch_size*size*size, way2_w, way2_b);
    linear_transform(grid_act, block_act, slice2, batch_size*size*size, way3_w, way3_b);

    
    double *concat_slice[] = {slice0, slice1, slice2};
    int channel_list[] = {1, 1, 1};
    double *input_new = channel_concat(grid_conv, block_conv, concat_slice, 3, batch_size, channel_list, size, size);
    cudaFree(slice0);
    cudaFree(slice1);
    cudaFree(slice2);
    

    // center processing

    double* c_1_o = c_1.basic_forward(grid_conv, block_conv, input_new, batch_size);
    relu(grid_act, block_act, c_1_o, batch_size * 32 * 149 * 149);
    cudaFree(input_new);
    

    double* c_2_o = c_2.basic_forward(grid_conv, block_conv, c_1_o, batch_size);
    relu(grid_act, block_act, c_2_o, batch_size * 32 * 147 * 147);
    cudaFree(c_1_o);

    double* c_3_o = c_3.basic_forward(grid_conv, block_conv, c_2_o, batch_size);
    relu(grid_act, block_act, c_3_o, batch_size * 64 * 147 * 147);
    cudaFree(c_2_o);

    double *maxpool_o = m1.basic_forward(grid_conv, block_conv, c_3_o, batch_size);
    cudaFree(c_3_o);

    double* c_4_o = c_4.basic_forward(grid_conv, block_conv, maxpool_o, batch_size);
    relu(grid_act, block_act, c_4_o, batch_size * 80 * 73 * 73);
    cudaFree(maxpool_o);
    
    double* c_5_o = c_5.basic_forward(grid_conv, block_conv, c_4_o, batch_size);
    relu(grid_act, block_act, c_5_o, batch_size * 192 * 71 * 71);
    cudaFree(c_4_o);
    
    double *final = m2.basic_forward(grid_conv, block_conv, c_5_o, batch_size);
    cudaFree(c_5_o);

    return final;
}

InceptionLayer1 :: ~InceptionLayer1() {}

InceptionLayer2 :: InceptionLayer2(const int in_channels, const int size, const int way4_ch) : in_channels(in_channels), size(size), way4_ch(way4_ch), way1(in_channels, 64, size, size), way2_1(in_channels, 48, size, size), way2_2(48, 64, size, size, 5, 5, 1, 1, 2, 2), way3_1(in_channels, 64, size, size), way3_2(64, 96, size, size, 3, 3, 1, 1, 1, 1), way3_3(96, 96, size, size, 3, 3, 1, 1, 1, 1), way4(in_channels, way4_ch, size, size), avgpool(in_channels, size, 3, 1, 1) {
    out_size = size;
    out_channels = 64 + 64 + 96 + way4_ch;
}

int InceptionLayer2 :: get_out_size() const {
    return out_size;
}

int InceptionLayer2 :: get_out_channels() const {
    return out_channels;
}

void InceptionLayer2 :: set_params(struct InceptionLayer2params params) {
    way1.set_params(params.way1_w, params.way1_b);
    way2_1.set_params(params.way2_1_w, params.way2_1_b);
    way2_2.set_params(params.way2_2_w, params.way2_2_b);
    way3_1.set_params(params.way3_1_w, params.way3_1_b);
    way3_2.set_params(params.way3_2_w, params.way3_2_b);
    way3_3.set_params(params.way3_3_w, params.way3_3_b);
    way4.set_params(params.way4_w, params.way4_b);
}

double* InceptionLayer2 :: cpu_forward(double *input, const int batch_size) {
    //way1
    double *way1_o = way1.cpu_forward(input, batch_size);
    cpu_relu(way1_o, batch_size * 64 * size * size);
    //way2
    double *way2_1_o = way2_1.cpu_forward(input, batch_size);
    cpu_relu(way2_1_o, batch_size * 48 * size * size);
    double *way2_2_o = way2_2.cpu_forward(way2_1_o, batch_size);
    cpu_relu(way2_2_o, batch_size * 64 * size * size);
    free(way2_1_o);
    //way3
    double *way3_1_o = way3_1.cpu_forward(input, batch_size);
    cpu_relu(way3_1_o, batch_size * 64 * size * size);
    double *way3_2_o = way3_2.cpu_forward(way3_1_o, batch_size);
    cpu_relu(way3_2_o, batch_size * 96 * size * size);
    double *way3_3_o = way3_3.cpu_forward(way3_2_o, batch_size);
    cpu_relu(way3_3_o, batch_size * 96 * size * size);
    free(way3_1_o);
    free(way3_2_o);
    //way4
    double *way4_o1 = avgpool.cpu_forward(input, batch_size);
    double *way4_o = way4.cpu_forward(way4_o1, batch_size);
    cpu_relu(way4_o, batch_size * way4_ch * size * size);
    free(way4_o1);
    //final
    double *concat_in_final[] = {way1_o, way2_2_o, way3_3_o, way4_o};
    int concat_ch_final[] = {64, 64, 96, way4_ch};
    double *final = cpu_channel_concat(concat_in_final, 4, batch_size, concat_ch_final, size, size);
    free(way1_o);
    free(way2_2_o);
    free(way3_3_o);
    free(way4_o);
    return final;
}

double* InceptionLayer2 :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);
    //way1
    double *way1_o = way1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way1_o, batch_size * 64 * size * size);
    //way2
    double *way2_1_o = way2_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way2_1_o, batch_size * 48 * size * size);
    double *way2_2_o = way2_2.basic_forward(grid_conv, block_conv, way2_1_o, batch_size);
    relu(grid_act, block_act, way2_2_o, batch_size * 64 * size * size);
    cudaFree(way2_1_o);
    //way3
    double *way3_1_o = way3_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way3_1_o, batch_size * 64 * size * size);
    double *way3_2_o = way3_2.basic_forward(grid_conv, block_conv, way3_1_o, batch_size);
    relu(grid_act, block_act, way3_2_o, batch_size * 96 * size * size);
    double *way3_3_o = way3_3.basic_forward(grid_conv, block_conv, way3_2_o, batch_size);
    relu(grid_act, block_act, way3_3_o, batch_size * 96 * size * size);
    cudaFree(way3_1_o);
    cudaFree(way3_2_o);
    //way4
    double *way4_o1 = avgpool.basic_forward(grid_conv, block_conv, input, batch_size);
    double *way4_o = way4.basic_forward(grid_conv, block_conv, way4_o1, batch_size);
    relu(grid_act, block_act, way4_o, batch_size * way4_ch * size * size);
    cudaFree(way4_o1);
    //final
    double *concat_in_final[] = {way1_o, way2_2_o, way3_3_o, way4_o};
    int concat_ch_final[] = {64, 64, 96, way4_ch};
    double *final = channel_concat(grid_conv, block_conv, concat_in_final, 4, batch_size, concat_ch_final, size, size);
    cudaFree(way1_o);
    cudaFree(way2_2_o);
    cudaFree(way3_3_o);
    cudaFree(way4_o);
    return final;
}

InceptionLayer2 :: ~InceptionLayer2() {}

InceptionLayer3 :: InceptionLayer3(const int in_channels, const int size) : in_channels(in_channels), size(size), way1(in_channels, 384, size, size, 3, 3, 2, 2, 0, 0), way2_1(in_channels, 64, size, size), way2_2(64, 96, size, size, 3, 3, 1, 1, 1, 1), way2_3(96, 96, size, size, 3, 3, 2, 2, 0, 0), maxpool(in_channels, size, 3, 2) {
    out_size = (size - 1) / 2;
    out_channels = 384 + 96 + in_channels;
}

int InceptionLayer3 :: get_out_size() const {
    return out_size;
}

int InceptionLayer3 :: get_out_channels() const {
    return out_channels;
}

void InceptionLayer3 :: set_params(struct InceptionLayer3params params) {
    way1.set_params(params.way1_w, params.way1_b);
    way2_1.set_params(params.way2_1_w, params.way2_1_b);
    way2_2.set_params(params.way2_2_w, params.way2_2_b);
    way2_3.set_params(params.way2_3_w, params.way2_3_b);
}

double* InceptionLayer3 :: cpu_forward(double *input, const int batch_size) {
    //way1
    double *way1_o = way1.cpu_forward(input, batch_size);
    cpu_relu(way1_o, batch_size * 384 * out_size * out_size);
    //way2
    double *way2_1_o = way2_1.cpu_forward(input, batch_size);
    cpu_relu(way2_1_o, batch_size * 64 * size * size);
    double *way2_2_o = way2_2.cpu_forward(way2_1_o, batch_size);
    cpu_relu(way2_2_o, batch_size * 96 * size * size);
    double *way2_3_o = way2_3.cpu_forward(way2_2_o, batch_size);
    cpu_relu(way2_3_o, batch_size * 96 * out_size * out_size);
    free(way2_1_o);
    free(way2_2_o);
    //way3
    double *way3_o = maxpool.cpu_forward(input, batch_size);
    //final
    double *concat_in_final[] = {way1_o, way2_3_o, way3_o};
    int concat_ch_final[] = {384, 96, in_channels};
    double *final = cpu_channel_concat(concat_in_final, 3, batch_size, concat_ch_final, out_size, out_size);
    free(way1_o);
    free(way2_3_o);
    free(way3_o);
    return final;
}

double* InceptionLayer3 :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);
    //way1
    double *way1_o = way1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way1_o, batch_size * 384 * out_size * out_size);
    //way2
    double *way2_1_o = way2_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way2_1_o, batch_size * 64 * size * size);
    double *way2_2_o = way2_2.basic_forward(grid_conv, block_conv, way2_1_o, batch_size);
    relu(grid_act, block_act, way2_2_o, batch_size * 96 * size * size);
    double *way2_3_o = way2_3.basic_forward(grid_conv, block_conv, way2_2_o, batch_size);
    relu(grid_act, block_act, way2_3_o, batch_size * 96 * out_size * out_size);
    cudaFree(way2_1_o);
    cudaFree(way2_2_o);
    //way3
    double *way3_o = maxpool.basic_forward(grid_conv, block_conv, input, batch_size);
    //final
    double *concat_in_final[] = {way1_o, way2_3_o, way3_o};
    int concat_ch_final[] = {384, 96, in_channels};
    double *final = channel_concat(grid_conv, block_conv, concat_in_final, 3, batch_size, concat_ch_final, out_size, out_size);
    cudaFree(way1_o);
    cudaFree(way2_3_o);
    cudaFree(way3_o);
    return final;
}

InceptionLayer3 :: ~InceptionLayer3() {}

// Layer4
InceptionLayer4 :: InceptionLayer4(const int in_channels, const int size) : in_channels(in_channels), size(size), way1(in_channels, 192, size, size), way2_1(in_channels, 128, size, size), way2_2(128, 128, size, size, 1, 7, 1, 1, 0, 3), way2_3(128, 192, size, size, 7, 1, 1, 1, 3, 0), way3_1(in_channels, 128, size, size), way3_2(128, 128, size, size, 7, 1, 1, 1, 3, 0), way3_3(128, 128, size, size, 1, 7, 1, 1, 0, 3), way3_4(128, 128, size, size, 7, 1, 1, 1, 3, 0), way3_5(128, 192, size, size, 1, 7, 1, 1, 0, 3), way4(in_channels, 192, size, size), meanpool(in_channels, size, 3, 1, 1) {
    out_size = size;
    out_channels = in_channels;
}

int InceptionLayer4 :: get_out_size() const {
    return out_size;
}

int InceptionLayer4 :: get_out_channels() const {
    return out_channels;
}

void InceptionLayer4 :: set_params(struct InceptionLayer4params params) {
    way1.set_params(params.way1_w, params.way1_b);
    way2_1.set_params(params.way2_1_w, params.way2_1_b);
    way2_2.set_params(params.way2_2_w, params.way2_2_b);
    way2_3.set_params(params.way2_3_w, params.way2_3_b);
    way3_1.set_params(params.way3_1_w, params.way3_1_b);
    way3_2.set_params(params.way3_2_w, params.way3_2_b);
    way3_3.set_params(params.way3_3_w, params.way3_3_b);
    way3_4.set_params(params.way3_4_w, params.way3_4_b);
    way3_5.set_params(params.way3_5_w, params.way3_5_b);
    way4.set_params(params.way4_w, params.way4_b);
}

double* InceptionLayer4 :: cpu_forward(double *input, const int batch_size) {
    //way1
    double *way1_o = way1.cpu_forward(input, batch_size);
    cpu_relu(way1_o, batch_size * 192 * out_size * out_size);

    //way2
    double *way2_1_o = way2_1.cpu_forward(input, batch_size);
    cpu_relu(way2_1_o, batch_size * 128 * size * size);

    double *way2_2_o = way2_2.cpu_forward(way2_1_o, batch_size);
    cpu_relu(way2_2_o, batch_size * 128 * size * size);
    free(way2_1_o);

    double *way2_3_o = way2_3.cpu_forward(way2_2_o, batch_size);
    cpu_relu(way2_3_o, batch_size * 192 * out_size * out_size);
    free(way2_2_o);

    //way3
    double *way3_1_o = way3_1.cpu_forward(input, batch_size);
    cpu_relu(way3_1_o, batch_size * 128 * size * size);

    double *way3_2_o = way3_2.cpu_forward(way3_1_o, batch_size);
    cpu_relu(way3_2_o, batch_size * 128 * size * size);
    free(way3_1_o);

    double *way3_3_o = way3_3.cpu_forward(way3_2_o, batch_size);
    cpu_relu(way3_3_o, batch_size * 128 * size * size);
    free(way3_2_o);

    double *way3_4_o = way3_4.cpu_forward(way3_3_o, batch_size);
    cpu_relu(way3_4_o, batch_size * 128 * size * size);
    free(way3_3_o);

    double *way3_5_o = way3_5.cpu_forward(way3_4_o, batch_size);
    cpu_relu(way3_5_o, batch_size * 192 * out_size * out_size);
    free(way3_4_o);
    

    //way4
    double *avg_o = meanpool.cpu_forward(input, batch_size);

    double *way4_o = way4.cpu_forward(avg_o, batch_size);
    cpu_relu(way4_o, batch_size * 192 * out_size * out_size);
    free(avg_o);

    //final
    double *concat_in_final[] = {way1_o, way2_3_o, way3_5_o, way4_o};
    int concat_ch_final[] = {192, 192, 192, 192};
    double *final = cpu_channel_concat(concat_in_final, 4, batch_size, concat_ch_final, out_size, out_size);
    free(way1_o);
    free(way2_3_o);
    free(way3_5_o);
    free(way4_o);
    return final;
}

double* InceptionLayer4 :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);

    //way1
    double *way1_o = way1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way1_o, batch_size * 192 * out_size * out_size);

    //way2
    double *way2_1_o = way2_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way2_1_o, batch_size * 128 * size * size);

    double *way2_2_o = way2_2.basic_forward(grid_conv, block_conv, way2_1_o, batch_size);
    relu(grid_act, block_act, way2_2_o, batch_size * 128 * size * size);
    cudaFree(way2_1_o);

    double *way2_3_o = way2_3.basic_forward(grid_conv, block_conv, way2_2_o, batch_size);
    relu(grid_act, block_act, way2_3_o, batch_size * 192 * out_size * out_size);
    cudaFree(way2_2_o);
    
    //way3
    double *way3_1_o = way3_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way3_1_o, batch_size * 128 * size * size);

    double *way3_2_o = way3_2.basic_forward(grid_conv, block_conv, way3_1_o, batch_size);
    relu(grid_act, block_act, way3_2_o, batch_size * 128 * size * size);
    cudaFree(way3_1_o);

    double *way3_3_o = way3_3.basic_forward(grid_conv, block_conv, way3_2_o, batch_size);
    relu(grid_act, block_act, way3_3_o, batch_size * 128 * size * size);
    cudaFree(way3_2_o);

    double *way3_4_o = way3_4.basic_forward(grid_conv, block_conv, way3_3_o, batch_size);
    relu(grid_act, block_act, way3_4_o, batch_size * 128 * size * size);
    cudaFree(way3_3_o);

    double *way3_5_o = way3_5.basic_forward(grid_conv, block_conv, way3_4_o, batch_size);
    relu(grid_act, block_act, way3_5_o, batch_size * 192 * out_size * out_size);
    cudaFree(way3_4_o);

    //way4
    double *avg_o = meanpool.basic_forward(grid_conv, block_conv, input, batch_size);
    double *way4_o = way4.basic_forward(grid_conv, block_conv, avg_o, batch_size);
    relu(grid_act, block_act, way4_o, batch_size * 192 * out_size * out_size);
    cudaFree(avg_o);

    //final
    double *concat_in_final[] = {way1_o, way2_3_o, way3_5_o, way4_o};
    int concat_ch_final[] = {192, 192, 192, 192};
    double *final = channel_concat(grid_conv, block_conv, concat_in_final, 4, batch_size, concat_ch_final, out_size, out_size);


    cudaFree(way1_o);
    cudaFree(way2_3_o);
    cudaFree(way3_5_o);
    cudaFree(way4_o);
    return final;
}

InceptionLayer4 :: ~InceptionLayer4() {}


InceptionLayer5 :: InceptionLayer5(const int in_channels, const int size) : in_channels(in_channels), size(size), way1_1(in_channels, 192, size, size), way1_2(192, 320, size, size, 3, 3, 2, 2, 0, 0), way2_1(in_channels, 192, size, size), way2_2(192, 192, size, size, 1, 7, 1, 1, 0, 3), way2_3(192, 192, size, size, 7, 1, 1, 1, 3, 0), way2_4(192, 192, size, size, 3, 3, 2, 2, 0, 0), maxpool(in_channels, size, 3, 2) {
    out_size = (size - 3) / 2 + 1;
    out_channels = 320 + 192 + 768;
}

int InceptionLayer5 :: get_out_size() const {
    return out_size;
}

int InceptionLayer5 :: get_out_channels() const {
    return out_channels;
}

void InceptionLayer5 :: set_params(struct InceptionLayer5params params) {
    way1_1.set_params(params.way1_1_w, params.way1_1_b);
    way1_2.set_params(params.way1_2_w, params.way1_2_b);
    way2_1.set_params(params.way2_1_w, params.way2_1_b);
    way2_2.set_params(params.way2_2_w, params.way2_2_b);
    way2_3.set_params(params.way2_3_w, params.way2_3_b);
    way2_4.set_params(params.way2_4_w, params.way2_4_b);
}

double* InceptionLayer5 :: cpu_forward(double *input, const int batch_size) {
    // way1
    double *way1_o1 = way1_1.cpu_forward(input, batch_size);
    cpu_relu(way1_o1, batch_size * 192 * size * size);
    double *way1_o = way1_2.cpu_forward(way1_o1, batch_size);
    cpu_relu(way1_o, batch_size * 320 * out_size * out_size);
    free(way1_o1);
    // way2
    double *way2_o1 = way2_1.cpu_forward(input, batch_size);
    cpu_relu(way2_o1, batch_size * 192 * size * size);
    double *way2_o2 = way2_2.cpu_forward(way2_o1, batch_size);
    cpu_relu(way2_o2, batch_size * 192 * size * size);
    double *way2_o3 = way2_3.cpu_forward(way2_o2, batch_size);
    cpu_relu(way2_o3, batch_size * 192 * size * size);
    double *way2_o = way2_4.cpu_forward(way2_o3, batch_size);
    cpu_relu(way2_o, batch_size * 192 * out_size * out_size);
    free(way2_o1);
    free(way2_o2);
    free(way2_o3);
    // way3
    double *way3_o = maxpool.cpu_forward(input, batch_size);
    // final
    double *concat_in_final[] = {way1_o, way2_o, way3_o};
    int concat_ch_final[] = {320, 192, 768};
    double *final = cpu_channel_concat(concat_in_final, 3, batch_size, concat_ch_final, out_size, out_size);
    free(way1_o);
    free(way2_o);
    free(way3_o);
    return final;
}

double* InceptionLayer5 :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);
    // way1
    double *way1_o1 = way1_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way1_o1, batch_size * 192 * size * size);
    double *way1_o = way1_2.basic_forward(grid_conv, block_conv, way1_o1, batch_size);
    relu(grid_act, block_act, way1_o, batch_size * 320 * out_size * out_size);
    cudaFree(way1_o1);
    // way2
    double *way2_o1 = way2_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way2_o1, batch_size * 192 * size * size);
    double *way2_o2 = way2_2.basic_forward(grid_conv, block_conv, way2_o1, batch_size);
    relu(grid_act, block_act, way2_o2, batch_size * 192 * size * size);
    double *way2_o3 = way2_3.basic_forward(grid_conv, block_conv, way2_o2, batch_size);
    relu(grid_act, block_act, way2_o3, batch_size * 192 * size * size);
    double *way2_o = way2_4.basic_forward(grid_conv, block_conv, way2_o3, batch_size);
    relu(grid_act, block_act, way2_o, batch_size * 192 * out_size * out_size);
    cudaFree(way2_o1);
    cudaFree(way2_o2);
    cudaFree(way2_o3);
    // way3
    double *way3_o = maxpool.basic_forward(grid_conv, block_conv, input, batch_size);
    // final
    double *concat_in_final[] = {way1_o, way2_o, way3_o};
    int concat_ch_final[] = {320, 192, 768};
    double *final = channel_concat(grid_conv, grid_conv, concat_in_final, 3, batch_size, concat_ch_final, out_size, out_size);
    cudaFree(way1_o);
    cudaFree(way2_o);
    cudaFree(way3_o);
    return final;
}

InceptionLayer5 :: ~InceptionLayer5() {}

InceptionLayer6 :: InceptionLayer6(const int in_channels, const int size) : in_channels(in_channels), size(size), way1(in_channels, 320, size, size), way23_1(in_channels, 384, size, size), way2_2(384, 384, size, size, 1, 3, 1, 1, 0, 1), way3_2(384, 384, size, size, 3, 1, 1, 1, 1, 0), way45_1(in_channels, 448, size, size), way45_2(448, 384, size, size, 3, 3, 1, 1, 1, 1), way4_3(384, 384, size, size, 1, 3, 1, 1, 0, 1), way5_3(384, 384, size, size, 3, 1, 1, 1, 1, 0), way6(in_channels, 192, size, size), avgpool(in_channels, size, 3, 1, 1) {
    out_size = size;
    out_channels = 320 + 768 + 768 + 192;
}

int InceptionLayer6 :: get_out_size() const {
    return out_size;
}

int InceptionLayer6 :: get_out_channels() const {
    return out_channels;
}

void InceptionLayer6 :: set_params(struct InceptionLayer6params params) {
    way1.set_params(params.way1_w, params.way1_b);
    way23_1.set_params(params.way23_1_w, params.way23_1_b);
    way2_2.set_params(params.way2_2_w, params.way2_2_b);
    way3_2.set_params(params.way3_2_w, params.way3_2_b);
    way45_1.set_params(params.way45_1_w, params.way45_1_b);
    way45_2.set_params(params.way45_2_w, params.way45_2_b);
    way4_3.set_params(params.way4_3_w, params.way4_3_b);
    way5_3.set_params(params.way5_3_w, params.way5_3_b);
    way6.set_params(params.way6_w, params.way6_b);
}

double* InceptionLayer6 :: cpu_forward(double *input, const int batch_size) {
    // way1
    double *way1_o = way1.cpu_forward(input, batch_size);
    cpu_relu(way1_o, batch_size * 320 * size * size);
    // way2 & way3
    double *way23_o1 = way23_1.cpu_forward(input, batch_size);
    cpu_relu(way23_o1, batch_size * 384 * size * size);
    double *way2_o2 = way2_2.cpu_forward(way23_o1, batch_size);
    cpu_relu(way2_o2, batch_size * 384 * size * size);
    double *way3_o2 = way3_2.cpu_forward(way23_o1, batch_size);
    cpu_relu(way3_o2, batch_size * 384 * size * size);
    double *concat_in1[] = {way2_o2, way3_o2};
    int concat_ch1[] = {384, 384};
    double *way23_o = cpu_channel_concat(concat_in1, 2, batch_size, concat_ch1, size, size);
    free(way23_o1);
    free(way2_o2);
    free(way3_o2);
    // way4 & way5
    double *way45_o1 = way45_1.cpu_forward(input, batch_size);
    cpu_relu(way45_o1, batch_size * 448 * size * size);
    double *way45_o2 = way45_2.cpu_forward(way45_o1, batch_size);
    cpu_relu(way45_o2, batch_size * 384 * size * size);
    double *way4_o3 = way4_3.cpu_forward(way45_o2, batch_size);
    cpu_relu(way4_o3, batch_size * 384 * size * size);
    double *way5_o3 = way5_3.cpu_forward(way45_o2, batch_size);
    cpu_relu(way5_o3, batch_size * 384 * size * size);
    double *concat_in2[] = {way4_o3, way5_o3};
    int concat_ch2[] = {384, 384};
    double *way45_o = cpu_channel_concat(concat_in2, 2, batch_size, concat_ch2, size, size);
    free(way45_o1);
    free(way45_o2);
    free(way4_o3);
    free(way5_o3);
    // way6
    double *way6_o1 = avgpool.cpu_forward(input, batch_size);
    double *way6_o = way6.cpu_forward(way6_o1, batch_size);
    cpu_relu(way6_o, batch_size * 192 * size * size);
    free(way6_o1);
    // final
    double *concat_in_final[] = {way1_o, way23_o, way45_o, way6_o};
    int concat_ch_final[] = {320, 768, 768, 192};
    double *final = cpu_channel_concat(concat_in_final, 4, batch_size, concat_ch_final, size, size);
    free(way1_o);
    free(way23_o);
    free(way45_o);
    free(way6_o);
    return final;
}

double* InceptionLayer6 :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);
    // way1
    double *way1_o = way1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way1_o, batch_size * 320 * size * size);
    // way2 & way3
    double *way23_o1 = way23_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way23_o1, batch_size * 384 * size * size);
    double *way2_o2 = way2_2.basic_forward(grid_conv, block_conv, way23_o1, batch_size);
    relu(grid_act, block_act, way2_o2, batch_size * 384 * size * size);
    double *way3_o2 = way3_2.basic_forward(grid_conv, block_conv, way23_o1, batch_size);
    relu(grid_act, block_act, way3_o2, batch_size * 384 * size * size);
    double *concat_in1[] = {way2_o2, way3_o2};
    int concat_ch1[] = {384, 384};
    double *way23_o = channel_concat(grid_conv, block_conv, concat_in1, 2, batch_size, concat_ch1, size, size);
    cudaFree(way23_o1);
    cudaFree(way2_o2);
    cudaFree(way3_o2);
    // way4 & way5
    double *way45_o1 = way45_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way45_o1, batch_size * 448 * size * size);
    double *way45_o2 = way45_2.basic_forward(grid_conv, block_conv, way45_o1, batch_size);
    relu(grid_act, block_act, way45_o2, batch_size * 384 * size * size);
    double *way4_o3 = way4_3.basic_forward(grid_conv, block_conv, way45_o2, batch_size);
    relu(grid_act, block_act, way4_o3, batch_size * 384 * size * size);
    double *way5_o3 = way5_3.basic_forward(grid_conv, block_conv, way45_o2, batch_size);
    relu(grid_act, block_act, way5_o3, batch_size * 384 * size * size);
    double *concat_in2[] = {way4_o3, way5_o3};
    int concat_ch2[] = {384, 384};
    double *way45_o = channel_concat(grid_conv, block_conv, concat_in2, 2, batch_size, concat_ch2, size, size);
    cudaFree(way45_o1);
    cudaFree(way45_o2);
    cudaFree(way4_o3);
    cudaFree(way5_o3);
    // way6
    double *way6_o1 = avgpool.basic_forward(grid_conv, block_conv, input, batch_size);
    double *way6_o = way6.basic_forward(grid_conv, block_conv, way6_o1, batch_size);
    relu(grid_act, block_act, way6_o, batch_size * 192 * size * size);
    cudaFree(way6_o1);
    // final
    double *concat_in_final[] = {way1_o, way23_o, way45_o, way6_o};
    int concat_ch_final[] = {320, 768, 768, 192};
    double *final = channel_concat(grid_conv, block_conv, concat_in_final, 4, batch_size, concat_ch_final, size, size);
    cudaFree(way1_o);
    cudaFree(way23_o);
    cudaFree(way45_o);
    cudaFree(way6_o);
    return final;
}

InceptionLayer6 :: ~InceptionLayer6() {}



InceptionOutputLayer :: InceptionOutputLayer(const int in_channels, const int size) : in_channels(in_channels), size(size), fc(2048, 1000), avgpool(in_channels, size, size, 1, 0) {
    out_size = 1;
    out_channels = 1000;
}

int InceptionOutputLayer :: get_out_size() const {
    return out_size;
}

int InceptionOutputLayer :: get_out_channels() const {
    return out_channels;
}

void InceptionOutputLayer :: set_params(struct InceptionOutputLayerparams params) {
    fc.set_params(params.fc_w, params.fc_b);
}

double* InceptionOutputLayer :: cpu_forward(double *input, const int batch_size) {
    double *avg_o = avgpool.cpu_forward(input, batch_size);
    double *final = fc.cpu_forward(avg_o, batch_size);
    free(avg_o);
    return final;
}

double* InceptionOutputLayer :: gpu_forward(double *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    double *avg_o = avgpool.basic_forward(grid_conv, block_conv, input, batch_size);
    double *final = fc.basic_forward(grid_conv, block_conv, avg_o, batch_size);
    cudaFree(avg_o);
    return final;
}

InceptionOutputLayer :: ~InceptionOutputLayer() {}
