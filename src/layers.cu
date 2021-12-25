# include "layers.h"



InceptionLayer6 :: InceptionLayer6(const int in_channels, const int size) : in_channels(in_channels), size(size), way1(in_channels, 320, size, size), way23_1(in_channels, 384, size, size), way2_2(384, 384, size, size, 1, 3, 1, 1, 0, 1), way3_2(384, 384, size, size, 3, 1, 1, 1, 1, 0), way45_1(in_channels, 448, size, size), way45_2(448, 384, size, size, 3, 3, 1, 1, 1, 1), way4_3(384, 384, size, size, 1, 3, 1, 1, 0, 1), way5_3(384, 384, size, size, 3, 1, 1, 1, 1, 0), way6(in_channels, 192, size, size), avgpool(in_channels, size, 3, 1, 1) {}

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

float* InceptionLayer6 :: cpu_forward(float *input, const int batch_size) {
    // way1
    float *way1_o = way1.cpu_forward(input, batch_size);
    cpu_relu(way1_o, batch_size * 320 * size * size);
    // way2 & way3
    float *way23_o1 = way23_1.cpu_forward(input, batch_size);
    cpu_relu(way23_o1, batch_size * 384 * size * size);
    float *way2_o2 = way2_2.cpu_forward(way23_o1, batch_size);
    cpu_relu(way2_o2, batch_size * 384 * size * size);
    float *way3_o2 = way3_2.cpu_forward(way23_o1, batch_size);
    cpu_relu(way3_o2, batch_size * 384 * size * size);
    float *concat_in1[] = {way2_o2, way3_o2};
    int concat_ch1[] = {384, 384};
    float *way23_o = cpu_channel_concat(concat_in1, 2, batch_size, concat_ch1, size, size);
    free(way23_o1);
    free(way2_o2);
    free(way3_o2);
    // way4 & way5
    float *way45_o1 = way45_1.cpu_forward(input, batch_size);
    cpu_relu(way45_o1, batch_size * 448 * size * size);
    float *way45_o2 = way45_2.cpu_forward(way45_o1, batch_size);
    cpu_relu(way45_o2, batch_size * 384 * size * size);
    float *way4_o3 = way4_3.cpu_forward(way45_o2, batch_size);
    cpu_relu(way4_o3, batch_size * 384 * size * size);
    float *way5_o3 = way5_3.cpu_forward(way45_o2, batch_size);
    cpu_relu(way5_o3, batch_size * 384 * size * size);
    float *concat_in2[] = {way4_o3, way5_o3};
    int concat_ch2[] = {384, 384};
    float *way45_o = cpu_channel_concat(concat_in2, 2, batch_size, concat_ch2, size, size);
    free(way45_o1);
    free(way45_o2);
    free(way4_o3);
    free(way5_o3);
    // way6
    float *way6_o1 = avgpool.cpu_forward(input, batch_size);
    float *way6_o = way6.cpu_forward(way6_o1, batch_size);
    cpu_relu(way6_o, batch_size * 192 * size * size);
    free(way6_o1);
    // final
    float *concat_in_final[] = {way1_o, way23_o, way45_o, way6_o};
    int concat_ch_final[] = {320, 768, 768, 192};
    float *final = cpu_channel_concat(concat_in_final, 4, batch_size, concat_ch_final, size, size);
    free(way1_o);
    free(way23_o);
    free(way45_o);
    free(way6_o);
    return final;
}

float* InceptionLayer6 :: gpu_forward(float *input, const int batch_size) {
    dim3 grid_conv(8, batch_size);
    dim3 block_conv(32);
    dim3 grid_act(32);
    dim3 block_act(32);
    // way1
    float *way1_o = way1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way1_o, batch_size * 320 * size * size);
    // way2 & way3
    float *way23_o1 = way23_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way23_o1, batch_size * 384 * size * size);
    float *way2_o2 = way2_2.basic_forward(grid_conv, block_conv, way23_o1, batch_size);
    relu(grid_act, block_act, way2_o2, batch_size * 384 * size * size);
    float *way3_o2 = way3_2.basic_forward(grid_conv, block_conv, way23_o1, batch_size);
    relu(grid_act, block_act, way3_o2, batch_size * 384 * size * size);
    float *concat_in1[] = {way2_o2, way3_o2};
    int concat_ch1[] = {384, 384};
    float *way23_o = channel_concat(grid_act, block_act, concat_in1, 2, batch_size, concat_ch1, size, size);
    cudaFree(way23_o1);
    cudaFree(way2_o2);
    cudaFree(way3_o2);
    // way4 & way5
    float *way45_o1 = way45_1.basic_forward(grid_conv, block_conv, input, batch_size);
    relu(grid_act, block_act, way45_o1, batch_size * 448 * size * size);
    float *way45_o2 = way45_2.basic_forward(grid_conv, block_conv, way45_o1, batch_size);
    relu(grid_act, block_act, way45_o2, batch_size * 384 * size * size);
    float *way4_o3 = way4_3.basic_forward(grid_conv, block_conv, way45_o2, batch_size);
    relu(grid_act, block_act, way4_o3, batch_size * 384 * size * size);
    float *way5_o3 = way5_3.basic_forward(grid_conv, block_conv, way45_o2, batch_size);
    relu(grid_act, block_act, way5_o3, batch_size * 384 * size * size);
    float *concat_in2[] = {way4_o3, way5_o3};
    int concat_ch2[] = {384, 384};
    float *way45_o = channel_concat(grid_act, block_act, concat_in2, 2, batch_size, concat_ch2, size, size);
    cudaFree(way45_o1);
    cudaFree(way45_o2);
    cudaFree(way4_o3);
    cudaFree(way5_o3);
    // way6
    float *way6_o1 = avgpool.basic_forward(grid_conv, block_conv, input, batch_size);
    float *way6_o = way6.basic_forward(grid_conv, block_conv, way6_o1, batch_size);
    relu(grid_act, block_act, way6_o, batch_size * 192 * size * size);
    cudaFree(way6_o1);
    // final
    float *concat_in_final[] = {way1_o, way23_o, way45_o, way6_o};
    int concat_ch_final[] = {320, 768, 768, 192};
    float *final = channel_concat(grid_act, block_act, concat_in_final, 4, batch_size, concat_ch_final, size, size);
    cudaFree(way1_o);
    cudaFree(way23_o);
    cudaFree(way45_o);
    cudaFree(way6_o);
    return final;
}

InceptionLayer6 :: ~InceptionLayer6() {}