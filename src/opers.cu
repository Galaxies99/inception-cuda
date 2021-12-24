# include "opers.h"


__global__ void forward_linear_transform(float *input, float *output, const int size, const float alpha, const float beta) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int begin_idx = size * thread_pos / total_threads;
    const int end_idx = size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i)
        output[i] = input[i] * alpha + beta;
}

float* cpu_linear_transform(float *input, const int size, const float alpha, const float beta, bool inplace) {
    float *output;
    if (inplace) output = input;
    else output = (float*) malloc (sizeof(float) * size);
    for (int i = 0; i < size; ++ i)
        output[i] = input[i] * alpha + beta;
    return output;
}

float* linear_transform(dim3 grid, dim3 block, float *input, const int size, const float alpha, const float beta, bool inplace) {
    float *output;
    if (inplace) output = input;
    else cudaMalloc((void **)&output, sizeof(float) * size);
    forward_linear_transform <<<grid, block>>> (input, output, size, alpha, beta);
    return output;
}

float* cpu_channel_concat(float *input[], const int num, const int batch_size, const int channel[], const int size_r, const int size_c) {
    int total_channels = 0;
    for (int i = 0; i < num; ++ i) total_channels += channel[i];
    float *output;
    output = (float*) malloc (sizeof(float) * batch_size * total_channels * size_r * size_c);
    int cur_channels = 0;
    for (int i = 0; i < num; ++ i) {
        for (int b = 0; b < batch_size; ++ b)
            for (int ch = 0; ch < channel[i]; ++ ch)
                for (int r = 0; r < size_r; ++ r)
                    for (int c = 0; c < size_c; ++ c)
                        output[b * total_channels * size_r * size_c + (cur_channels + ch) * size_r * size_c + r * size_c + c] = input[i][b * channel[i] * size_r * size_c + ch * size_r * size_c + r * size_c + c];
        cur_channels += channel[i];
    }
    return output;
}

__global__ void forward_channel_concat_2(float *input1, float *input2, float *output, const int batch_size, const int channel1, const int channel2, const int size_r, const int size_c) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    const int total_threads = blockDim.x * gridDim.x;
    const int in_size_1 = channel1 * size_r * size_c;
    const int in_size_2 = channel2 * size_r * size_c;
    const int total_channels = channel1 + channel2;
    const int out_size = total_channels * size_r * size_c;
    const int begin_idx = out_size * thread_pos / total_threads;
    const int end_idx = out_size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int c = temp % size_c;
        const int r = (temp /= size_c) % size_r;
        const int ch = (temp /= size_r) % total_channels;
        const int in_size = (ch < channel1 ? in_size_1 : in_size_2);
        const int cur_channels = (ch < channel1 ? 0 : channel1);
        float *cur_input = (ch < channel1 ? input1 : input2);
        output[batch_id * out_size + ch * size_r * size_c + r * size_c + c] = cur_input[batch_id * in_size + (ch - cur_channels) * size_r * size_c + r * size_c + c];
    }
}

__global__ void forward_channel_concat_3(float *input1, float *input2, float *input3, float *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int size_r, const int size_c) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    const int total_threads = blockDim.x * gridDim.x;
    const int in_size_1 = channel1 * size_r * size_c;
    const int in_size_2 = channel2 * size_r * size_c;
    const int in_size_3 = channel3 * size_r * size_c;
    const int total_channels = channel1 + channel2 + channel3;
    const int out_size = total_channels * size_r * size_c;
    const int begin_idx = out_size * thread_pos / total_threads;
    const int end_idx = out_size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int c = temp % size_c;
        const int r = (temp /= size_c) % size_r;
        const int ch = (temp /= size_r) % total_channels;
        const int in_size = (ch < channel1 ? in_size_1 : ((ch < channel1 + channel2) ? in_size_2 : in_size_3));
        const int cur_channels = (ch < channel1 ? 0 : ((ch < channel1 + channel2) ? channel1 : (channel1 + channel2)));
        float *cur_input = (ch < channel1 ? input1 : ((ch < channel1 + channel2) ? input2 : input3));
        output[batch_id * out_size + ch * size_r * size_c + r * size_c + c] = cur_input[batch_id * in_size + (ch - cur_channels) * size_r * size_c + r * size_c + c];
    }
}

__global__ void forward_channel_concat_4(float *input1, float *input2, float *input3, float *input4, float *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int channel4, const int size_r, const int size_c) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    const int total_threads = blockDim.x * gridDim.x;
    const int in_size_1 = channel1 * size_r * size_c;
    const int in_size_2 = channel2 * size_r * size_c;
    const int in_size_3 = channel3 * size_r * size_c;
    const int in_size_4 = channel4 * size_r * size_c;
    const int total_channels = channel1 + channel2 + channel3 + channel4;
    const int out_size = total_channels * size_r * size_c;
    const int begin_idx = out_size * thread_pos / total_threads;
    const int end_idx = out_size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int c = temp % size_c;
        const int r = (temp /= size_c) % size_r;
        const int ch = (temp /= size_r) % total_channels;
        const int in_size = (ch < channel1 ? in_size_1 : ((ch < channel1 + channel2) ? in_size_2 : ((ch < channel1 + channel2 + channel3) ? in_size_3 : in_size_4)));
        const int cur_channels = (ch < channel1 ? 0 : ((ch < channel1 + channel2) ? channel1 : ((ch < channel1 + channel2 + channel3) ? (channel1 + channel2) : (channel1 + channel2 + channel3))));
        float *cur_input = (ch < channel1 ? input1 : ((ch < channel1 + channel2) ? input2 : ((ch < channel1 + channel2 + channel3) ? input3 : input4)));
        output[batch_id * out_size + ch * size_r * size_c + r * size_c + c] = cur_input[batch_id * in_size + (ch - cur_channels) * size_r * size_c + r * size_c + c];
    }
}

float* channel_concat(dim3 grid, dim3 block, float *input[], const int num, const int batch_size, const int channel[], const int size_r, const int size_c) {
    assert(num >= 2 && num <= 4);
    int total_channels = 0;
    for (int i = 0; i < num; ++ i) total_channels += channel[i];
    float *output;
    cudaMalloc((void **)&output, sizeof(float) * batch_size * total_channels * size_r * size_c);
    if (num == 2) 
        forward_channel_concat_2 <<<grid, block>>> (input[0], input[1], output, batch_size, channel[0], channel[1], size_r, size_c);
    else if (num == 3)
        forward_channel_concat_3 <<<grid, block>>> (input[0], input[1], input[2], output, batch_size, channel[0], channel[1], channel[2], size_r, size_c);
    else
        forward_channel_concat_4 <<<grid, block>>> (input[0], input[1], input[2], input[3], output, batch_size, channel[0], channel[1], channel[2], channel[3], size_r, size_c);
    return output;
}
