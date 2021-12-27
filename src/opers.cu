# include "opers.h"

double* cpu_pad(double* input, const int batch_size, const int channels, const int size, const int pads[]){
    double *output;
    int new_r_size = size+pads[0]+pads[2];
    int new_c_size = size+pads[1]+pads[3];
    output = (double*) malloc (sizeof(double) * batch_size * channels * new_r_size * new_c_size);
    int pos, pos_in;

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for (int channel_idx = 0; channel_idx < channels; channel_idx++){
            for (int row = 0; row < new_r_size;row++){
                for ( int col = 0; col < new_c_size; col++){
                    pos = batch_idx * channels * new_r_size * new_c_size + channel_idx * new_r_size * new_c_size + row * new_c_size + col;
                    if (row<pads[0] || row>=pads[0]+size || col<pads[1] || col>=pads[1]+size){                       
                        output[pos] = 0;
                    }
                    else{
                        pos_in = batch_idx * channels * size * size + channel_idx * size * size + (row-pads[0]) * size + (col-pads[1]);
                        output[pos] = input[pos_in];
                    }
                }
            }
        }
    }
    return output;
}


__global__ void forward_gather(double *input, double *output, const int size, const int channels, const int channel_idx) {
    const int batch_id = blockIdx.y;
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int begin_idx = 1ll * size * size * thread_pos / total_threads;
    const int end_idx = 1ll * size * size * (thread_pos + 1) / total_threads;
    int i_pos, o_pos;
    for (int i = begin_idx; i < end_idx; ++ i){
        i_pos = batch_id * channels * size * size + channel_idx * size * size;
        o_pos = batch_id * size * size;
        output[o_pos + i] = input[i_pos + i];
    }
}

double* cpu_gather(double *input, const int batch_size, const int size, const int channels, const int channel_idx) {
    double *output;
    output = (double*) malloc (sizeof(double) * batch_size * size * size);
    int i_pos, o_pos;
    for (int i = 0; i < batch_size; ++ i){
        for(int j = 0; j < size * size; ++ j){
            i_pos = i * channels * size * size + channel_idx * size * size;
            o_pos = i * size * size;
            output[j + o_pos] = input[j + i_pos];
        }
    }
    return output;
}

double* gather(dim3 grid, dim3 block, double *input, const int batch_size, const int size, const int channels, const int channel_idx) {
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * size * size);
    forward_gather <<<grid, block>>> (input, output, size, channels, channel_idx);
    return output;
}


__global__ void forward_linear_transform(double *input, double *output, const int size, const double alpha, const double beta) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int begin_idx = 1ll * size * thread_pos / total_threads;
    const int end_idx = 1ll * size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i)
        output[i] = input[i] * alpha + beta;
}

double* cpu_linear_transform(double *input, const int size, const double alpha, const double beta, bool inplace) {
    double *output;
    if (inplace) output = input;
    else output = (double*) malloc (sizeof(double) * size);
    for (int i = 0; i < size; ++ i)
        output[i] = input[i] * alpha + beta;
    return output;
}

double* linear_transform(dim3 grid, dim3 block, double *input, const int size, const double alpha, const double beta, bool inplace) {
    double *output;
    if (inplace) output = input;
    else cudaMalloc((void **)&output, sizeof(double) * size);
    forward_linear_transform <<<grid, block>>> (input, output, size, alpha, beta);
    return output;
}

double* cpu_channel_concat(double *input[], const int num, const int batch_size, const int channel[], const int size_r, const int size_c) {
    int total_channels = 0;
    for (int i = 0; i < num; ++ i) total_channels += channel[i];
    double *output;
    output = (double*) malloc (sizeof(double) * batch_size * total_channels * size_r * size_c);
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

__global__ void forward_channel_concat_2(double *input1, double *input2, double *output, const int batch_size, const int channel1, const int channel2, const int size_r, const int size_c) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    const int total_threads = blockDim.x * gridDim.x;
    const int in_size_1 = channel1 * size_r * size_c;
    const int in_size_2 = channel2 * size_r * size_c;
    const int total_channels = channel1 + channel2;
    const int out_size = total_channels * size_r * size_c;
    const int begin_idx = 1ll * out_size * thread_pos / total_threads;
    const int end_idx = 1ll * out_size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int c = temp % size_c;
        const int r = (temp /= size_c) % size_r;
        const int ch = (temp /= size_r) % total_channels;
        const int in_size = (ch < channel1 ? in_size_1 : in_size_2);
        const int cur_channels = (ch < channel1 ? 0 : channel1);
        double *cur_input = (ch < channel1 ? input1 : input2);
        output[batch_id * out_size + ch * size_r * size_c + r * size_c + c] = cur_input[batch_id * in_size + (ch - cur_channels) * size_r * size_c + r * size_c + c];
    }
}

__global__ void forward_channel_concat_3(double *input1, double *input2, double *input3, double *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int size_r, const int size_c) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    const int total_threads = blockDim.x * gridDim.x;
    const int in_size_1 = channel1 * size_r * size_c;
    const int in_size_2 = channel2 * size_r * size_c;
    const int in_size_3 = channel3 * size_r * size_c;
    const int total_channels = channel1 + channel2 + channel3;
    const int out_size = total_channels * size_r * size_c;
    const int begin_idx = 1ll * out_size * thread_pos / total_threads;
    const int end_idx = 1ll * out_size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int c = temp % size_c;
        const int r = (temp /= size_c) % size_r;
        const int ch = (temp /= size_r) % total_channels;
        const int in_size = (ch < channel1 ? in_size_1 : ((ch < channel1 + channel2) ? in_size_2 : in_size_3));
        const int cur_channels = (ch < channel1 ? 0 : ((ch < channel1 + channel2) ? channel1 : (channel1 + channel2)));
        double *cur_input = (ch < channel1 ? input1 : ((ch < channel1 + channel2) ? input2 : input3));
        output[batch_id * out_size + ch * size_r * size_c + r * size_c + c] = cur_input[batch_id * in_size + (ch - cur_channels) * size_r * size_c + r * size_c + c];
    }
}

__global__ void forward_channel_concat_4(double *input1, double *input2, double *input3, double *input4, double *output, const int batch_size, const int channel1, const int channel2, const int channel3, const int channel4, const int size_r, const int size_c) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    const int total_threads = blockDim.x * gridDim.x;
    const int in_size_1 = channel1 * size_r * size_c;
    const int in_size_2 = channel2 * size_r * size_c;
    const int in_size_3 = channel3 * size_r * size_c;
    const int in_size_4 = channel4 * size_r * size_c;
    const int total_channels = channel1 + channel2 + channel3 + channel4;
    const int out_size = total_channels * size_r * size_c;
    const int begin_idx = 1ll * out_size * thread_pos / total_threads;
    const int end_idx = 1ll * out_size * (thread_pos + 1) / total_threads;
    for (int i = begin_idx; i < end_idx; ++ i) {
        int temp = i;
        const int c = temp % size_c;
        const int r = (temp /= size_c) % size_r;
        const int ch = (temp /= size_r) % total_channels;
        const int in_size = (ch < channel1 ? in_size_1 : ((ch < channel1 + channel2) ? in_size_2 : ((ch < channel1 + channel2 + channel3) ? in_size_3 : in_size_4)));
        const int cur_channels = (ch < channel1 ? 0 : ((ch < channel1 + channel2) ? channel1 : ((ch < channel1 + channel2 + channel3) ? (channel1 + channel2) : (channel1 + channel2 + channel3))));
        double *cur_input = (ch < channel1 ? input1 : ((ch < channel1 + channel2) ? input2 : ((ch < channel1 + channel2 + channel3) ? input3 : input4)));
        output[batch_id * out_size + ch * size_r * size_c + r * size_c + c] = cur_input[batch_id * in_size + (ch - cur_channels) * size_r * size_c + r * size_c + c];
    }
}

double* channel_concat(dim3 grid, dim3 block, double *input[], const int num, const int batch_size, const int channel[], const int size_r, const int size_c) {
    assert(num >= 2 && num <= 4);
    int total_channels = 0;
    for (int i = 0; i < num; ++ i) total_channels += channel[i];
    double *output;
    cudaMalloc((void **)&output, sizeof(double) * batch_size * total_channels * size_r * size_c);
    if (num == 2) 
        forward_channel_concat_2 <<<grid, block>>> (input[0], input[1], output, batch_size, channel[0], channel[1], size_r, size_c);
    else if (num == 3)
        forward_channel_concat_3 <<<grid, block>>> (input[0], input[1], input[2], output, batch_size, channel[0], channel[1], channel[2], size_r, size_c);
    else
        forward_channel_concat_4 <<<grid, block>>> (input[0], input[1], input[2], input[3], output, batch_size, channel[0], channel[1], channel[2], channel[3], size_r, size_c);
    return output;
}
