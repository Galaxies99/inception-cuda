# include "pooling.h"

void maxpooling_cpu(float* bottom_data, float* top_data, int* maxidx, const int batch_size, const int channel, const int size, const int kernel_size, const int stride){
    int i , j, u, v, pos, index, idx;
    float s;
    int len = size / stride + (size % stride != 0);
    
    int input_size = batch_size * channel * size *size;
    // int output_size = batch_size * channel * len * len;
    int index2;

    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for(int channel_idx = 0; channel_idx < channel; channel_idx++){
            pos = batch_idx * channel + channel_idx;
            index2 = pos * len * len;
            for (i = 0; i < len; ++i){
                for (j = 0; j < len; ++j)
                {
                    index = pos * size * size + i * stride * size + j * stride;
                    s=-1e18;
                    for (u = 0; u < kernel_size && (u + stride * i) < size; ++u)
                        for (v = 0; v < kernel_size && (v + stride * j) < size; ++v)
                            if (index + u * size + v < input_size && *(bottom_data + index + u * size + v) > s){
                                s = *(bottom_data + index + u * size + v);
                                idx = index + u * size + v;
                            }
                    *(top_data + index2) = s;
                    *(maxidx + index2) = idx;
                    ++index2;
                }
            }
        }
    }
}

void meanpooling_cpu(float* bottom_data, float* top_data, const int batch_size, const int channel, const int size, const int kernel_size, const int stride, const int padding){
    int i , j, u, v, pos, index;
    float s;
    int size_padding = size + padding * 2 - kernel_size + 1;
    int len = size_padding / stride + (size_padding % stride != 0);
    
    // int input_size = batch_size * channel * size *size;
    // int output_size = batch_size * channel * len * len;
    int index2;

    float weight = 1.0 / (kernel_size*kernel_size);
    // printf("weight is %0.4f\n", weight);

    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for(int channel_idx = 0; channel_idx < channel; channel_idx++){
            pos = batch_idx * channel + channel_idx;
            index2 = pos * len * len;
            for (i = 0; i < len; ++i){
                for (j = 0; j < len; ++j)
                {
                    index = pos * size * size + i * stride * size + j * stride;
                    s = 0.0;
                    for (u = -padding; u < kernel_size-padding && (u + stride * i) < size; ++u)
                        for (v = -padding; v < kernel_size-padding && (v + stride * j) < size; ++v){
                            if (i * stride + u >= 0 && j * stride + v >= 0 && i* stride + u < size && j * stride + v < size){
                                s += *(bottom_data + index + u * size + v);
                            }
                        }
                    *(top_data + index2) = s * weight;
                    ++index2;
                }
            }
        }
    }
}

// batch * channel * height * width
__global__ void maxpool_forward(float* bottom_data, float* top_data, int* maxidx, const int size, const int kernel_size, const int stride)
{
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int i , j, u, v, index, idx;
    float s;
    int len = size / stride + (size % stride != 0);
    int index2 = thread_pos * len * len;
    for (i = 0; i < len; ++i)
        for (j = 0; j < len; ++j)
        {
            index = thread_pos * size * size + i * stride * size + j * stride;
            s=-1e18;
            for (u = 0; u < kernel_size && (u + stride * i) < size; ++u)
                for (v = 0; v < kernel_size && (v + stride * j) < size; ++v)
                    if (*(bottom_data + index + u * size + v) > s){
                        s = *(bottom_data + index + u * size + v);
                        idx = index + u * size + v;
                    }
            *(top_data + index2) = s;
            *(maxidx + index2) = idx;
            ++index2;
        }
}

__global__ void meanpool_forward(float* bottom_data, float* top_data, const int size, const int kernel_size, const int stride, const int padding)
{
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int i , j, u, v, index;
    float s = 0;
    int size_padding = size + padding * 2 - kernel_size + 1;
    int len = size_padding / stride + (size_padding % stride != 0);
    int index2 = thread_pos * len * len;
    for (i = 0; i < len; ++i)
        for (j = 0; j < len; ++j)
        {
            s = 0;
            index = thread_pos * size * size + i * stride * size + j * stride;
            for (u = -padding; u < kernel_size-padding && (u + stride * i) < size; ++u)
                for (v = -padding; v < kernel_size-padding && (v + stride * j) < size; ++v){
                    if (i * stride + u >= 0 && j * stride + v >= 0 && i* stride + u < size && j * stride + v < size){
                        s += *(bottom_data + index + u * size + v) / (kernel_size * kernel_size);
                    }
                }
            *(top_data + index2) = s;
            ++index2;
        }
}

// Construction function of convolution layer.
MaxpoolingLayer :: MaxpoolingLayer(int _channels, int _size, int _kernel_size, int _stride){
    channels = _channels;
    size = _size;
    kernel_size = _kernel_size;
    stride = _stride;
    int len = size / stride + (size % stride != 0);
    output_size = channels * len * len;
}
// Destruction function of maxpooling layer.
MaxpoolingLayer :: ~MaxpoolingLayer() {}

float* MaxpoolingLayer :: basic_forward(dim3 grid, dim3 block, float *input, const int batch_size) {
    float *output;
    int* maxidx;
    cudaMalloc((void **)&output, sizeof(float) * batch_size * output_size);
    cudaMemset(output, 0, sizeof(float) * batch_size * output_size);
    cudaMalloc((void **)&maxidx, sizeof(float) * batch_size * output_size);
    cudaMemset(maxidx, 0, sizeof(float) * batch_size * output_size);

    maxpool_forward <<<batch_size, channels>>> (input, output, maxidx, size, kernel_size, stride);
    cudaDeviceSynchronize();
    cudaFree(maxidx);
    return output;
}

float* MaxpoolingLayer :: cpu_forward(float *input, const int batch_size) {
    float *output;
    int *maxidx;
    output = (float *) malloc (sizeof(float) * batch_size * output_size);
    maxidx = (int *) malloc (sizeof(int) * batch_size * output_size);
    maxpooling_cpu(input, output, maxidx, batch_size, channels, size, kernel_size, stride);
    free(maxidx);
    return output;
}

// Construction function of convolution layer.
MeanpoolingLayer :: MeanpoolingLayer(int _channels, int _size, int _kernel_size, int _stride, int _padding){
    channels = _channels;
    size = _size;
    kernel_size = _kernel_size;
    stride = _stride;
    padding = _padding;
    int size_padding = size + 2*padding;
    int len = size_padding / stride + (size_padding % stride != 0);
    output_size = channels * len * len;
}
// Destruction function of meanpooling layer.
MeanpoolingLayer :: ~MeanpoolingLayer() {}

float* MeanpoolingLayer :: basic_forward(dim3 grid, dim3 block, float *input, const int batch_size) {
    float *output;
    cudaMalloc((void **)&output, sizeof(float) * batch_size * output_size);
    cudaMemset(output, 0, sizeof(float) * batch_size * output_size);

    meanpool_forward <<<batch_size, channels>>> (input, output, size, kernel_size, stride, padding);
    cudaDeviceSynchronize();
    return output;
}

float* MeanpoolingLayer :: cpu_forward(float *input, const int batch_size) {
    float *output;
    output = (float *) malloc (sizeof(float) * batch_size * output_size);
    meanpooling_cpu(input, output, batch_size, channels, size, kernel_size, stride, padding);
    return output;
}