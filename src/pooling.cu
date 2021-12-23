# include <cuda.h>
# include "pooling.h"


// batch * channel * height * width
__global__ void maxpool_forward(float* bottom_data, const int size, const int kernel_size, float* top_data, float* maxidx)
{
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int i , j, u, v, index, idx;
    float s;
    len = size / kernel_size + (size % kernel_size != 0)
    int index2 = thread_pos * len * len;
    for (i = 0; i < len; ++i)
        for (j = 0; j < len; ++j)
        {
            index = thread_pos * size * size + i * kernel_size * size + j * kernel_size;
            s=-10000.0;
            for (u = 0; u < kernel_size && (u + kernel_size * i) < size; ++u)
                for (v = 0; v < kernel_size && (v + kernel_size * j) < size; ++v)
                    if (*(bottom_data + index + u * size + v) > s){
                        s = *(bottom_data + index + u * size + v);
                        idx = index + u * size + v
                    }
            *(top_data + index2) = s;
            *(maxidx + index2) = idx
            ++index2;
        }
}

__global__ void meanpool_forward(float* bottom_data, const int size, const int kernel_size, float* top_data)
{
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int i , j, u, v, index, idx;
    float s = 0;
    len = size / kernel_size + (size % kernel_size != 0)
    int index2 = thread_pos * len * len;
    for (i = 0; i < len; ++i)
        for (j = 0; j < len; ++j)
        {
            index = thread_pos * size * size + i * kernel_size * size + j * kernel_size;
            for (u = 0; u < kernel_size && (u + kernel_size * i) < size; ++u)
                for (v = 0; v < kernel_size && (v + kernel_size * j) < size; ++v){
                        s += *(bottom_data + index + u * size + v) / (kernel_size * kernel_size);
                }
            *(top_data + index2) = s;
            ++index2;
        }
}