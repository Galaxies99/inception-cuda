# include <cuda.h>

// batch_size * channel * height * width
__global__ void maxpool_forward(float* bottom_data, const int height, const int width, const int kernel_size,float* top_data)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int i,j,u,v,index;
    float s;
    height_len = height / kernel_size + (height % kernel_size != 0)
    width_len = width / kernel_size + (width % kernel_size != 0)
    int index2=x * gridDim.y * height_len * width_len + y * height_len * width_len;
    for (i = 0; i < height_len; ++i)
        for (j = 0; j < width_len; ++j)
        {
            index = x * gridDim.y * height * width + y * height * width + i * kernel_size * width + j * kernel_size;
            s=-10000.0;
            for (u = 0; u < kernel_size && (u + kernel_size * i) < height; ++u)
                for (v = 0; v < kernel_size && (v + kernel_size * j) < width; ++v)
                    if (*(bottom_data + index + u * width + v) > s)
                        s = *(bottom_data + index + u * width + v);
            *(top_data + index2) = s;
            ++index2;
        }
}


__global__ maxpool_backward(float* top_data, const int height, const int width, const int kernel_size,float* bottom_data)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int i,j,u,v,index;
    int local_x, local_y;
    float s;
    height_len = height / kernel_size + (height % kernel_size != 0)
    width_len = width / kernel_size + (width % kernel_size != 0)
    int index2=x * gridDim.y * height_len * width_len + y * height_len * width_len;
    for (i = 0; i < height_len; ++i)
        for (j = 0; j < width_len; ++j)
        {
            index = x * gridDim.y * height * width + y * height * width + i * kernel_size * width + j * kernel_size;
            s=-10000.0;
            for (u = 0; u < kernel_size && (u + kernel_size * i) < height; ++u)
                for (v = 0; v < kernel_size && (v + kernel_size * j) < width; ++v)
                    if (*(bottom_data + index + u * width + v) > s){
                        s = *(bottom_data + index + u * width + v);
                        local_x = u;
                        local_y = v;
                    }
            *(bottom_data + index + local_x * width + local_y) = *(top_data + index2)
            for (u = 0; u < kernel_size && (u + kernel_size * i) < height && u != local_x; ++u)
                for (v = 0; v < kernel_size && (v + kernel_size * j) < width && v != local_y; ++v)
                    *(bottom_data + index + u * width + v) = 0;

            ++index2;
        }
}