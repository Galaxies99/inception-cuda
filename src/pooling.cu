// # include <cuda.h>
// # include "pooling.h"
#include<math.h>
#include<time.h>
#include<stdio.h>
#include"cuda_runtime.h"
#include<sys/time.h>
using namespace std;

void maxpool_cpu(float* bottom_data, const int channel, const int size, const int kernel_size, const int batch_size, float* top_data, int* maxidx){
    int i , j, u, v, pos, index, idx;
    float s;
    int len = size / kernel_size + (size % kernel_size != 0);
    
    int input_size = batch_size * channel * size *size;
    int output_size = batch_size * channel * len * len;
    int index2;

    for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
        for(int channel_idx = 0; channel_idx < channel; channel_idx++){
            pos = batch_idx * channel + channel_idx;
            index2 = pos * len * len;
            for (i = 0; i < len; ++i){
                for (j = 0; j < len; ++j)
                {
                    index = pos * size * size + i * kernel_size * size + j * kernel_size;
                    s=-10000.0;
                    for (u = 0; u < kernel_size && (u + kernel_size * i) < size; ++u)
                        for (v = 0; v < kernel_size && (v + kernel_size * j) < size; ++v)
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

void meanpool_cpu(float* bottom_data, const int channel, const int size, const int kernel_size, const int batch_size, float* top_data){
    int i , j, u, v, pos, index;
    float s;
    int len = size / kernel_size + (size % kernel_size != 0);
    
    int input_size = batch_size * channel * size *size;
    int output_size = batch_size * channel * len * len;
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
                    index = pos * size * size + i * kernel_size * size + j * kernel_size;
                    s = 0.0;
                    for (u = 0; u < kernel_size && (u + kernel_size * i) < size; ++u)
                        for (v = 0; v < kernel_size && (v + kernel_size * j) < size; ++v)
                            if (index + u * size + v < input_size){
                                // printf("%d %0.4f\t",index + u * size + v, *(bottom_data + index + u * size + v));
                                s += *(bottom_data + index + u * size + v);
                            }
                    // printf("\n");
                    *(top_data + index2) = s * weight;
                    ++index2;
                }
            }
        }
    }
}

// batch * channel * height * width
__global__ void maxpool_forward(float* bottom_data, const int size, const int kernel_size, float* top_data, int* maxidx)
{
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int i , j, u, v, index, idx;
    float s;
    int len = size / kernel_size + (size % kernel_size != 0);
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
                        idx = index + u * size + v;
                    }
            *(top_data + index2) = s;
            *(maxidx + index2) = idx;
            ++index2;
        }
}

__global__ void meanpool_forward(float* bottom_data, const int size, const int kernel_size, float* top_data)
{
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int i , j, u, v, index;
    float s = 0;
    int len = size / kernel_size + (size % kernel_size != 0);
    int index2 = thread_pos * len * len;
    for (i = 0; i < len; ++i)
        for (j = 0; j < len; ++j)
        {
            s = 0;
            index = thread_pos * size * size + i * kernel_size * size + j * kernel_size;
            for (u = 0; u < kernel_size && (u + kernel_size * i) < size; ++u)
                for (v = 0; v < kernel_size && (v + kernel_size * j) < size; ++v){
                        s += *(bottom_data + index + u * size + v) / (kernel_size * kernel_size);
                }
            *(top_data + index2) = s;
            ++index2;
        }
}

float maxpooling_cuda(float* bottom_data, const int channel, const int size, const int kernel_size, const int batch_size, float* top_data, int* maxidx){
    cudaEvent_t start, end;
    float duration = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // dim3 grid(m/tw, n/tw);
    // dim3 block(tw, tw);
    dim3 grid(batch_size);
    dim3 block(channel);
    float *Bottom_data, *Top_data;
    int *Maxidx;

    int input_size = batch_size * channel * size * size;
    int len = size / kernel_size + (size % kernel_size != 0);
    int output_size = batch_size * channel * len * len;

    cudaMalloc((void**)&Bottom_data, input_size*sizeof(float));
    cudaMalloc((void**)&Top_data, output_size*sizeof(float));
    cudaMalloc((void**)&Maxidx, output_size*sizeof(float));

    cudaMemcpy(Bottom_data, bottom_data, input_size*sizeof(float), cudaMemcpyHostToDevice);
    

    cudaEventRecord(start, 0);
    maxpool_forward <<<grid, block>>> (Bottom_data, size, kernel_size, Top_data, Maxidx);
    cudaDeviceSynchronize();

    cudaEventRecord(end, 0);

    cudaEventElapsedTime(&duration, start, end);

    cudaMemcpy(top_data, Top_data, output_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxidx, Maxidx, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(Bottom_data);
    cudaFree(Top_data);
    cudaFree(Maxidx);

    return duration;
}

float meanpooling_cuda(float* bottom_data, const int channel, const int size, const int kernel_size, const int batch_size, float* top_data){
    cudaEvent_t start, end;
    float duration = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // dim3 grid(m/tw, n/tw);
    // dim3 block(tw, tw);
    dim3 grid(batch_size);
    dim3 block(channel);
    float *Bottom_data, *Top_data;

    int input_size = batch_size * channel * size * size;
    int len = size / kernel_size + (size % kernel_size != 0);
    int output_size = batch_size * channel * len * len;

    cudaMalloc((void**)&Bottom_data, input_size*sizeof(float));
    cudaMalloc((void**)&Top_data, output_size*sizeof(float));

    cudaMemcpy(Bottom_data, bottom_data, input_size*sizeof(float), cudaMemcpyHostToDevice);
    

    cudaEventRecord(start, 0);
    meanpool_forward <<<grid, block>>> (Bottom_data, size, kernel_size, Top_data);
    cudaDeviceSynchronize();

    cudaEventRecord(end, 0);

    cudaEventElapsedTime(&duration, start, end);

    cudaMemcpy(top_data, Top_data, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(Bottom_data);
    cudaFree(Top_data);

    return duration;
}

int maxpooling(){
    // printf("here");
    const int batch_size = 32, channel = 3, size = 16, kernel_size = 3;
    float input[batch_size * channel * size * size];
    // printf("Input_size: %d\n", batch_size * channel * size * size);

    
    int len = size / kernel_size + (size % kernel_size != 0);
    float output[batch_size * channel * len * len];
    float output_cuda[batch_size * channel * len * len];
    int maxidx[batch_size * channel * len * len];
    int maxidx_cuda[batch_size * channel * len * len];
    // printf("Output_size: %d\n", batch_size * channel * len * len);

    for(int i = 0;i < batch_size * channel * size * size;i++){
        input[i] = 1.0 / (rand()%100+1.0);;
    }
    int res_idx;

    // printf("The input is as follows:\n");
    // for(int i = 0;i < batch_size; i++){
    //     for(int j = 0; j < channel; j++){
    //         printf("----------- Batch %d Channel %d -----------\n", i, j);
    //         for(int row = 0; row < size; row++){
    //             for(int col = 0; col < size; col++){
    //                 res_idx = i * channel * size * size + j * size * size + row * size + col;
    //                 printf("%.6f ", input[res_idx]);
    //             }
    //             printf("\n");
    //         }
    //     }        
    // }
    // printf("\n");


    float duration = 0.0;

    timeval start, end;
    gettimeofday(&start, 0);
    maxpool_cpu(input, channel, size, kernel_size, batch_size, output, maxidx);

    gettimeofday(&end, 0);
    duration = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

    printf("The duration of CPU calculation is %.2f us\nThe result is as follows:\n", duration);
    // for(int i = 0;i < batch_size; i++){
    //     for(int j = 0; j < channel; j++){
    //         printf("----------- Batch %d Channel %d -----------\n", i, j);
    //         for(int row = 0; row < len; row++){
    //             for(int col = 0; col < len; col++){
    //                 res_idx = i * channel * len * len + j * len * len + row * len + col;
    //                 printf("%.6f(%d)", output[res_idx], maxidx[res_idx]);
    //             }
    //             printf("\n");
    //         }
    //     }        
    // }

    // printf("\n");

    duration = maxpooling_cuda(input, channel, size, kernel_size, batch_size, output_cuda, maxidx_cuda);

    printf("The duration of CUDA calculation is %.2f us\nThe result is as follows:\n", duration);
    // for(int i = 0;i < batch_size; i++){
    //     for(int j = 0; j < channel; j++){
    //         printf("----------- Batch %d Channel %d -----------\n", i, j);
    //         for(int row = 0; row < len; row++){
    //             for(int col = 0; col < len; col++){
    //                 res_idx = i * channel * len * len + j * len * len + row * len + col;
    //                 printf("%.6f(%d)", output_cuda[res_idx], maxidx_cuda[res_idx]);
    //             }
    //             printf("\n");
    //         }
    //     }        
    // }
    // printf("\n");

    for(int i = 0;i < batch_size; i++){
        for(int j = 0; j < channel; j++){
            for(int row = 0; row < len; row++){
                for(int col = 0; col < len; col++){
                    res_idx = i * channel * len * len + j * len * len + row * len + col;
                    if(maxidx[res_idx]!=maxidx_cuda[res_idx]){
                        printf("Position %d %d %d %d maxidx is wrong.", i,j,row,col);
                    }
                    if(abs(output[res_idx]-output_cuda[res_idx])>1e-4){
                        printf("Position %d %d %d %d output is wrong.", i,j,row,col);
                    }
                }
            }
        }        
    }

    return 0;
}

int meanpooling(bool pr){
    // printf("here");
    const int batch_size = 32, channel = 3, size = 16, kernel_size = 3;
    float input[batch_size * channel * size * size];
    // printf("Input_size: %d\n", batch_size * channel * size * size);

    
    int len = size / kernel_size + (size % kernel_size != 0);
    float output[batch_size * channel * len * len];
    float output_cuda[batch_size * channel * len * len];
    // printf("Output_size: %d\n", batch_size * channel * len * len);

    for(int i = 0;i < batch_size * channel * size * size;i++){
        input[i] = 1.0 / (rand()%100+1.0);;
    }
    int res_idx;

    if(pr){
        printf("The input is as follows:\n");
        for(int i = 0;i < batch_size; i++){
            for(int j = 0; j < channel; j++){
                printf("----------- Batch %d Channel %d -----------\n", i, j);
                for(int row = 0; row < size; row++){
                    for(int col = 0; col < size; col++){
                        res_idx = i * channel * size * size + j * size * size + row * size + col;
                        printf("%.6f ", input[res_idx]);
                    }
                    printf("\n");
                }
            }        
        }
        printf("\n");
    }


    float duration = 0.0;

    timeval start, end;
    gettimeofday(&start, 0);
    meanpool_cpu(input, channel, size, kernel_size, batch_size, output);

    gettimeofday(&end, 0);
    duration = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);

    if(pr){
        printf("The duration of CPU calculation is %.2f us\nThe result is as follows:\n", duration);
        for(int i = 0;i < batch_size; i++){
            for(int j = 0; j < channel; j++){
                printf("----------- Batch %d Channel %d -----------\n", i, j);
                for(int row = 0; row < len; row++){
                    for(int col = 0; col < len; col++){
                        res_idx = i * channel * len * len + j * len * len + row * len + col;
                        printf("%.6f ", output[res_idx]);
                    }
                    printf("\n");
                }
            }        
        }
        printf("\n");
    }

    duration = meanpooling_cuda(input, channel, size, kernel_size, batch_size, output_cuda);

    if(pr){
        printf("The duration of CUDA calculation is %.2f us\nThe result is as follows:\n", duration);
        for(int i = 0;i < batch_size; i++){
            for(int j = 0; j < channel; j++){
                printf("----------- Batch %d Channel %d -----------\n", i, j);
                for(int row = 0; row < len; row++){
                    for(int col = 0; col < len; col++){
                        res_idx = i * channel * len * len + j * len * len + row * len + col;
                        printf("%.6f ", output_cuda[res_idx]);
                    }
                    printf("\n");
                }
            }        
        }
        printf("\n");
    }

    for(int i = 0;i < batch_size; i++){
        for(int j = 0; j < channel; j++){
            for(int row = 0; row < len; row++){
                for(int col = 0; col < len; col++){
                    res_idx = i * channel * len * len + j * len * len + row * len + col;
                    if(abs(output[res_idx]-output_cuda[res_idx])>1e-4){
                        printf("Position %d %d %d %d output is wrong.\n", i,j,row,col);
                    }
                }
            }
        }        
    }

    return 0;
}

int main(){
    // maxpooling(false);

    meanpooling(false);
    return 0;
}