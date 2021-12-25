# include "pooling.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

const int batch_size = 4, channels = 2048, size = 4, kernel_size = 3, stride = 1, padding = 1;
MaxpoolingLayer maxpool(channels, size, kernel_size, stride);
MeanpoolingLayer meanpool(channels, size, kernel_size, stride, padding);
const int intput_size = batch_size * channels * size * size;
const int len = size / stride + (size % stride != 0);
const int size_padding = size + padding * 2 - kernel_size + 1;
const int len_mean =  size_padding / stride + (size_padding % stride != 0);
const int output_size_max = channels * len * len;
const int output_size_mean = channels * len_mean * len_mean;
int maxpool_test() {
    double *input;

    input = (double*) malloc (sizeof(double) * batch_size * channels * size * size);
    for (int i = 0; i < batch_size * channels * size * size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    double *cpu_output = maxpool.cpu_forward(input, batch_size);

    dim3 grid(8, batch_size);
    dim3 block(32);

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * channels * size * size, cudaMemcpyHostToDevice);
    double *cuda_output = maxpool.basic_forward(grid, block, cuda_input, batch_size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * batch_size * output_size_max);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * batch_size * output_size_max, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (int i = 0; i < batch_size * output_size_max; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect.\n";
    else cout << "Correct.\n";
    return 0;    
}

int meanpool_test() {
    double *input;

    input = (double*) malloc (sizeof(double) * batch_size * channels * size * size);
    for (int i = 0; i < batch_size * channels * size * size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    // printf("Input:\n");
    // for(int i = 0; i<batch_size;i++){
    //     for(int j = 0; j < channels; j++){
    //         // printf("%d %d\n",i,j);
    //         printf("------ Batch %d Channel %d ------\n", i, j);
    //         for(int x=0;x<size;x++){
    //             for(int y=0;y<size;y++){
    //                 printf("%0.4f ", input[i*channels*size*size+j*size*size+x*size+y]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    double *cpu_output = meanpool.cpu_forward(input, batch_size);


    dim3 grid(8, batch_size);
    dim3 block(32);

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * channels * size * size, cudaMemcpyHostToDevice);
    double *cuda_output = meanpool.basic_forward(grid, block, cuda_input, batch_size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * batch_size * output_size_mean);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * batch_size * output_size_mean, cudaMemcpyDeviceToHost);

    // printf("Output:\n");
    // for(int i = 0; i<batch_size;i++){
    //     for(int j = 0; j < channels; j++){
    //         printf("------ Batch %d Channel %d ------\n", i , j);
    //         for(int x=0;x<len_mean;x++){
    //             for(int y=0;y<len_mean;y++){
    //                 printf("%0.4f ", cpu_output[i*channels*len_mean*len_mean+j*len_mean*len_mean+x*len_mean+y]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // printf("Output:\n");
    // for(int i = 0; i<batch_size;i++){
    //     for(int j = 0; j < channels; j++){
    //         printf("------ Batch %d Channel %d ------\n", i , j);
    //         for(int x=0;x<len_mean;x++){
    //             for(int y=0;y<len_mean;y++){
    //                 printf("%0.4f ", cuda_output_device[i*channels*len_mean*len_mean+j*len_mean*len_mean+x*len_mean+y]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    double max_error = 0.0;
    for (int i = 0; i < batch_size * output_size_mean; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect.\n";
    else cout << "Correct.\n";
    return 0;    
}

int main(){
    printf("Max pooling test(input size: %d):\n", intput_size);
    maxpool_test();

    printf("Mean pooling test(input size: %d):\n", intput_size);
    meanpool_test();
    return 0;
}