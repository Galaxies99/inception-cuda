# include "conv.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

int batch_size = 4, in_channels = 2048, out_channels = 320, in_size_r = 8, in_size_c = 8, kernel_r = 1, kernel_c = 1, stride_r = 1, stride_c = 1, padding_r = 0, padding_c = 0;
ConvolutionLayer conv(in_channels, out_channels, in_size_r, in_size_c, kernel_r, kernel_c, stride_r, stride_c, padding_r, padding_c);

int main() {
    double *input;
    int out_size_r, out_size_c;
    conv.get_output_size(out_size_r, out_size_c);

    input = (double*) malloc (sizeof(double) * batch_size * in_channels * in_size_r * in_size_c);
    for (int i = 0; i < batch_size * in_channels * in_size_r * in_size_c; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    double *cpu_output = conv.cpu_forward(input, batch_size);

    dim3 grid(8, batch_size);
    dim3 block(32);

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_channels * in_size_r * in_size_c);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_channels * in_size_r * in_size_c, cudaMemcpyHostToDevice);
    double *cuda_output = conv.basic_forward(grid, block, cuda_input, batch_size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * batch_size * out_channels * out_size_r * out_size_c);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * batch_size * out_channels * out_size_r * out_size_c, cudaMemcpyDeviceToHost);

    /*
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    double *cudnn_output = conv.cudnn_forward(cudnn, input, batch_size);
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * batch_size * out_channels * out_size_r * out_size_c);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * batch_size * out_channels * out_size_r * out_size_c, cudaMemcpyDeviceToHost);
    */

    double max_error = 0.0;
    // double max_error_cudnn = 0.0;
    for (int i = 0; i < batch_size * out_channels * out_size_r * out_size_c; ++ i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    //    max_error_cudnn = max(max_error, fabs(cudnn_output_device[i] - cpu_output[i]));
    }
    cout << "Max Error (CUDA vs CPU) = " << max_error << endl;
    // cout << "Max Error (CUDNN vs CPU) = " << max_error_cudnn << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    return 0;   
}