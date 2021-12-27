# include "layers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

const int batch_size = 4, in_channels = 2048, size = 8;
InceptionLayer6 layer(in_channels, size);

int main() {
    double *input;

    input = (double*) malloc (sizeof(double) * batch_size * in_channels * size * size);
    for (int i = 0; i < batch_size * in_channels * size * size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    cout << "cpu begin.\n";
    double *cpu_output = layer.cpu_forward(input, batch_size);
    cout << "cpu end.\n";

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_channels * size * size, cudaMemcpyHostToDevice);
    cout << "gpu begin.\n";
    double *cuda_output = layer.gpu_forward(cuda_input, batch_size);
    cout << "gpu end.\n";
    int out_channels = layer.get_out_channels(), out_size = layer.get_out_size();
    int output_N = batch_size * out_channels * out_size * out_size;
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * output_N);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * output_N, cudaMemcpyDeviceToHost);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cout << "cudnn begin.\n";
    double *cudnn_output = layer.cudnn_forward(cudnn, cuda_input, batch_size);
    cout << "cudnn end.\n";
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * output_N);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * output_N, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    double max_error_cudnn = 0.0;
    for (int i = 0; i < output_N; ++ i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
        max_error_cudnn = max(max_error_cudnn, fabs(cudnn_output_device[i] - cpu_output[i]));
    }
    cout << "Max Error (CUDA vs CPU) = " << max_error << endl;
    cout << "Max Error (CUDNN vs CPU) = " << max_error_cudnn << endl;
    if (max_error > 1e-5 || max_error_cudnn > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cudnnDestroy(cudnn);
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    cudaFree(cudnn_output);
    free(input);
    free(cpu_output);
    free(cuda_output_device);
    free(cudnn_output_device);
    return 0;   
}