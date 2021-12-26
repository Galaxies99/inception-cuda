# include "fc.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>
using namespace std;

int batch_size = 4, in_features = 10, out_features = 6;
FullyConnectedLayer fc(in_features, out_features);

int main() {
    double *input;
    input = (double*) malloc (sizeof(double) * batch_size * in_features);
    for (int i = 0; i < batch_size * in_features; ++i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    double *cpu_output = fc.cpu_forward(input, batch_size);

    dim3 grid(2, batch_size);
    dim3 block(4);

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_features);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_features, cudaMemcpyHostToDevice);
    double *cuda_output = fc.basic_forward(grid, block, cuda_input, batch_size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * batch_size * out_features);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * batch_size * out_features, cudaMemcpyDeviceToHost);
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    double *cudnn_output = fc.cudnn_forward(cudnn, cuda_input, batch_size);
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * batch_size * out_features);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * batch_size * out_features, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    double max_error_cudnn = 0.0;
    for (int i = 0; i < batch_size * out_features; ++i) {
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
