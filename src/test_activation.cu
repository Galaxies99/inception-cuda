# include "activation.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;


int main() {
    double *input;
    const int batch_size = 8, channel = 2, f_size = 8;
    int size = batch_size * channel * f_size * f_size;

    input = (double*) malloc (sizeof(double) * size);
    for (int i = 0; i < size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);
    
    double *cpu_output = cpu_relu(input, size);

    dim3 grid(4);
    dim3 block(4);

    double *cuda_output = relu(grid, block, cuda_input, size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * size, cudaMemcpyDeviceToHost);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    double *cudnn_output = cudnn_relu(cudnn, cuda_input, batch_size, channel, f_size, f_size);
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * size);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * size, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    double max_error_cudnn = 0.0;
    for (int i = 0; i < size; ++ i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
        max_error_cudnn = max(max_error_cudnn, fabs(cudnn_output_device[i] - cpu_output[i]));
    }
    cout << "Max Error (CUDA vs CPU) = " << max_error << endl;
    cout << "Max Error (CUDNN vs CPU) = " << max_error_cudnn << endl;
    if (max_error > 1e-5 || max_error_cudnn > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cout << endl << "Inplace testing: \n";

    double *cuda_input_device;
    cuda_input_device = (double*) malloc (sizeof(double) * size);
    cudaMemcpy(cuda_input_device, cuda_input, sizeof(double) * size, cudaMemcpyDeviceToHost);

    max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_input_device[i] - input[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cudnnDestroy(cudnn);
    cudaFree(cuda_input);
    free(input);
    free(cuda_output_device);
    free(cudnn_output_device);
    free(cuda_input_device);
    return 0;   
}