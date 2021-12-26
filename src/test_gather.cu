# include "opers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

int main(){
    double *input;
    const int size = 4;
    const int batch_size = 4;
    const int channels = 3;
    const int input_size = batch_size * channels * size * size;
    const int output_size = batch_size * size * size;
    input = (double*) malloc (sizeof(double) * input_size);
    for (int i = 0; i < input_size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * input_size);
    cudaMemcpy(cuda_input, input, sizeof(double) * input_size, cudaMemcpyHostToDevice);
    
    double *cpu_output_0 = cpu_gather(input, batch_size, size, channels, 0);
    double *cpu_output_1 = cpu_gather(input, batch_size, size, channels, 1);
    double *cpu_output_2 = cpu_gather(input, batch_size, size, channels, 2);

    dim3 grid(4, batch_size);
    dim3 block(4);

    double *cuda_output_0 = gather(grid, block, cuda_input, batch_size, size, channels, 0);
    double *cuda_output_1 = gather(grid, block, cuda_input, batch_size, size, channels, 1);
    double *cuda_output_2 = gather(grid, block, cuda_input, batch_size, size, channels, 2);
    double *cuda_output_device_0;
    double *cuda_output_device_1;
    double *cuda_output_device_2;
    cuda_output_device_0 = (double*) malloc (sizeof(double) * output_size);
    cuda_output_device_1 = (double*) malloc (sizeof(double) * output_size);
    cuda_output_device_2 = (double*) malloc (sizeof(double) * output_size);
    cudaMemcpy(cuda_output_device_0, cuda_output_0, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_output_device_1, cuda_output_1, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_output_device_2, cuda_output_2, sizeof(double) * output_size, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (int i = 0; i < output_size; ++i){
        max_error = max(max_error, fabs(cuda_output_device_0[i] - cpu_output_0[i]));
        max_error = max(max_error, fabs(cuda_output_device_1[i] - cpu_output_1[i]));
        max_error = max(max_error, fabs(cuda_output_device_2[i] - cpu_output_2[i]));
    }
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    free(input);
    free(cpu_output);
    free(cuda_output_device);
    return 0;
}