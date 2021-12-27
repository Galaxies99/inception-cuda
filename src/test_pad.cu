# include "opers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;


int main() {
    double *input;
    const int channels = 100, size_r = 32, size_c = 32, batch_size = 4, out_size_r = 34, out_size_c = 34;
    int input_N = batch_size * channels * size_r * size_c;
    int output_N = batch_size * channels * out_size_r * out_size_c;

    input = (double*) malloc (sizeof(double) * input_N);
    for (int i = 0; i < input_N; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * input_N);
    cudaMemcpy(cuda_input, input, sizeof(double) * input_N, cudaMemcpyHostToDevice);
    
    double *cpu_output = cpu_pad(input, batch_size, channels, size_r, size_c, 0, 2, 0, 2, 0);

    dim3 grid(4, batch_size);
    dim3 block(4);

    double *cuda_output = pad(grid, block, cuda_input, batch_size, channels, size_r, size_c, 0, 2, 0, 2, 0);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * output_N);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * output_N, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (int i = 0; i < output_N; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cudaFree(cuda_input);
    free(input);
    free(cuda_output_device);
    return 0;   
}