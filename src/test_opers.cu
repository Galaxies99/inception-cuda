# include "opers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;


int main() {
    double *input;
    const int size = 1024;
    const double alpha = 0.2, beta = 0.3;

    input = (double*) malloc (sizeof(double) * size);
    for (int i = 0; i < size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);
    
    double *cpu_output = cpu_linear_transform(input, size, alpha, beta);

    dim3 grid(4);
    dim3 block(4);

    double *cuda_output = linear_transform(grid, block, cuda_input, size, alpha, beta);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * size, cudaMemcpyDeviceToHost);

    cout << "====> Linear Transform\n";
    double max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
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

    cudaFree(cuda_input);
    free(input);
    free(cuda_output_device);
    free(cuda_input_device);

    cout << endl << "====> Concat\n";
    double *inputs[4], *cuda_inputs[4];
    const int batch_size = 4, channels[] = {1, 2, 3, 5}, total_channels = 11, size_r = 5, size_c = 5;
    for (int i = 0; i < 4; ++ i) {
        inputs[i] = (double*) malloc (sizeof(double) * batch_size * channels[i] * size_r * size_c);
        cudaMalloc((void **) &cuda_inputs[i], sizeof(double) * batch_size * channels[i] * size_r * size_c);
        for (int j = 0; j < batch_size * channels[i] * size_r * size_c; ++ j)
            inputs[i][j] = (rand() % 32768) / 32768.0;
        cudaMemcpy(cuda_inputs[i], inputs[i], sizeof(double) * batch_size * channels[i] * size_r * size_c, cudaMemcpyHostToDevice);
    }
    
    double *cpu_concat_output = cpu_channel_concat(inputs, 4, batch_size, channels, size_r, size_c);
    
    dim3 grid2(2, batch_size);
    dim3 block2(4);
    double *cuda_concat_output = channel_concat(grid2, block2, cuda_inputs, 4, batch_size, channels, size_r, size_c);
    double *cuda_concat_output_device;
    cuda_concat_output_device = (double*) malloc (sizeof(double) * batch_size * total_channels * size_r * size_c);
    cudaMemcpy(cuda_concat_output_device, cuda_concat_output, sizeof(double) * batch_size * total_channels * size_r * size_c, cudaMemcpyDeviceToHost);

    max_error = 0.0;
    for (int i = 0; i < batch_size * total_channels * size_r * size_c; ++ i)  
        max_error = max(max_error, fabs(cpu_concat_output[i] - cuda_concat_output_device[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    
    for (int i = 0; i < 4; ++ i) {
        free(inputs[i]);
        cudaFree(cuda_inputs[i]);
    }
    free(cpu_concat_output);
    free(cuda_concat_output_device);
    cudaFree(cuda_concat_output);
    return 0;   
}