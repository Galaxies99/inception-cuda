# include "opers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;


int main() {
    float *input;
    const int size = 1024;
    const float alpha = 0.2, beta = 0.3;

    input = (float*) malloc (sizeof(float) * size);
    for (int i = 0; i < size; ++ i)
        input[i] = (float) (rand() % 32768) / 32768.0;

    float *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(float) * size);
    cudaMemcpy(cuda_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    float *cpu_output = cpu_linear_transform(input, size, alpha, beta);

    dim3 grid(4);
    dim3 block(4);

    float *cuda_output = linear_transform(grid, block, cuda_input, size, alpha, beta);
    float *cuda_output_device;
    cuda_output_device = (float*) malloc (sizeof(float) * size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cout << "====> Linear Transform\n";
    float max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cout << endl << "Inplace testing: \n";

    float *cuda_input_device;
    cuda_input_device = (float*) malloc (sizeof(float) * size);
    cudaMemcpy(cuda_input_device, cuda_input, sizeof(float) * size, cudaMemcpyDeviceToHost);

    max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_input_device[i] - input[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cout << endl << "====> Concat\n";
    float *inputs[4], *cuda_inputs[4];
    const int batch_size = 4, channels[] = {1, 2, 3, 5}, total_channels = 11, size_r = 5, size_c = 5;
    for (int i = 0; i < 4; ++ i) {
        inputs[i] = (float*) malloc (sizeof(float) * batch_size * channels[i] * size_r * size_c);
        cudaMalloc((void **) &cuda_inputs[i], sizeof(float) * batch_size * channels[i] * size_r * size_c);
        for (int j = 0; j < batch_size * channels[i] * size_r * size_c; ++ j)
            inputs[i][j] = (rand() % 32768) / 32768.0;
        cudaMemcpy(cuda_inputs[i], inputs[i], sizeof(float) * batch_size * channels[i] * size_r * size_c, cudaMemcpyHostToDevice);
    }
    
    float *cpu_concat_output = cpu_channel_concat(inputs, 4, batch_size, channels, size_r, size_c);
    
    dim3 grid2(2, batch_size);
    dim3 block2(4);
    float *cuda_concat_output = channel_concat(grid2, block2, cuda_inputs, 4, batch_size, channels, size_r, size_c);
    float *cuda_concat_output_device;
    cuda_concat_output_device = (float*) malloc (sizeof(float) * batch_size * total_channels * size_r * size_c);
    cudaMemcpy(cuda_concat_output_device, cuda_concat_output, sizeof(float) * batch_size * total_channels * size_r * size_c, cudaMemcpyDeviceToHost);

    max_error = 0.0;
    for (int i = 0; i < batch_size * total_channels * size_r * size_c; ++ i)  
        max_error = max(max_error, fabs(cpu_concat_output[i] - cuda_concat_output_device[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    
    return 0;   
}