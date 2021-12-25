# include "part1.h"
# include "pooling.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>


using namespace std;
const int batch_size = 4, channels = 5, size = 4, kernel_size = 3, stride = 1;
const int len = size / stride + (size % stride != 0);    
const int output_size = channels * len * len;
MaxpoolingLayer maxpool(channels, size, kernel_size, stride);
int maxpool_test() {
    float *input;

    input = (float*) malloc (sizeof(float) * batch_size * channels * size * size);
    for (int i = 0; i < batch_size * channels * size * size; ++ i)
        input[i] = (float) (rand() % 32768) / 32768.0;
    
    float *cpu_output = maxpool.cpu_forward(input, batch_size);

    dim3 grid(batch_size);
    dim3 block(channels);

    float *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(float) * batch_size * channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(float) * batch_size * channels * size * size, cudaMemcpyHostToDevice);
    float *cuda_output = maxpool.basic_forward(grid, block, cuda_input, batch_size);
    float *cuda_output_device;
    cuda_output_device = (float*) malloc (sizeof(float) * batch_size * output_size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(float) * batch_size * output_size, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < batch_size * output_size; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect.\n";
    else cout << "Correct.\n";
    return 0;    
}