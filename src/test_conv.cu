# include "conv.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

int batch_size = 4, in_channels = 5, out_channels = 6, in_size_r = 30, in_size_c = 30, kernel_r = 3, kernel_c = 3, stride_r = 1, stride_c = 1, padding_r = 1, padding_c = 1;
ConvolutionLayer conv(in_channels, out_channels, in_size_r, in_size_c, kernel_r, kernel_c, stride_r, stride_c, padding_r, padding_c);

int main() {
    float *input;
    int out_size_r, out_size_c;
    conv.get_output_size(out_size_r, out_size_c);

    input = (float*) malloc (sizeof(float) * batch_size * in_channels * in_size_r * in_size_c);
    for (int i = 0; i < batch_size * in_channels * in_size_r * in_size_c; ++ i)
        input[i] = (float) rand() / 32768.0;
    
    float *cpu_output = conv.cpu_forward(input, batch_size);

    dim3 grid(2, batch_size);
    dim3 block(4);

    float *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(float) * batch_size * in_channels * in_size_r * in_size_c);
    cudaMemcpy(cuda_input, input, sizeof(float) * batch_size * in_channels * in_size_r * in_size_c, cudaMemcpyHostToDevice);
    float *cuda_output = conv.basic_forward(grid, block, cuda_input, batch_size);
    float *cuda_output_device;
    cuda_output_device = (float*) malloc (sizeof(float) * batch_size * out_channels * out_size_r * out_size_c);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(float) * batch_size * out_channels * out_size_r * out_size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < batch_size * out_channels * out_size_r * out_size_c; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect.";
    else cout << "Correct.";
    return 0;
    
}