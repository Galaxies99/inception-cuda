# include "layers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

const int batch_size = 4, in_channels = 2048, size = 8, out_channels = 2048;
InceptionLayer6 layer(in_channels, size);

int main() {
    float *input;

    input = (float*) malloc (sizeof(float) * batch_size * in_channels * size * size);
    for (int i = 0; i < batch_size * in_channels * size * size; ++ i)
        input[i] = (float) (rand() % 32768) / 32768.0;
    
    float *cpu_output = layer.cpu_forward(input, batch_size);

    float *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(float) * batch_size * in_channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(float) * batch_size * in_channels * size * size, cudaMemcpyHostToDevice);
    float *cuda_output = conv.basic_forward(cuda_input, batch_size);
    float *cuda_output_device;
    cuda_output_device = (float*) malloc (sizeof(float) * batch_size * out_channels * size * size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(float) * batch_size * out_channels * size * size, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < * batch_size * out_channels * size * size; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    return 0;   
}