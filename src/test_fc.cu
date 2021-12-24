# include "fc.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>
using namespace std;

int batch_size = 4, in_features = 10, out_features = 6;
FullyConnectedLayer fc(in_features, out_features);

int main() {
    float *input;
    input = (float*) malloc (sizeof(float) * batch_size * in_features);
    for (int i = 0; i < batch_size * in_features; ++i)
        input[i] = (float) (rand() % 32768) / 32768.0;
    
    float *cpu_output = fc.cpu_forward(input, batch_size);

    dim3 grid(2, batch_size);
    dim3 block(4);

    float *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(float) * batch_size * in_features);
    cudaMemcpy(cuda_input, input, sizeof(float) * batch_size * in_features, cudaMemcpyHostToDevice);
    float *cuda_output = fc.basic_forward(grid, block, cuda_input, batch_size);
    float *cuda_output_device;
    cuda_output_device = (float*) malloc (sizeof(float) * batch_size * out_features);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(float) * batch_size * out_features, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < batch_size * out_features; ++i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
        cout << cuda_output_device[i] << "        " << cpu_output[i] << "        " << max_error << endl;
    }
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    return 0;
}
