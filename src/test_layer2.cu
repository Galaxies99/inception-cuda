# include "layers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>
using namespace std;

const int batch_size = 4, in_channels = 192, size = 35;
InceptionLayer2 layer(in_channels, size, 32);

int main() {
    double *input;

    input = (double *) malloc (sizeof(double) * batch_size * in_channels * size * size);
    for (int i = 0; i < batch_size * in_channels * size * size; ++i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    cout << "cpu begin.\n";
    double *cpu_output = layer.cpu_forward(input, batch_size);
    cout << "cpu end.\n";

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_channels * size * size, cudaMemcpyHostToDevice);
    cout << "gpu begin.\n";
    double *cuda_output = layer.gpu_forward(cuda_input, batch_size);
    cout << "gpu end.\n";
    int out_channels = layer.get_out_channels(), out_size = layer.get_out_size();
    int output_N = batch_size * out_channels * out_size * out_size;
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * output_N);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * output_N, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (int i = 0; i < output_N; ++ i)
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    return 0; 
}
