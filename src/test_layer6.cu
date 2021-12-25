# include "layers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

const int batch_size = 4, in_channels = 2048, size = 8, out_channels = 2048;
InceptionLayer6 layer(in_channels, size);

int main() {
    double *input;

    input = (double*) malloc (sizeof(double) * batch_size * in_channels * size * size);
    for (int i = 0; i < batch_size * in_channels * size * size; ++ i)
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
    int out_size = batch_size * out_channels * size * size;
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * out_size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * out_size, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (int i = 0; i < out_size; ++ i) {
        if (fabs(cuda_output_device[i] - cpu_output[i]) > 2000.0) {
            int id = i;
            int c = id % size;
            int r = (id /= size) % size;
            int ch = (id /= size) % out_channels;
            int b = (id /= out_channels) % batch_size;
            cout << b << ' ' << ch << ' ' << r << ' ' << c << ' ' << "cpu: " << cpu_output[i] << ", gpu:" << cuda_output_device[i] << endl;
        }
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    }
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    return 0;   
}