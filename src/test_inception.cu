# include "inception.h"
# include "cuda_runtime.h"
# include "loader.hpp"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;

const int batch_size = 4, in_channels = 3, size = 299;

int main() {
    double *input;

    Inception inc = load_weights_from_json("../data/inceptionV3.json", true);

    input = (double *) malloc (sizeof(double) * batch_size * in_channels * size * size);
    for (int i = 0; i < batch_size * in_channels * size * size; ++i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    cout << "cpu begin.\n";
    double *cpu_output = inc.cpu_forward(input, batch_size);
    cout << "cpu end.\n";

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_channels * size * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_channels * size * size, cudaMemcpyHostToDevice);
    cout << "gpu begin.\n";
    double *cuda_output = inc.gpu_forward(cuda_input, batch_size);
    cout << "gpu end.\n";
    int out_channels = inc.get_out_channels(), out_size = inc.get_out_size();
    int output_N = batch_size * out_channels * out_size * out_size;
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * output_N);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * output_N, cudaMemcpyDeviceToHost);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cout << "cudnn begin.\n";
    double *cudnn_output = inc.cudnn_forward(cudnn, cuda_input, batch_size);
    cout << "cudnn end.\n";
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * output_N);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * output_N, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    double max_error_cudnn = 0.0;
    double max_error_cuda_cudnn = 0.0;
    double avg_output = 0.0;
    for (int i = 0; i < output_N; ++ i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
        max_error_cudnn = max(max_error_cudnn, fabs(cudnn_output_device[i] - cpu_output[i]));
        max_error_cuda_cudnn = max(max_error_cuda_cudnn, fabs(cudnn_output_device[i] - cuda_output_device[i]));
        avg_output = avg_output + cpu_output[i] / output_N;
    }
    cout << "Output Scale: " << avg_output << endl;
    cout << "Max Error (CUDA vs CPU) = " << max_error << endl;
    cout << "Max Error (CUDNN vs CPU) = " << max_error_cudnn << endl;
    cout << "Max Error (CUDA vs CUDNN) = " << max_error_cuda_cudnn << endl;
    if (max_error > 1e-5 || max_error_cudnn > 1e-5 || max_error_cuda_cudnn > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cudnnDestroy(cudnn);
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    cudaFree(cudnn_output);
    free(input);
    free(cpu_output);
    free(cuda_output_device);
    free(cudnn_output_device);
    return 0; 
}
