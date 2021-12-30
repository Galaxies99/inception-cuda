# include "fc.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>
# include <sys/time.h>
using namespace std;

int batch_size = 4, in_features = 2048, out_features = 1000;
FullyConnectedLayer fc(in_features, out_features);

int main() {
    double *input;
    input = (double*) malloc (sizeof(double) * batch_size * in_features);
    for (int i = 0; i < batch_size * in_features; ++i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    timeval start_, end_;
    float duration = 0;
    gettimeofday(&start_, 0);
    double *cpu_output = fc.cpu_forward(input, batch_size);
    gettimeofday(&end_, 0);
    duration = (end_.tv_sec - start_.tv_sec) * 1e6 + (end_.tv_usec - start_.tv_usec);
    cout << "Time of CPU: " << duration / 1000 << " ms.\n";

    dim3 grid(32, batch_size);
    dim3 block(32);

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_features);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_features, cudaMemcpyHostToDevice);

    float Onetime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    double *cuda_output = fc.basic_forward(grid, block, cuda_input, batch_size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * batch_size * out_features);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * batch_size * out_features, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);
    cout << "Time of Basic Forward: " << Onetime << endl;

    cudaEventRecord(start, 0);
    double *cuda_output_opt = fc.opt_forward(grid, block, cuda_input, batch_size);
    double *cuda_output_device_opt;
    cuda_output_device_opt = (double*) malloc (sizeof(double) * batch_size * out_features);
    cudaMemcpy(cuda_output_device_opt, cuda_output_opt, sizeof(double) * batch_size * out_features, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);
    cout << "Time of Optimized Forward: " << Onetime << endl;
    
    cudaEventRecord(start, 0);
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    double *cudnn_output = fc.cudnn_forward(cudnn, cuda_input, batch_size);
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * batch_size * out_features);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * batch_size * out_features, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);
    cout << "Time of CUDNN: " << Onetime << endl;

    double max_error = 0.0;
    double max_error_opt = 0.0;
    double max_error_cudnn = 0.0;
    for (int i = 0; i < batch_size * out_features; ++i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
        max_error_opt = max(max_error_opt, fabs(cuda_output_device_opt[i] - cpu_output[i]));
        max_error_cudnn = max(max_error_cudnn, fabs(cudnn_output_device[i] - cpu_output[i]));
    }
    cout << "Max Error (CUDA vs CPU) = " << max_error << endl;
    cout << "Max Error (CUDAopt vs CPU) = " << max_error_opt << endl;
    cout << "Max Error (CUDNN vs CPU) = " << max_error_cudnn << endl;
    if (max_error > 1e-5 || max_error_cudnn > 1e-5 || max_error_opt > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    
    cudnnDestroy(cudnn);
    cudaFree(cuda_input);
    cudaFree(cuda_output);
    cudaFree(cuda_output_opt);
    cudaFree(cudnn_output);
    free(input);
    free(cpu_output);
    free(cuda_output_device);
    free(cuda_output_device_opt);
    free(cudnn_output_device);
    return 0;
}
