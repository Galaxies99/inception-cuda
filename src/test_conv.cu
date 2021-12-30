# include "conv.h"
# include "cuda_runtime.h"
# include <cudnn.h>
# include <stdlib.h>
# include <stdio.h>
# include <iostream>
# include <sys/time.h>

using namespace std;

int batch_size = 4, in_channels = 16, out_channels = 16, in_size_r = 128, in_size_c = 128, kernel_r = 3, kernel_c = 3, stride_r = 1, stride_c = 1, padding_r = 1, padding_c = 1;
ConvolutionLayer conv(in_channels, out_channels, in_size_r, in_size_c, kernel_r, kernel_c, stride_r, stride_c, padding_r, padding_c);

int main() {
    double *input;
    int out_size_r, out_size_c;
    conv.get_output_size(out_size_r, out_size_c);

    input = (double*) malloc (sizeof(double) * batch_size * in_channels * in_size_r * in_size_c);
    for (int i = 0; i < batch_size * in_channels * in_size_r * in_size_c; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;
    
    timeval start_, end_;
    float duration = 0;
    gettimeofday(&start_, 0);
    double *cpu_output = conv.cpu_forward(input, batch_size);
    gettimeofday(&end_, 0);
    duration = (end_.tv_sec - start_.tv_sec) * 1e6 + (end_.tv_usec - start_.tv_usec);
    cout << "Time of CPU: " << duration / 1000 << " ms.\n";

    dim3 grid(8, batch_size);
    dim3 block(32);

    float Onetime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * batch_size * in_channels * in_size_r * in_size_c);
    cudaMemcpy(cuda_input, input, sizeof(double) * batch_size * in_channels * in_size_r * in_size_c, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    double *cuda_output = conv.basic_forward(grid, block, cuda_input, batch_size);    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);
    cout << "Time of Basic Forward: " << Onetime << " ms.\n";
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * batch_size * out_channels * out_size_r * out_size_c);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * batch_size * out_channels * out_size_r * out_size_c, cudaMemcpyDeviceToHost);


    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    conv.cudnn_forward(cudnn, cuda_input, batch_size);
    cudaEventRecord(start, 0);
    double *cudnn_output = conv.cudnn_forward(cudnn, cuda_input, batch_size);    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);
    cout << "Time of CUDNN Forward: " << Onetime << " ms.\n";
    double *cudnn_output_device;
    cudnn_output_device = (double*) malloc (sizeof(double) * batch_size * out_channels * out_size_r * out_size_c);
    cudaMemcpy(cudnn_output_device, cudnn_output, sizeof(double) * batch_size * out_channels * out_size_r * out_size_c, cudaMemcpyDeviceToHost);    
    cudaDeviceSynchronize();
    

    double max_error = 0.0;
    double max_error_cudnn = 0.0;    
    for (int i = 0; i < batch_size * out_channels * out_size_r * out_size_c; ++ i) {
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
        max_error_cudnn = max(max_error_cudnn, fabs(cudnn_output_device[i] - cpu_output[i]));
    }
    cout << "Max Error (CUDA vs CPU) = " << max_error << endl;
    cout << "Max Error (CUDNN vs CPU) = " << max_error_cudnn << endl;
    if (max_error > 1e-5 || max_error_cudnn > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cout << "\nTesting im2col ===>\n";
    double *cpu_basic_output = conv.cpu_basic_forward(input, batch_size);
    double *cpu_im2col_output = conv.cpu_im2col_forward(input, batch_size);
    max_error = 0.0;
    for (int i = 0; i < batch_size * out_channels * out_size_r * out_size_c; ++ i) 
        max_error = max(max_error, fabs(cpu_basic_output[i] - cpu_im2col_output[i]));
    cout << "Max Error (basic vs im2col) = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    free(cpu_im2col_output);

    cudaEventRecord(start, 0);
    double *cuda_im2col_output = conv.implicit_im2col_forward(cuda_input, batch_size);
    cudaDeviceSynchronize();    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);
    cout << "Time of im2col Forward: " << Onetime << " ms.\n";
    double *cuda_im2col_output_device;
    cuda_im2col_output_device = (double*) malloc (sizeof(double) * batch_size * out_channels * out_size_r * out_size_c);
    cudaMemcpy(cuda_im2col_output_device, cuda_im2col_output, sizeof(double) * batch_size * out_channels * out_size_r * out_size_c, cudaMemcpyDeviceToHost);    
    max_error = 0.0;
    for (int i = 0; i < batch_size * out_channels * out_size_r * out_size_c; ++ i)
        max_error = max(max_error, fabs(cudnn_output_device[i] - cuda_im2col_output_device[i]));
    cout << "Max Error (cudnn vs cuda im2col) = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;
    free(cuda_im2col_output_device);
    free(cpu_basic_output);
    cudaFree(cuda_im2col_output);

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