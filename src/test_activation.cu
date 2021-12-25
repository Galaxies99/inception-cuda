# include "activation.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;


int main() {
    double *input;
    const int size = 1024;

    input = (double*) malloc (sizeof(double) * size);
    for (int i = 0; i < size; ++ i)
        input[i] = (double) (rand() % 32768) / 32768.0;

    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * size);
    cudaMemcpy(cuda_input, input, sizeof(double) * size, cudaMemcpyHostToDevice);
    
    double *cpu_output = cpu_relu(input, size);

    dim3 grid(4);
    dim3 block(4);

    double *cuda_output = relu(grid, block, cuda_input, size);
    double *cuda_output_device;
    cuda_output_device = (double*) malloc (sizeof(double) * size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(double) * size, cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cout << endl << "Inplace testing: \n";

    double *cuda_input_device;
    cuda_input_device = (double*) malloc (sizeof(double) * size);
    cudaMemcpy(cuda_input_device, cuda_input, sizeof(double) * size, cudaMemcpyDeviceToHost);

    max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_input_device[i] - input[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    return 0;   
}