# include "opers.h"
# include "cuda_runtime.h"
# include <stdlib.h>
# include <stdio.h>
# include <iostream>

using namespace std;


int main() {
    float *input;
    const int size = 1024;
    const float alpha = 0.2, beta = 0.3;

    input = (float*) malloc (sizeof(float) * size);
    for (int i = 0; i < size; ++ i)
        input[i] = (float) (rand() % 32768) / 32768.0;

    float *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(float) * size);
    cudaMemcpy(cuda_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    float *cpu_output = cpu_linear_transform(input, size, alpha, beta);

    dim3 grid(4);
    dim3 block(4);

    float *cuda_output = linear_transform(grid, block, cuda_input, size, alpha, beta);
    float *cuda_output_device;
    cuda_output_device = (float*) malloc (sizeof(float) * size);
    cudaMemcpy(cuda_output_device, cuda_output, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cout << "====> Linear Transform\n";
    float max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_output_device[i] - cpu_output[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    cout << endl << "Inplace testing: \n";

    float *cuda_input_device;
    cuda_input_device = (float*) malloc (sizeof(float) * size);
    cudaMemcpy(cuda_input_device, cuda_input, sizeof(float) * size, cudaMemcpyDeviceToHost);

    max_error = 0.0;
    for (int i = 0; i < size; ++ i) 
        max_error = max(max_error, fabs(cuda_input_device[i] - input[i]));
    cout << "Max Error = " << max_error << endl;
    if (max_error > 1e-5) cout << "Incorrect." << endl;
    else cout << "Correct." << endl;

    return 0;   
}