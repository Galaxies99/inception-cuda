# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <cuda.h>
# include <cuda_runtime.h>
# include "inception.h"
# include "loader.hpp"

# define INPUTSHAPE 3 * 299 * 299
# define OUTPUTSHAPE 1000
# define TESTNUM 10
# define ITERNUM 500
double inputArr[TESTNUM][INPUTSHAPE];
double benchOutArr[TESTNUM][OUTPUTSHAPE];

void readInput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++)
            fscanf(fp, "%lf", &inputArr[i][j]);
}

void readOutput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%lf", &benchOutArr[i][j]);
}

void checkOutput(double *out1, double *out2)
{
    double maxDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7f\n", maxDiff);
    }
}


Inception initModel() {
    Inception net = load_weights_from_json("../data/inceptionV3.json", true);
    return net;
}

void inference(Inception &net, double *input, double *output) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    double *cuda_input;
    cudaMalloc((void **)&cuda_input, sizeof(double) * INPUTSHAPE);
    cudaMemcpy(cuda_input, input, sizeof(double) * INPUTSHAPE, cudaMemcpyHostToDevice);
    double *cuda_output = net.cudnn_forward(cudnn, cuda_input, 1);
    cudaMemcpy(output, cuda_output, sizeof(double) * OUTPUTSHAPE, cudaMemcpyDeviceToHost);
    cudnnDestroy(cudnn);
}


int main()
{
    Inception net = initModel();    // 读取网络权重
    
    readInput("/models/inceptionInput.txt");   // 读取输入
    readOutput("/models/inceptionOutput.txt"); // 读取标准输出
    float sumTime = 0;
    for (int i = 0; i < TESTNUM; i++)
    {
        double inferOut[1000];
        for (int j = 0; j < ITERNUM; j++)
        {
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            inference(net, inputArr[i], inferOut);   // 执行Inference
            
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            
            sumTime += Onetime; // 累加单次推理消耗时间
        }
        checkOutput(benchOutArr[i], inferOut);
    }
    printf("Average Time is: %f\n", (sumTime / TESTNUM / ITERNUM));
}