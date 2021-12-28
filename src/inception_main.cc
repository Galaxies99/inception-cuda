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
# define ITERNUM 1
double inputArr[TESTNUM][INPUTSHAPE];
double benchOutArr[TESTNUM][OUTPUTSHAPE];

void readInput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++) 
            fscanf(fp, "%lf", &inputArr[i][j]);
    fclose(fp);
}

void readOutput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%lf", &benchOutArr[i][j]);
    fclose(fp);
}

void checkOutput(double *out1, double *out2)
{
    double maxDiff = 0;
    double meanDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
        meanDiff += (fabs(out1[i] - out2[i])) / OUTPUTSHAPE;
        if (fabs(out1[i] - out2[i]) > 5) {
            printf("i = %d, expect %.7lf, found %.7lf\n", i, out1[i], out2[i]);
        }
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7lf, MeanDiff is %.7lf\n", maxDiff, meanDiff);
    }
}


Inception initModel() {
    return load_weights_from_json("../data/inceptionV3.json", true);
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
    cudaFree(cuda_input);
    cudaFree(cuda_output);
}


int main()
{
    Inception net = initModel();
    
    readInput("../data/inceptionInput.txt"); 
    readOutput("../data/inceptionOutput.txt"); 
    float sumTime = 0;
    for (int i = 0; i < TESTNUM; i++)
    {
        double inferOut[OUTPUTSHAPE];
        for (int j = 0; j < ITERNUM; j++)
        {
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            inference(net, inputArr[i], inferOut);
            
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            
            sumTime += Onetime;
        }
        checkOutput(benchOutArr[i], inferOut);
        return 0;
    }
    printf("Average Time is: %f\n", (sumTime / TESTNUM / ITERNUM));
}