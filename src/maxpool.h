# ifndef _LAYER_H_
# define _LAYER_H_
# include "cuda_runtime.h"
# include "utils.h"
# endif

void maxpooling_cpu(float*, const int, const int, float*, float*);
void meanpooling_cpu(float*, const int, const int, float*);

__global__ void maxpooling_forward(float*, const int, const int, float*, float*);
__global__ void meanpool_forward(float*, const int, const int, float*);