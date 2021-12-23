# ifndef _UTILS_H
# define _UTILS_H
# include "cuda_runtime.h"
# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# endif

float init_rand(void);
__device__ float activation_sigmoid(float);
__device__ float activation_relu(float);