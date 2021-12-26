# ifndef _CUDNN_UTILS_H_
# define _CUDNN_UTILS_H_

# include <cudnn.h>
# include <cudnn_v8.h>
# include <iostream>

# define checkCUDNN(expression) \
  { \
    cudnnStatus_t status = (expression); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std :: cerr << "Error on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std :: endl; \
        std :: exit(EXIT_FAILURE); \
    } \
  }

# endif