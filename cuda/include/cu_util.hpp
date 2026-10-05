#ifndef CU_UTIL_HPP
#define CU_UTIL_HPP

#include "cuda_runtime.h"

#define HANDLE_ERROR(api) \
{\
  cudaError_t res; \
  if ((res = api) != cudaSuccess)\
  {\
    std::cout << "res: " << res << ", line: " << __FILE__ << ", " << __LINE__ << std::endl; \
    throw std::runtime_error("cudaMalloc error");\
  }\
}
#endif