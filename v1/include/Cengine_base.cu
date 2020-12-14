#include "Cengine_base.hpp"

__device__ __constant__ unsigned char cg_cmem[CG_CONST_MEM_SIZE];

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
cublasHandle_t Cengine_cublas;
//cublasCreate(&Cengine_cublas);
#endif 
