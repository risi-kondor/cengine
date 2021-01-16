#include "Cengine_base.hpp"
#include "CengineTraceback.hpp"

namespace Cengine{

#ifdef CENGINE_OBJ_COUNT
  atomic<int> Cnode_count; 
  atomic<int> Chandle_count; 
  atomic<int> Coperator_count; 
  atomic<int> RscalarB_count; 
  atomic<int> CscalarB_count; 
  atomic<int> CtensorB_count; 
  atomic<int> CmatrixB_count; 
  atomic<int> CtensorBarray_count; 
#endif 

#ifdef CENGINE_TRACEBACK_FLAG
  Cengine::CengineTraceback traceback;
#endif

}

#include "Cengine.hpp"

//std::default_random_engine rndGen;
mutex Cengine::CoutLock::mx;
Cengine::Cengine* cengine; //=new Cengine::Cengine();


#ifdef _WITH_CUDA
//__device__ __constant__ unsigned char cg_cmem[CG_CONST_MEM_SIZE];
#endif

#ifdef _WITH_CUBLAS
//#include <cublas_v2.h>
//cublasHandle_t Cengine_cublas;
//cublasCreate(&Cengine_cublas);
#endif 

#include "CengineSession.hpp"

namespace Cengine{

  void shutdown(){
    delete cengine;
  }


}


/*
#ifdef _WITH_CUDA
__device__ __constant__ unsigned char cg_cmem[CG_CONST_MEM_SIZE];
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
cublasHandle_t Cengine_cublas;
//cublasCreate(&Cengine_cublas);
#endif 
*/
