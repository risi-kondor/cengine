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
#endif 

#ifdef CENGINE_TRACEBACK_FLAG
  Cengine::CengineTraceback traceback;
#endif

}

#include "Cengine.hpp"
#include "CtensorB_add_Mprod_ops.hpp"
#include "CtensorB_rbatched_ops.hpp"

std::default_random_engine rndGen;
mutex Cengine::CoutLock::mx;
Cengine::Cengine* Cengine_engine=new Cengine::Cengine();

int Cengine::ctensor_add_op::_batcher_id=0; 
int Cengine::ctensor_add_op::_rbatcher_id=0; 

template<> int Cengine::ctensor_add_Mprod_op<0,0>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<0,1>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<0,2>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<0,3>::_batcher_id=0; 

template<> int Cengine::ctensor_add_Mprod_op<1,0>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<1,1>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<1,2>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<1,3>::_batcher_id=0; 

template<> int Cengine::ctensor_add_Mprod_op<2,0>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<2,1>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<2,2>::_batcher_id=0; 
template<> int Cengine::ctensor_add_Mprod_op<2,3>::_batcher_id=0; 

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
    delete Cengine_engine;
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
