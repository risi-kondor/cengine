#include "Cengine_base.hpp"

//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
//#include "Factorial.hpp"
//#include "WignerMatrix.hpp"
#include "Cengine.hpp"

std::default_random_engine rndGen;
mutex Cengine::CoutLock::mx;

//vector<int> Cengine::factorial::fact;

Cengine::Cengine* Cengine_engine=new Cengine::Cengine();


#ifdef _WITH_CUDA
//__device__ __constant__ unsigned char cg_cmem[CG_CONST_MEM_SIZE];
#endif

