#ifndef _CtensorBpack_cu
#define _CtensorBpack_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "CtensorBpack.hpp"

//extern GEnet::SO3_CGbank SO3_cgbank;
//extern __device__ __constant__ unsigned char cg_cmem[];



__global__ void CtensorBpack_copy_kernel(float** parr, float** parrc, float** x_parr, float** x_parrc, int warps){

  const int b=blockIdx.x;
  const int t=threadIdx.x;

  float* ptr=parr[b];
  float* ptrc=parrc[b];
  float* xptr=x_parr[b];
  float* xptrc=x_parrc[b];

  for(int i=0; i<warps; i++)
    ptr[i*32+t]=xptr[i*32+t];

  for(int i=0; i<warps; i++)
    ptrc[i*32+t]=xptrc[i*32+t];

}


__global__ void CtensorBpack_copy_kernel(float* arrg, float* arrgc, float* arrg_src, float* arrgc_src, int warps){

  const int t=threadIdx.x;

  for(int i=0; i<warps; i++)
    arrg[i*32+t]=arrg_src[i*32+t];

  for(int i=0; i<warps; i++)
    arrgc[i*32+t]=arrgc_src[i*32+t];

}


__global__ void CtensorBpack_add_kernel(float** parr, float** parrc, float** x_parr, float** x_parrc, int warps){

  const int b=blockIdx.x;
  const int t=threadIdx.x;

  float* ptr=parr[b];
  float* ptrc=parrc[b];
  float* xptr=x_parr[b];
  float* xptrc=x_parrc[b];

  for(int i=0; i<warps; i++)
    ptr[i*32+t]+=xptr[i*32+t];

  for(int i=0; i<warps; i++)
    ptrc[i*32+t]+=xptrc[i*32+t];

}


// deprecated
/*
__global__ void CtensorBpack_reduce_kernel(float* arrg, float* arrgc, float** parr, float** parrc, const int warps, int d, const int N){

  const int b=blockIdx.x;
  const int t=threadIdx.x;

  if(b<N-d){
      float* ptr=parr[b];
      float* ptrc=parrc[b];
      float* xptr=parr[b+d];
      float* xptrc=parrc[b+d];
      for(int i=0; i<warps; i++)
	ptr[i*32+t]=xptr[i*32+t];
      for(int i=0; i<warps; i++)
	ptrc[i*32+t]=xptrc[i*32+t];
  }
  d/=2;

  while(d>0){
    if(b<d){
      float* ptr=parr[b];
      float* ptrc=parrc[b];
      float* xptr=parr[b+d];
      float* xptrc=parrc[b+d];
      for(int i=0; i<warps; i++)
	ptr[i*32+t]=xptr[i*32+t];
      for(int i=0; i<warps; i++)
	ptrc[i*32+t]=xptrc[i*32+t];
    }
    d/=2;
  }

  if(b==0){
    float* xptr=parr[0];
    float* xptrc=parrc[0];
    for(int i=0; i<warps; i++)
      arrg[i*32+t]=xptr[i*32+t];
    for(int i=0; i<warps; i++)
      arrgc[i*32+t]=xptrc[i*32+t];
  }

}
*/

__global__ void CtensorBpack_reduce_kernel(float** parr, float** parrc, const int warps, int d, const int N){

  const int b=blockIdx.x;
  const int t=threadIdx.x;

  float* ptr=parr[b];
  float* ptrc=parrc[b];
  float* xptr=parr[b+d];
  float* xptrc=parrc[b+d];
  for(int i=0; i<warps; i++)
    ptr[i*32+t]+=xptr[i*32+t];
  for(int i=0; i<warps; i++)
    ptrc[i*32+t]+=xptrc[i*32+t];

}


namespace Cengine{


  void CtensorBpack::copy_cu(const CtensorBpack& x, const cudaStream_t& stream){
    if(!parr_valid) renew_parr();
    if(!x.parr_valid) x.renew_parr();

    // x.get_parr();
    int cst=x.pack[0]->cst;

    CtensorBpack_copy_kernel<<<N,32,0,stream>>>
      (parr,parrc,x.parr,x.parrc,cst/32);
 
  }


  void CtensorBpack::add_cu(const CtensorBpack& x, const cudaStream_t& stream){
    if(!parr_valid) renew_parr();
    if(!x.parr_valid) x.renew_parr();

    // x.get_parr();
    int cst=x.pack[0]->cst;

    CtensorBpack_add_kernel<<<N,32,0,stream>>>
      (parr,parrc,x.parr,x.parrc,cst/32);
 
  }

  
  void CtensorBpack::sum_into_cu(const CtensorB& R, const cudaStream_t& stream){

    const int cst=R.cst;
    int d=1;
    while(d<N) d*=2;
    d/=2;

    CtensorBpack_reduce_kernel<<<N-d,32,0,stream>>>
      (parr,parrc,cst/32,d,N);
    d/=2;

    while(d>0){
      CtensorBpack_reduce_kernel<<<d,32,0,stream>>>
	(parr,parrc,cst/32,d,N);
      d/=2;
    }

    CtensorBpack_copy_kernel<<<1,32,0,stream>>>
      (R.arrg,R.arrgc,arrg,arrgc,cst/32);

  }

}

#endif 
