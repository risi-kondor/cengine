/*
 * This file is part of Cengine, an asynchronous C++/CUDA compute engine. 
 *  
 * Copyright (c) 2020- Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */
#ifndef _CscalarBpack
#define _CscalarBpack

#include "CscalarB.hpp"
#include "Cnode.hpp"

namespace Cengine{

  class CscalarBpack{
  public:

    int nbu=-1;
    int N;

    vector<CscalarB*> pack;
    mutable float** arrg=nullptr;

    CscalarBpack(const int _N, const int _nbu=-1): 
      N(_N), nbu(_nbu){
      assert(nbu==-1);
      CUDA_SAFE(cudaMalloc((void **)&arrg, 2*N*sizeof(float)));
    }

    CscalarBpack(const vector<Cnode*>& nodes, const int s):
      CscalarBpack(nodes.size()){
      pack.resize(N);
      float arr[2*N];
      for(int i=0; i<N; i++){
	CscalarB* x=dynamic_cast<CscalarB*>(nodes[i]->op->inputs[s]->obj);
	pack[i]=x;
	arr[2*i]=std::real(x->val);
	arr[2*i+1]=std::imag(x->val);
      }
      CUDA_SAFE(cudaMemcpy(arrg,arr,2*N*sizeof(float),cudaMemcpyHostToDevice));
    }

    ~CscalarBpack(){
      if(!arrg) return; 
      float arr[2*N];
      CUDA_SAFE(cudaMemcpy(arr,arrg,2*N*sizeof(float),cudaMemcpyDeviceToHost)); 
      for(int i=0; i<N; i++){
	pack[i]->val=complex<float>(arr[2*i],arr[2*i+1]);
      }
      CUDA_SAFE(cudaFree(arrg));
    }


    CscalarBpack(const CscalarBpack& x)=delete;
    CscalarBpack& operator=(const CscalarBpack& x)=delete;


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


#ifdef _WITH_CUDA
    //void copy_cu(const CscalarBpack& x, const cudaStream_t& stream);
    void reduce_cu(const cudaStream_t& stream);
#endif 

    /*
    void copy(const CscalarBpack& x){
      assert(temp);
      assert(x.pack.size()==N);

      if(device==0){
	FCG_UNIMPL(); 
      }else{
#ifdef _WITH_CUDA
	x.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	copy_cu(x,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	NOCUDA_ERROR;
#endif
      }

    }
    
    void add(const CscalarBpack& x){
      assert(!temp);
      const int N=pack.size();
      assert(x.pack.size()==N);

      if(device==0){
	for(int i=0; i<N; i++)
	  pack[i]->add(*x.pack[i]);
      }else{
	//if(!parr_valid) renew_parr();
	//if(!x.parr_valid) x.renew_parr();
	//CUBLAS_SAFE(gemmBatched(genet_cublas,CUBLAS_OP_N,CUBLAS_OP_N);)
      }

    }
    */


    void reduce(){
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      //sum_into_cu(R,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      NOCUDA_ERROR;
#endif
    }
    


  };

}

#endif
