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
#ifndef _CscalarBreducer
#define _CscalarBreducer

#include "CscalarBpack.hpp"

namespace Cengine{

  class CscalarBreducer: public CscalarBpack{
  public:

    CscalarB& target;

    CscalarBreducer(const int _N, CscalarB& _target):
      CscalarBpack(_N), target(_target){
    }


    ~CscalarBreducer(){
#ifdef _WITH_CUDA
	target.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	//reduce_cu(target,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	NOCUDA_ERROR;
#endif
	CUDA_SAFE(cudaMemcpy(&target.val,arrg,2*sizeof(float),cudaMemcpyDeviceToHost)); 
	CUDA_SAFE(cudaFree(arrg));
    }


  };

}

#endif
