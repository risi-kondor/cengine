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
#ifndef _CtensorBpack
#define _CtensorBpack

#include "CtensorB.hpp"
#include "CscalarBpack.hpp"
#include "Cnode.hpp"

namespace Cengine{

  class CtensorBpack{
  public:

    int N;
    Gdims dims; 
    int nbu=-1;
    mutable int device=1;
    int memsize=0; 

    float* arrg=nullptr;
    float* arrgc=nullptr;

    vector<CtensorB*> pack;
    mutable float** parr=nullptr;
    mutable float** parrc=nullptr;
    mutable bool parr_valid=false;

    CtensorBpack(){}

    CtensorBpack(const int _N, const Gdims& _dims, const int _nbu=-1, const int dev=1):
      N(_N), dims(_dims), nbu(_nbu), device(dev){
      CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
      CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
      parr_valid=true;
    }

    CtensorBpack(const int _N, const CtensorB& x):
      CtensorBpack(_N,x.dims,x.nbu,x.device){
      memsize=x.memsize;
    }

    CtensorBpack(const CtensorB& x):
      dims(x.dims), nbu(x.nbu), device(x.device), memsize(x.memsize){}
    
    CtensorBpack(const vector<CtensorB*>& v):
      CtensorBpack(v.size(),*v[0]){
      memsize=v[0]->memsize;
      pack.resize(N);
      for(int i=0; i<N; i++){
	pack[i]=v[i];
	pack[i]->to_device(device);
      }
      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=pack[i]->arrg;
	arrc[i]=pack[i]->arrgc;
      }
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  
    }

    CtensorBpack(const vector<Cnode*>& v, const int s):
      CtensorBpack(v.size(),*dynamic_cast<CtensorB*>(v[0]->op->inputs[s]->obj)){
      pack.resize(N);
      for(int i=0; i<N; i++){
	pack[i]=dynamic_cast<CtensorB*>(v[i]->op->inputs[s]->obj);
	pack[i]->to_device(device);
      }
      memsize=pack[0]->memsize;
      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=pack[i]->arrg;
	arrc[i]=pack[i]->arrgc;
      }
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  
    }

    ~CtensorBpack(){ // what about arrg? memory leak?
      if(parr) CUDA_SAFE(cudaFree(parr));
      if(parrc) CUDA_SAFE(cudaFree(parrc));
    }

    CtensorBpack(const CtensorBpack& x)=delete;
    CtensorBpack& operator=(const CtensorBpack& x)=delete;

    CtensorBpack& operator=(CtensorBpack&& x){
      if(parr) CUDA_SAFE(cudaFree(parr));
      if(parrc) CUDA_SAFE(cudaFree(parrc));
      N=x.N; dims=x.dims; nbu=x.nbu; device=x.device; 
      parr=x.parr; x.parr=nullptr;
      parrc=x.parrc; x.parrc=nullptr;
      pack=std::move(x.pack);
      memsize=x.memsize;
      return *this; 
    }


  public: // -------------------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }


    Gdims get_dims() const{
      return dims; 
    }


    float** get_parr() const{
      if(!parr || !parr_valid) renew_parr();
      return parr;
    }


    float** get_parrc() const{
      if(!parrc || !parr_valid) renew_parr();
      return parrc;
    }


    void renew_parr() const{
      if(parr) CUDA_SAFE(cudaFree(parr)); parr=nullptr;
      if(parrc) CUDA_SAFE(cudaFree(parrc)); parrc=nullptr;

      to_device(1);
      const int N=pack.size(); 
      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=pack[i]->arrg;
	arrc[i]=pack[i]->arrgc;
      }
      
      CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  

      device=1;
      parr_valid=true;
    }


    void to_device(const device_id& _dev) const{
      assert(false); 
      if(_dev.id()==device) return; 
      parr_valid=false; 
      if(parr) CUDA_SAFE(cudaFree(parr)); parr=nullptr;
      if(parrc) CUDA_SAFE(cudaFree(parrc)); parrc=nullptr;
      device=_dev.id();
      for(auto p: pack)
	p->to_device(device);
    }



  public: // ---- Cumulative Operations ----------------------------------------------------------------------


#ifdef _WITH_CUDA
    void copy_cu(const CtensorBpack& x, const cudaStream_t& stream);
    void add_cu(const CtensorBpack& x, const cudaStream_t& stream);
    void sum_into_cu(const CtensorB& R, const cudaStream_t& stream);
#endif 



    void copy(const CtensorBpack& x){
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
    
    void add(const CtensorBpack& x){
      const int N=pack.size();
      //assert(x.pack.size()==N);

      if(device==0){
	for(int i=0; i<N; i++)
	  pack[i]->add(*x.pack[i]);
      }else{
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	add_cu(x,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
	//if(!parr_valid) renew_parr();
	//if(!x.parr_valid) x.renew_parr();
	//CUBLAS_SAFE(gemmBatched(genet_cublas,CUBLAS_OP_N,CUBLAS_OP_N);)
      }
    }

    void add_prod(const CscalarBpack& c, const CtensorBpack& A){
      CENGINE_UNIMPL();
    }


    void sum_into(CtensorB& R){
      //assert(temp);

      if(device==0){
	FCG_UNIMPL(); 
      }else{
#ifdef _WITH_CUDA
	R.to_device(1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	sum_into_cu(R,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	NOCUDA_ERROR;
#endif
      }

    }

    void add_inp_into(CscalarBpack& R, CtensorBpack& Y){
      CENGINE_UNIMPL(); 
    }
    
    #include "CtensorBpack_add_Mprod.hpp"


  };

}

#endif
    //bool temp=false; // eliminate these
    //float* temp_base;
    
    //CtensorBpack(const Gdims& _dims, const int _nbu=-1, const int dev=0):
    //dims(_dims), nbu(_nbu), device(dev){}

//public: // -------------------------------------------------------------------------------------------------

    /*
    CtensorBpack(const int N, const Gdims& _dims, const int _nbu, const fill_raw& dummy, const int dev=0):
      dims(_dims), nbu(_nbu), device(dev){
      assert(dev==1);
    }
    */

    /*
    CtensorBpack(const int _N, const CtensorB& model, const fill_raw& dummy, const int dev=1):
      dims(model.dims), nbu(model.nbu), device(dev), N(_N){
      assert(dev==1);
      int cst=model.cst;
      int memsize=model.memsize;
      float* temp_base;
      //int cst=roundup(model->cst*sizeof(float),128);
      //int memsize=roundup(model->memsize*sizeof(float),128);
      CUDA_SAFE(cudaMalloc((void **)&temp_base, memsize*N*sizeof(float)));

      float* arr[N]; 
      float* arrc[N]; 
      for(int i=0; i<N; i++){
	arr[i]=temp_base+i*memsize;
	arrc[i]=temp_base+i*memsize+cst;
      }
      
      CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
      CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
      CUDA_SAFE(cudaMemcpy(parr,arr,N*sizeof(float*),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(parrc,arrc,N*sizeof(float*),cudaMemcpyHostToDevice));  

      device=1;
      parr_valid=true;
      temp=true;
    }
    */
    /*
    void push_back(CtensorB* x){
      assert(x->dims==dims);
      assert(x->nbu==nbu);
      pack.push_back(x);
      parr_valid=false;
    }


    void push_back(CtensorB& x){
      assert(x.dims==dims);
      assert(x.nbu==nbu);
      pack.push_back(&x);
      parr_valid=false;
    }
    */

