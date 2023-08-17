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
#ifndef _RFtensor
#define _RFtensor

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t Cengine_cublas;
#endif 

#include <fstream>

#include "Cengine_base.hpp"
#include "Gindex.hpp"
#include "Gdims.hpp"
#include "Gtensor.hpp"
//#include "RFtensorHelpers.hpp"

extern default_random_engine rndGen;


namespace Cengine{


  class RFtensor{
  public:

    int k;
    Gdims dims;

    vector<int> strides;
    int asize=0;
    int memsize=0;

    mutable float* arr=nullptr;
    mutable float* arrg=nullptr;

    bool is_view=false;
    bool is_contiguous=true;

    mutable int device=0;

    ~RFtensor(){
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const {return "Cengine::RFtensor";}


  public: // ---- Constructors -------------------------------------------------------------------------------


    RFtensor(){
      arr=nullptr;
    }

    RFtensor(const Gdims& _dims, const device_id& dev=0): 
      dims(_dims), strides(_dims.size()){
      make_strides();
      if(dev.id()==0){
	arr=new float[memsize];
	device=0;
      }
      if(dev.id()==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	device=1;
      }
    }

    RFtensor(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, 
      const int _memsize, const int _cst, const device_id& _dev=0):
      k(_k), dims(_dims), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst), device(_dev.id()){
      if(device==0){
	arr=new float[memsize];
      }
      if(device==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      }
    }
    
    void make_strides(){
      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      memsize=roundup(asize,32); 
    }

    void reallocate(const device_id& dev=0) const{
      if(dev.id()==0){
	arr=new float[memsize];
	device=0;
      }
      if(dev.id()==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	device=1;
      }
    }


  public: // ---- Filled constructors -------------------------------------------------------------------------


    RFtensor(const Gdims& _dims, const fill_raw& dummy, const device_id& dev=0): 
      RFtensor(_dims,dev){}
    
    RFtensor(const Gdims& _dims, const fill_zero& dummy, const device_id& dev=0): 
      RFtensor(_dims,dev) {
      if(dev.id()==0) std::fill(arr,arr+memsize,0);
      if(dev.id()==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    RFtensor(const Gdims& _dims, const fill_ones& dummy, const device_id& dev=0): 
      RFtensor(_dims,dev) {
      if(dev.id()==0){
	std::fill(arr,arr+asize,1);
      }
      if(dev.id()==1){
	CUDA_SAFE(cudaMemset(arrg,1,asize*sizeof(float)));
      }
    }

    RFtensor(const Gdims& _dims, const fill_identity& dummy, const device_id& dev=0): 
      RFtensor(_dims){
      assert(dims[k-1]==dims[k-2]);
      std::fill(arr,arr+memsize,0);
      for(int i=0; i<dims[k-1]; i++)
	arr[i*(strides[k-2]+1)]=1;
      to_device(dev);
    }

    RFtensor(const Gdims& _dims, const fill_gaussian& dummy, const device_id& dev=0):
      RFtensor(_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      to_device(dev);
    }

    RFtensor(const Gdims& _dims, const fill_sequential& dummy, const device_id& dev=0):
      RFtensor(_dims,fill::zero){
      for(int i=0; i<asize; i++) arr[i]=i;
      to_device(dev);
    }

    RFtensor(const Gdims& _dims, const fill_const<float>& dummy, const device_id& dev=0):
      RFtensor(_dims){
      for(int i=0; i<asize; i++) arr[i]=dummy.p;
      to_device(dev);
    }


  public: // ---- Copying -------------------------------------------------------------------------------------


    RFtensor(const RFtensor& x): 
      RFtensor(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.device){
      COPY_WARNING;
      if(device==0) std::copy(x.arr,x.arr+memsize,arr);
      if(device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }
        
    RFtensor(const RFtensor& x, const nowarn_flag& dummy): 
      RFtensor(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.device){
      if(device==0) std::copy(x.arr,x.arr+memsize,arr);
      if(device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
    }

    RFtensor(const RFtensor& x, const device_id& dev): 
      RFtensor(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,dev){
      if(device==0){
	if(x.device==0) std::copy(x.arr,x.arr+memsize,arr);
	if(x.device==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost)); 
      }
      if(device==1){
	if(x.device==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
	if(x.device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }
    
    RFtensor(RFtensor&& x){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize;
      memsize=x.memsize; 
      arr=x.arr; x.arr=nullptr; arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
      is_contiguous=x.is_contiguous;
      device=x.device;
    }
    
    RFtensor& operator=(const RFtensor& x){
      memsize=x.memsize; cst=x.cst; 
      if(!is_view) delete arr;
      if(!is_view && arrg) cudaFree(arrg); 
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; device=x.device;
      if(device==0){
	arr=new float[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(device==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }

    RFtensor& operator=(RFtensor&& x){
      memsize=x.memsize; cst=x.cst; 
      if(!is_view && arr) delete arr;
      if(!is_view && arrg) CUDA_SAFE(cudaFree(arrg));
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; device=x.device; 
      arr=x.arr; x.arr=nullptr; arrg=x.arrg; x.arrg=nullptr; 
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Transporters ------------------------------------------------------------------------------
 

    const CFtensor& to_device(const device_id& dev) const{
      const int _dev=dev.id();
      if(_dev==0){
 	if(device==0) return *this;
 	delete[] arr;
	reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<CFtensor*>(this)->arrg=nullptr;
	device=0;
	return *this;
      }
      if(_dev>0){
	if(device==_dev) return *this;
	if(arrg) cudaFree(arrg);
	reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<CFtensor*>(this)->arr=nullptr;
	device=_dev;
	return *this;
      }
      return *this;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    template<typename TYPE>
    RFtensor(const Gtensor<TYPE>& x): 
      RFtensor(x.dims,fill::raw){
      int dev=x.device; 
      x.to_device(0);
      for(int i=0; i<asize; i++){
	arr[i]=x.arr[i];
      }
      to_device(dev);
      x.to_device(dev);
    }
    
    template<typename TYPE>
    operator Gtensor<TYPE>(){
      Gtensor<TYPE> R(dims,fill::raw);
      to_device(0);
      for(int i=0; i<asize; i++)
	R.arr[i]=arr[i];
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size(const int i) const{
      return dims[i];
    }

    int combined_size(const int a, const int b) const{
      assert(b<=k);
      assert(a<=b);
      if(a>0) return (strides[a-1])/(strides[b-1]);
      if(b>0) return asize/strides[b-1];
      return 1; 
    }

    complex<float> operator()(const Gindex& ix) const{
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }

    complex<float> get(const Gindex& ix) const{
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }
    
    RFtensor& set(const Gindex& ix, const float v){
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=v;
      return *this;
    }

    RFtensor& inc(const Gindex& ix, const float v){
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]+=v;
      return *this;
    }

    RFtensor& set(const float v){
      FCG_CPUONLY();
      if(device==0){
	std::fill(arr,arr+asize,v);
      }
      if(device==1){
      }
      return *this;
    }	

    #include "RFtensor_access.hpp"


  public: // ---- Elementwise Operations --------------------------------------------------------------------


    void zero(){
      if(device==0) std::fill(arr,arr+memsize,0);
      if(device==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    void set(const fill_gaussian& dummy){
      const int _dev=device;
      to_device(0);
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      to_device(_dev);
    }


    bool operator==(const RFtensor& x) const{
      if(x.asize!=asize) return false; 
      pullin2<RFtensor> p(*this,x);
      for(int i=0; i<asize; i++)
	if(arr[i]!=x.arr[i]) return false;
      return true;
    }


  public: // ---- Non in-place operations --------------------------------------------------------------------


    RFtensor plus(const RFtensor& x) const{
      assert(asize==x.asize);
      x.to_device(device);
      if(device==0){
	RFtensor R(dims,fill::raw,device);
	for(int i=0; i<asize; i++) R.arr[i]=arr[i]+x.arr[i];
	return R;
      }
      RFtensor R(*this,nowarn);
      const float alpha = 1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, R.arrg, 1));
      return R;
    }


    RFtensor minus(const RFtensor& x) const{
      assert(asize==x.asize);
      x.to_device(device);
      if(device==0){
	RFtensor R(dims,fill::raw,device);
	for(int i=0; i<asize; i++) R.arr[i]=arr[i]-x.arr[i];
	return R;
      }
      RFtensor R(*this,nowarn);
      const float alpha = -1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, R.arrg, 1));
      return R;
    }


    RFtensor times(const float c) const{
      if(device==0){
	RFtensor R(dims,fill::raw);
	for(int i=0; i<asize; i++) R.arr[i]=c*arr[i];
	return R;
      }
      RFtensor R(dims,fill::zero,device);
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, arrg, 1, R.arrg, 1));
      return R;
    }


    RFtensor elementwise_times(const RFtensor& x) const{
      FCG_CPUONLY();
      assert(asize==x.asize);
      RFtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]*x.arr[i];
      return R;
    }

    RFtensor elementwise_divide(const RFtensor& x) const{
      FCG_CPUONLY();
      assert(asize==x.asize);
      RFtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]/x.arr[i];
      return R;
    }

    RFtensor elementwise_pow(const float p, const float c=1.0) const{
      FCG_CPUONLY();
      RFtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++){
	R.arr[i]=c*pow(arr[i],p);
      }
      return R;
    }

    /*
    Gtensor abs() const{
      FCG_CPUONLY();
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=std::abs(arr[i]);
      return R;
    }
    */

    RFtensor transp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(device==0){
	RFtensor R({I,J},fill::raw);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]=arr[j*I+i];
	  }
	return R;
      }
      RFtensor R(dims,fill::zero,device);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
      return R;
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    float norm2() const{
      if(device==0){
      float t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*arr[i]
      return t;
      }
      float t=0;
#ifdef _WITH_CUBLAS
      cublasSdot(Cengine_cublas, asize, arrg, 1, arrg, 1, &tr);
#else
      NOCUDA_ERROR;
#endif       
      return t;
    }


    complex<float> inp(const RFtensor& x) const{
      assert(asize==x.asize);
      x.to_device(device);
      if(device==0){
	float tr=0; 
	float ti=0; 
	for(int i=0; i<asize; i++){
	  tr+=arr[i]*x.arr[i];
	}
	//{CoutLock lk; cout<<*this<<endl<<endl; cout<<"  "<<asize<<" "<<tr<<":"<<ti<<endl;}
	return complex<float>(tr,ti);
      }
      float a;
#ifdef _WITH_CUBLAS
      cublasSdot(Cengine_cublas, asize, arrg, 1, x.arrg, 1, &a);
#else
      NOCUDA_ERROR;
#endif       
      return a;
    }

    float diff2(const RFtensor& x) const{
      FCG_CPUONLY();
      assert(x.dims==dims);
      assert(x.asize==asize);
      float t=0;
      for(int i=0; i<asize; i++)
	t+=(arr[i]-x.arr[i])*(arr[i]-x.arr[i]);
      return t;
    }

    RFtensor odot(const RFtensor& x) const{
      FCG_CPUONLY();
      assert(dims==x.dims);
      RFtensor R(dims);
      for(int i=0; i<asize; i++){
	R.arr[i]=arr[i]*x.arr[i];
       }
      return R;
    }


    RFtensor divide_cols(const RFtensor& N) const{
      assert(k>=2);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      RFtensor R(dims,fill::zero);
      if(device==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      R.arr[offs+i*J+j]=arr[offs+i*J+j]/z;
	    }
	  }    
	}
      }else{
	FCG_UNIMPL(); 
      }
      return R;
    }


    RFtensor normalize_cols() const{
      Gdims ndims=dims.chunk(0,dims.size()-1);
      const int J=dims[dims.size()-1];
      const int I=asize/J;
      RFtensor R(dims);
      //RFtensor N(ndims);
      if(device==0){
	for(int i=0; i<I; i++){
	  float t=0;
	  for(int j=0; j<J; j++){
	    t+=R.arr[i*J+j]*R.arr[i*J+j];
	  }
	  float z=sqrt(t);
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]/=z;
	  }    
	  //N.arr[i]=z;
	  //N.arrc[i]=0;
	}
      }else{
	FCG_UNIMPL(); 
      }
      return R;
    }


    void add_normalize_cols_back(const RFtensor& g, const RFtensor& x){
      Gdims ndims=dims.chunk(0,dims.size()-1);
      const int J=dims[dims.size()-1];
      const int I=asize/J;
      RFtensor R(dims);
      //RFtensor N(ndims);
      if(device==0){
	for(int i=0; i<I; i++){
	  float t=0;
	  for(int j=0; j<J; j++){
	    t+=R.arr[i*J+j]*R.arr[i*J+j];
	  }
	  float z=sqrt(tr+ti);
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]/=z;
	  }    
	  //N.arr[i]=z;
	  //N.arrc[i]=0;
	}
      }else{
	FCG_UNIMPL(); 
      }
    }
    


  public: // ---- Neural net nonlinearities -----------------------------------------------------------------


    RFtensor ReLU() const{
      FCG_CPUONLY();
      RFtensor R(dims);
      FCG_UNIMPL();
      for(int i=0; i<asize; i++) 
	R.arr[i]=(arr[i]>0)*arr[i];
      return R;
    }



  public: // ---- In-place operations ----------------------------------------------------------------------


    void operator+=(const RFtensor& x){
      assert(asize==x.asize);
      if(device==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	return; 
      }
      const float alpha = 1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
    }

    void operator-=(const RFtensor& x){
      assert(asize==x.asize);
      if(device==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	return; 
      }
      const float alpha = -1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
    }

    void operator*=(const float x){
      FCG_CPUONLY();
      pullin<RFtensor> p(*this);
      for(int i=0; i<asize; i++) arr[i]*=x;
    }

    void operator*=(const float x){
      pullin<RFtensor> p(*this);
      for(int i=0; i<asize; i++) arr[i]=arr[i]*x;
    }

    void operator/=(const float x){
      pullin<RFtensor> p(*this);
      for(int i=0; i<asize; i++) arr[i]/=x;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------

#include "RFtensor_add.hpp"

    void add_odot(const RFtensor& x, const RFtensor& y){
      assert(x.asize==asize);
      assert(y.asize==asize);
      for(int i=0; i<asize; i++){
	arr[i]+=x.arr[i]*y.arr[i]-x.arrc[i]*y.arrc[i];
      }
    }

    void add_odotc(const RFtensor& x, const RFtensor& y){
      assert(x.asize==asize);
      assert(y.asize==asize);
      for(int i=0; i<asize; i++){
	arr[i]+=x.arr[i]*y.arr[i]+x.arrc[i]*y.arrc[i];
      }
    }


    void add_ReLU(const RFtensor& x){
      assert(x.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*x.arr[i];
    }

    void add_LeakyReLU(const RFtensor& x, const float alpha){
      assert(x.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+alpha*(x.arr[i]<0))*x.arr[i];
    }

    void add_ReLU_back(const RFtensor& g, const RFtensor& x){
      assert(x.asize==asize);
      assert(g.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*g.arr[i];
    }

    void add_LeakyReLU_back(const RFtensor& g, const RFtensor& x, const float alpha){
      assert(x.asize==asize);
      assert(g.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+(x.arr[i]<=0)*alpha)*g.arr[i];
    }

#include "RFtensor_add_Mprod.hpp"


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      if(device>0) return RFtensor(*this,device_id(0)).str(indent);
      assert(device==0);
      ostringstream oss;

      if(k==1){
	  oss<<indent<<"[ ";
	  for(int j=0; j<dims[0]; j++)
	  oss<<arr[j]<<" ";
	  oss<<"]";
	}
	
	if(k==2){
	  for(int i=0; i<dims[0]; i++){
	    oss<<indent<<"[ ";
	    for(int j=0; j<dims[1]; j++){
	      oss<<(*this)({i,j})<<" ";
	    }
	    oss<<"]";
	    if(i<dims[0]-1) oss<<"\n";
	  }
	  oss<<endl;
	}
	
	if(k==3){
	  for(int c=0; c<dims[2]; c++){
	    oss<<indent<<"Slice (*,*,"<<c<<"):"<<endl;
	    for(int i=0; i<dims[0]; i++){
	      oss<<indent<<"[ ";
	      for(int j=0; j<dims[1]; j++)
		oss<<(*this)({i,j,c})<<" ";
	      oss<<"]";
	      if(i<dims[0]-1) oss<<"\n";
	    }
	    oss<<endl;
	  }
	  oss<<endl;  
	}

	return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const RFtensor& x){
      stream<<x.str(); return stream;
    }

  };
  


#endif 


cst=x.cst;  
