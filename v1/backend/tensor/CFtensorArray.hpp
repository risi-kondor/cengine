#ifndef _CFtensorArray
#define _CFtensorArray

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
//#include "CFtensorArrayHelpers.hpp"

extern default_random_engine rndGen;


namespace Cengine{


  class CFtensorArray{
  public:

    int ak; 
    int k;
    Gdims adims;
    Gdims dims;

    vector<int> astrides;
    vector<int> strides;
    int aasize=0; // total number of cells
    int asize=0; // total number of numbers per cell
    int cellstride;
    int memsize=0;
    int cst=0; 

    mutable float* arr=nullptr;
    mutable float* arrc=nullptr;
 
    mutable float* arrg=nullptr;
    mutable float* arrgc=nullptr;

    bool is_view=false;
    bool is_contiguous=true;

    mutable int device=0;

    ~CFtensorArray(){
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const {return "Cengine::CFtensorArray";}


  public: // ---- Constructors -------------------------------------------------------------------------------


    CFtensorArray(){
      arr=nullptr;
    }

    CFtensorArray(const Gdims& _adims, const Gdims& _dims, const int dev=0): 
      adims(_adims), dims(_dims), astrides(_adims.size()), strides(_dims.size()){
      make_strides();
      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst; 
	device=0;
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
	device=1;
      }
    }

    /*
    CFtensorArray(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, 
      const int _memsize, const int _cst, const device_id& _dev=0):
      k(_k), dims(_dims), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst), device(_dev.id()){
      //CENGINE_DEPRECATED();
      if(device==0){
	arr=new float[memsize];
	arrc=arr+cst;
      }
      if(device==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
      }
    }
    */
    
    /*
    CFtensorArray(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, 
      const int _memsize, const int _cst, const int dev=0):
      k(_k), dims(_dims), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst), device(dev){
      if(device==0){
	arr=new float[memsize];
	arrc=arr+cst;
      }
      if(device==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
      }
    }
    */
    
    void make_strides(){

      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      cellstride=2*cst; 

      ak=adims.size();
      astrides.resize(ak);
      astrides[k-1]=cellstride;
      for(int i=ak-2; i>=0; i--)
	astrides[i]=astrides[i+1]*adims[i+1];
      aasize=adims.asize();

      memsize=strides[0]*dims[0];
    }

    void reallocate(const int dev=0) const{
      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst;
	device=0;
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
	device=1;
      }
    }


  public: // ---- Filled constructors -------------------------------------------------------------------------

    CFtensorArray(const Gdims& _adims, const Gdims& _dims, const fill_raw& dummy, const int dev=0): 
      CFtensorArray(_adims,_dims,dev){}
    
    CFtensorArray(const Gdims& _adims, const Gdims& _dims, const fill_zero& dummy, const int dev=0): 
      CFtensorArray(_adims,_dims,dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    CFtensorArray(const Gdims& _adims, const Gdims& _dims, const fill_ones& dummy, const int dev=0): 
      CFtensorArray(_adims,_dims,dev){
      if(dev==0){
	std::fill(arr,arr+asize,1);
	std::fill(arrc,arrc+asize,0);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg,1,asize*sizeof(float)));
	CUDA_SAFE(cudaMemset(arrgc,0,asize*sizeof(float)));
      }
    }

    /*
    CFtensorArray(const Gdims& _adims, const Gdims& _dims, const fill_identity& dummy, const int dev=0): 
      CFtensorArray(_dims,fill::raw,0){
      assert(dims[k-1]==dims[k-2]);
      std::fill(arr,arr+memsize,0);
      for(int i=0; i<dims[k-1]; i++)
	arr[i*(strides[k-2]+1)]=1;
      to_device(dev);
    }
    */

    CFtensorArray(const Gdims& _adims, const Gdims& _dims, const fill_gaussian& dummy, const int dev):
      CFtensorArray(_adims,_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int j=0; j<aasize; j++){
	for(int i=0; i<asize; i++) arr[i+j*cellstride]=distr(rndGen);
	for(int i=0; i<asize; i++) arrc[i+j*cellstride]=distr(rndGen);
      }
      to_device(dev);
    }

    CFtensorArray(const Gdims& _dims, const fill_gaussian& dummy, const float c, const int dev):
      CFtensorArray(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int j=0; j<aasize; j++){
	for(int i=0; i<asize; i++) arr[i+j*cellstride]=c*distr(rndGen);
	for(int i=0; i<asize; i++) arrc[i+j*cellstride]=c*distr(rndGen);
      }
     to_device(dev);
    }

    CFtensorArray(const Gdims& _dims, const fill_sequential& dummy, const int dev=0):
      CFtensorArray(_dims,fill::zero,0){
      for(int j=0; j<aasize; j++){
	for(int i=0; i<asize; i++) arr[i+j*cellstride]=i+j*asize;
      }
      to_device(dev);
    }

    CFtensorArray(const Gdims& _dims, const fill_const<complex<float> >& dummy, const int dev=0):
      CFtensorArray(_dims,fill::raw,0){
      for(int j=0; j<aasize; j++){
	float re=std::real(dummy.p);
	for(int i=0; i<asize; i++) arr[i+j*cellstride]=re;
	float im=std::imag(dummy.p);
	for(int i=0; i<asize; i++) arrc[i+j*cellstride]=im;
      }
      to_device(dev);
    }

    /*
    CFtensorArray(const CFtensorArray& x, std::function<complex<float>(const complex<float>)> fn):
      CFtensorArray(x.dims,fill::raw,0){
      assert(device==0);
      for(int i=0; i<asize; i++){
	complex<float> t=fn(complex<float>(x.arr[i],x.arrc[i]));
	arr[i]=std::real(t);
	arrc[i]=std::imag(t);
      }
    }
    */


  public: // ---- Copying -------------------------------------------------------------------------------------


    CFtensorArray(const CFtensorArray& x): 
      CFtensorArray(x.ak,x.adims,x.astrides,x.aasize,x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.device){
      COPY_WARNING;
      if(device==0) std::copy(x.arr,x.arr+memsize,arr);
      if(device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }
        
    CFtensorArray(const CFtensorArray& x, const nowarn_flag& dummy): 
      CFtensorArray(x.ak,x.adims,x.astrides,x.aasize,x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.device){
      if(device==0) std::copy(x.arr,x.arr+memsize,arr);
      if(device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
    }

    CFtensorArray(const CFtensorArray& x, const fill_raw& dummy): 
      CFtensorArray(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.device){}

    /*
    CFtensorArray(const CFtensorArray& x, const device_id& dev): 
      CFtensorArray(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,dev){
      if(device==0){
	if(x.device==0) std::copy(x.arr,x.arr+memsize,arr);
	if(x.device==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost)); 
      }
      if(device==1){
	if(x.device==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
	if(x.device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }
    */
    
    CFtensorArray(const CFtensorArray& x, const int dev): 
      CFtensorArray(x.ak,x.adims,x.astrides,x.aasize,x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,dev){
      if(device==0){
	if(x.device==0) std::copy(x.arr,x.arr+memsize,arr);
	if(x.device==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost)); 
      }
      if(device==1){
	if(x.device==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
	if(x.device==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }
    
    CFtensorArray(CFtensorArray&& x){
      ak=x.ak; adims=x.adims; astrides=x.astrides; aasize=x.aasize;
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize;
      cellskip=x.cellskip; 
      memsize=x.memsize; cst=x.cst;  
      arr=x.arr; x.arr=nullptr; arrg=x.arrg; x.arrg=nullptr;
      arrc=arr+cst; arrgc=arrg+cst; 
      is_view=x.is_view;
      is_contiguous=x.is_contiguous;
      device=x.device;
    }
    
    CFtensorArray& operator=(const CFtensorArray& x){
      memsize=x.memsize; cst=x.cst;
      cellskip=x.cellskip; 
      if(!is_view) delete arr;
      if(!is_view && arrg) cudaFree(arrg); 
      ak=x.ak; adims=x.adims; astrides=x.astrides; aasize=x.aasize;
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; device=x.device;
      if(device==0){
	arr=new float[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
	arrc=arr+cst; 
      }
      if(device==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	arrgc=arrg+cst;
      }
      return *this;
    }

    CFtensorArray& operator=(CFtensorArray&& x){
      memsize=x.memsize; cst=x.cst; 
      cellskip=x.cellskip; 
      if(!is_view && arr) delete arr;
      if(!is_view && arrg) CUDA_SAFE(cudaFree(arrg));
      ak=x.ak; adims=x.adims; astrides=x.astrides; aasize=x.aasize;
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; device=x.device; 
      arr=x.arr; x.arr=nullptr; arrg=x.arrg; x.arrg=nullptr; 
      arrc=arr+cst; arrgc=arrg+cst; 
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Transporters ------------------------------------------------------------------------------
 

    const CFtensorArray& to_device(const int _dev) const{
      if(_dev==0){
 	if(device==0) return *this;
 	delete[] arr;
	reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<CFtensorArray*>(this)->arrg=nullptr;
	device=0;
	return *this;
      }
      if(_dev>0){
	if(device==_dev) return *this;
	if(arrg) cudaFree(arrg);
	reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<CFtensorArray*>(this)->arr=nullptr;
	device=_dev;
	//if(!is_view && arrg) CUDA_SAFE(cudaFree(arrg));
	return *this;
      }
      return *this;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------

    /*
    template<typename TYPE>
    CFtensorArray(const Gtensor<complex<TYPE> >& x): 
      CFtensorArray(x.dims,fill::raw,0){
      int dev=x.device; 
      x.to_device(0);
      for(int i=0; i<asize; i++){
	arr[i]=std::real(x.arr[i]);
	arrc[i]=std::imag(x.arr[i]);
      }
      to_device(dev);
      x.to_device(dev);
    }
    
    template<typename TYPE>
    operator Gtensor<complex<TYPE> >(){
      Gtensor<complex<TYPE> > R(dims,fill::raw);
      to_device(0);
      for(int i=0; i<asize; i++)
	R.arr[i]=complex<TYPE>(arr[i],arrc[i]);
      return R;
    }
    */


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
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get(const Gindex& ix) const{
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return complex<float>(arr[t],arrc[t]);
    }
    
    CFtensorArray& set(const Gindex& ix, const complex<float>& v){
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=std::real(v);
      arrc[t]=std::imag(v);
      return *this;
    }

    CFtensorArray& inc(const Gindex& ix, const complex<float>& v){
      FCG_CPUONLY();
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]+=std::real(v);
      arrc[t]+=std::imag(v);
      return *this;
    }

    CFtensorArray& set(const complex<float>& v){
      FCG_CPUONLY();
      if(device==0){
	std::fill(arr,arr+asize,std::real(v));
	std::fill(arrc,arrc+asize,std::imag(v));
      }
      if(device==1){
      }
      return *this;
    }	

    #include "CFtensorArray_access.hpp"


  public: // ---- Elementwise Operations --------------------------------------------------------------------


    void zero(){
      if(device==0) std::fill(arr,arr+memsize,0);
      if(device==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    /*
    void set(const fill_gaussian& dummy){
      const int _dev=device;
      to_device(0);
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      for(int i=0; i<asize; i++) arrc[i]=distr(rndGen);
      to_device(_dev);
    }
    */


    bool operator==(const CFtensorArray& x) const{
      if(x.aasize!=aasize) return false; 
      if(x.asize!=asize) return false; 
      pullin2<CFtensorArray> p(*this,x);
      for(int j=0; j<aasize; j++){
	for(int i=0; i<asize; i++)
	  if(arr[i]!=x.arr[i]) return false;
	for(int i=0; i<asize; i++)
	  if(arrc[i]!=x.arrc[i]) return false;
      }
      return true;
    }


  public: // ---- Non in-place operations --------------------------------------------------------------------


    CFtensorArray plus(const CFtensorArray& x) const{
      assert(asize==x.asize);
      x.to_device(device);
      if(device==0){
	CFtensorArray R(dims,fill::raw,device);
	for(int i=0; i<asize; i++) R.arr[i]=arr[i]+x.arr[i];
	for(int i=0; i<asize; i++) R.arrc[i]=arrc[i]+x.arrc[i];
	return R;
      }
      CFtensorArray R(*this,nowarn);
      const float alpha = 1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, R.arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, R.arrgc, 1));
      return R;
    }


    CFtensorArray minus(const CFtensorArray& x) const{
      assert(asize==x.asize);
      x.to_device(device);
      if(device==0){
	CFtensorArray R(dims,fill::raw,device);
	for(int i=0; i<asize; i++) R.arr[i]=arr[i]-x.arr[i];
	for(int i=0; i<asize; i++) R.arrc[i]=arrc[i]-x.arrc[i];
	return R;
      }
      CFtensorArray R(*this,nowarn);
      const float alpha = -1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, R.arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, R.arrgc, 1));
      return R;
    }


    CFtensorArray times(const float c) const{
      if(device==0){
	CFtensorArray R(dims,fill::raw,0);
	for(int i=0; i<asize; i++) R.arr[i]=c*arr[i];
	for(int i=0; i<asize; i++) R.arrc[i]=c*arrc[i];
	return R;
      }
      CFtensorArray R(dims,fill::zero,device);
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, arrg, 1, R.arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, arrgc, 1, R.arrgc, 1));
      return R;
    }


    CFtensorArray times(const complex<float> c) const{
      if(device==0){
	CFtensorArray R(dims,fill::raw,0);
	float cr=std::real(c);
	float ci=std::imag(c);
	for(int i=0; i<asize; i++) R.arr[i]=cr*arr[i]-ci*arrc[i];
	for(int i=0; i<asize; i++) R.arrc[i]=cr*arrc[i]+ci*arr[i];
	return R;
      }
      CFtensorArray R(dims,fill::zero,device);
      float cr=std::real(c);
      float ci=std::imag(c);
      float mci=-std::imag(c);
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, arrg, 1, R.arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, arrgc, 1, R.arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, arrgc, 1, R.arrgc, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, arrg, 1, R.arrgc, 1));
      return R;
    }

    CFtensorArray elementwise_times(const CFtensorArray& x) const{
      FCG_CPUONLY();
      assert(asize==x.asize);
      CFtensorArray R(dims,fill::raw,0);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]*x.arr[i]-arrc[i]*x.arrc[i];
      for(int i=0; i<asize; i++) R.arrc[i]=arr[i]*x.arrc[i]+arrc[i]*x.arr[i];
      return R;
    }

    CFtensorArray elementwise_divide(const CFtensorArray& x) const{
      FCG_CPUONLY();
      assert(asize==x.asize);
      CFtensorArray R(dims,fill::raw,0);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]/x.arr[i]+arrc[i]/x.arrc[i];
      for(int i=0; i<asize; i++) R.arrc[i]=arrc[i]/x.arr[i]-arr[i]/x.arrc[i];
      return R;
    }

    CFtensorArray elementwise_pow(const float p, const complex<float> c=1.0) const{
      FCG_CPUONLY();
      CFtensorArray R(dims,fill::raw,0);
      for(int i=0; i<asize; i++){
	complex<float> t=c*pow(complex<float>(arr[i],arrc[i]),p);
	R.arr[i]=std::real(t); 
	R.arrc[i]=std::imag(t);
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

    CFtensorArray conj() const{
      if(device==0){
	CFtensorArray R(dims,fill::raw,0);
	std::copy(arr,arr+asize,R.arr);
	for(int i=0; i<asize; i++) R.arrc[i]=-arrc[i];
	return R;
      }
      CFtensorArray R(dims,fill::zero,device);
      const float alpha = 1.0;
      const float malpha = -1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, arrg, 1, R.arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &malpha, arrgc, 1, R.arrgc, 1));
      return R;
    }

    CFtensorArray transp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(device==0){
	CFtensorArray R({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]=arr[j*I+i];
	    R.arrc[i*J+j]=arrc[j*I+i];
	  }
	return R;
      }
      CFtensorArray R(dims,fill::zero,device);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrgc,I,&beta,R.arrgc,J,R.arrgc,J));
      return R;
    }


    CFtensorArray herm(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(device==0){
	CFtensorArray R({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]=arr[j*I+i];
	    R.arrc[i*J+j]=-arrc[j*I+i];
	  }
	return R;
      }
      CFtensorArray R(dims,fill::zero,device);
      const float alpha = 1.0;
      const float malpha = -1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &malpha,arrgc,I,&beta,R.arrgc,J,R.arrgc,J));
      return R;
    }



  public: // ---- Operations --------------------------------------------------------------------------------


    float norm2() const{
      if(device==0){
      float t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*arr[i]-arrc[i]*arrc[i]; 
      return t;
      }
      float tr=0;
      float ti=0;
#ifdef _WITH_CUBLAS
      cublasSdot(Cengine_cublas, asize, arrg, 1, arrg, 1, &tr);
      cublasSdot(Cengine_cublas, asize, arrgc, 1, arrgc, 1, &ti);
#else
      NOCUDA_ERROR;
#endif       
      return tr+ti;
    }


    complex<float> inp(const CFtensorArray& x) const{
      assert(asize==x.asize);
      x.to_device(device);
      if(device==0){
	float tr=0; 
	float ti=0; 
	for(int i=0; i<asize; i++){
	  tr+=arr[i]*x.arr[i]+arrc[i]*x.arrc[i];
	  ti+=arrc[i]*x.arr[i]-arr[i]*x.arrc[i];
	}
	//{CoutLock lk; cout<<*this<<endl<<endl; cout<<"  "<<asize<<" "<<tr<<":"<<ti<<endl;}
	return complex<float>(tr,ti);
      }
      float a,b,c,d;
#ifdef _WITH_CUBLAS
      cublasSdot(Cengine_cublas, asize, arrg, 1, x.arrg, 1, &a);
      cublasSdot(Cengine_cublas, asize, arrgc, 1, x.arrgc, 1, &b);
      cublasSdot(Cengine_cublas, asize, arrg, 1, x.arrgc, 1, &c);
      cublasSdot(Cengine_cublas, asize, arrgc, 1, x.arrg, 1, &d);
#else
      NOCUDA_ERROR;
#endif       
      return complex<float>(a-b,c+d);
    }

    float diff2(const CFtensorArray& x) const{
      FCG_CPUONLY();
      assert(x.dims==dims);
      assert(x.asize==asize);
      float t=0;
      for(int i=0; i<asize; i++)
	t+=(arr[i]-x.arr[i])*(arr[i]-x.arr[i])+(arrc[i]-x.arrc[i])*(arrc[i]-x.arrc[i]);
      return t;
    }

    CFtensorArray odot(const CFtensorArray& x) const{
      FCG_CPUONLY();
      assert(dims==x.dims);
      CFtensorArray R(dims,fill::raw,0);
      for(int i=0; i<asize; i++){
	R.arr[i]=arr[i]*x.arr[i]+arrc[i]*x.arrc[i];
	R.arrc[i]=arrc[i]*x.arr[i]-arr[i]*x.arrc[i];
       }
      return R;
    }

    CFtensorArray odotc(const CFtensorArray& x) const{
      FCG_CPUONLY();
      assert(dims==x.dims);
      CFtensorArray R(dims,fill::raw,0);
      for(int i=0; i<asize; i++){
	R.arr[i]=arr[i]*x.arr[i]+arrc[i]*x.arrc[i];
	R.arrc[i]=arrc[i]*x.arr[i]-arr[i]*x.arrc[i];
      }
      return R;
    }


    CFtensorArray divide_cols(const CFtensorArray& N) const{
      assert(k>=2);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      CFtensorArray R(dims,fill::zero,0);
      if(device==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      R.arr[offs+i*J+j]=arr[offs+i*J+j]/z;
	      R.arrc[offs+i*J+j]=arrc[offs+i*J+j]/z;
	    }
	  }    
	}
      }else{
	FCG_UNIMPL(); 
      }
      return R;
    }


    CFtensorArray normalize_cols() const{
      Gdims ndims=dims.chunk(0,dims.size()-1);
      const int J=dims[dims.size()-1];
      const int I=asize/J;
      CFtensorArray R(*this);
      if(device==0){
	for(int i=0; i<I; i++){
	  float tr=0;
	  float ti=0;
	  for(int j=0; j<J; j++){
	    tr+=R.arr[i*J+j]*R.arr[i*J+j];
	    ti+=R.arrc[i*J+j]*R.arrc[i*J+j];
	  }
	  float z=sqrt(tr+ti);
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]/=z;
	    R.arrc[i*J+j]/=z;
	  }    
	}
      }else{
	FCG_UNIMPL(); 
      }
      return R;
    }


    void add_normalize_cols_back(const CFtensorArray& g, const CFtensorArray& x){
      Gdims ndims=dims.chunk(0,dims.size()-1);
      const int J=dims[dims.size()-1];
      const int I=asize/J;
      CFtensorArray R(dims,fill::zero,0);
      if(device==0){
	for(int i=0; i<I; i++){
	  float tr=0;
	  float ti=0;
	  for(int j=0; j<J; j++){
	    tr+=R.arr[i*J+j]*R.arr[i*J+j];
	    ti+=R.arrc[i*J+j]*R.arrc[i*J+j];
	  }
	  float z=sqrt(tr+ti);
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]/=z;
	    R.arrc[i*J+j]/=z;
	  }    
	  //N.arr[i]=z;
	  //N.arrc[i]=0;
	}
      }else{
	FCG_UNIMPL(); 
      }
    }
    


  public: // ---- Neural net nonlinearities -----------------------------------------------------------------


    CFtensorArray ReLU() const{
      FCG_CPUONLY();
      CFtensorArray R(dims,fill::raw,0);
      FCG_UNIMPL();
      for(int i=0; i<asize; i++) 
	R.arr[i]=(arr[i]>0)*arr[i];
      for(int i=0; i<asize; i++) 
	R.arrc[i]=(arrc[i]>0)*arrc[i];
      return R;
    }



  public: // ---- In-place operations ----------------------------------------------------------------------


    void operator+=(const CFtensorArray& x){
      assert(asize==x.asize);
      if(device==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i];
	return; 
      }
      const float alpha = 1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    }

    void operator-=(const CFtensorArray& x){
      assert(asize==x.asize);
      if(device==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i];
	return; 
      }
      const float alpha = -1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    }

    void operator*=(const float x){
      FCG_CPUONLY();
      pullin<CFtensorArray> p(*this);
      for(int i=0; i<asize; i++) arr[i]*=x;
      for(int i=0; i<asize; i++) arrc[i]*=x;
    }

    void operator*=(const complex<float> x){
      pullin<CFtensorArray> p(*this);
      float xr=std::real(x);
      float xi=std::imag(x);
      for(int i=0; i<asize; i++) arr[i]=arr[i]*xr-arrc[i]*xi;
      for(int i=0; i<asize; i++) arrc[i]=arr[i]*xi+arrc[i]*xr;
    }

    void operator/=(const float x){
      pullin<CFtensorArray> p(*this);
      for(int i=0; i<asize; i++) arr[i]/=x;
      for(int i=0; i<asize; i++) arrc[i]/=x;
    }

    void operator/=(const complex<float> x){
      pullin<CFtensorArray> p(*this);
      float xr=std::real(x);
      float xi=std::imag(x);
      for(int i=0; i<asize; i++) arr[i]=arr[i]/xr+arrc[i]/xi;
      for(int i=0; i<asize; i++) arrc[i]=arrc[i]*xr-arr[i]/xi;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------

#include "CFtensorArray_add.hpp"

    void add_odot(const CFtensorArray& x, const CFtensorArray& y){
      assert(x.asize==asize);
      assert(y.asize==asize);
      for(int i=0; i<asize; i++){
	arr[i]+=x.arr[i]*y.arr[i]-x.arrc[i]*y.arrc[i];
	arrc[i]+=x.arr[i]*y.arrc[i]+x.arrc[i]*y.arr[i];
      }
    }

    void add_odotc(const CFtensorArray& x, const CFtensorArray& y){
      assert(x.asize==asize);
      assert(y.asize==asize);
      for(int i=0; i<asize; i++){
	arr[i]+=x.arr[i]*y.arr[i]+x.arrc[i]*y.arrc[i];
	arr[i]+=x.arrc[i]*y.arr[i]-x.arr[i]*y.arrc[i];
      }
    }


    void add_ReLU(const CFtensorArray& x){
      assert(x.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*x.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=(x.arrc[i]>0)*x.arrc[i];
    }

    void add_LeakyReLU(const CFtensorArray& x, const float alpha){
      assert(x.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+alpha*(x.arr[i]<0))*x.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=((x.arrc[i]>0)+alpha*(x.arrc[i]<0))*x.arrc[i];
    }

    void add_ReLU_back(const CFtensorArray& g, const CFtensorArray& x){
      assert(x.asize==asize);
      assert(g.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*g.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=(x.arrc[i]>0)*g.arr[i];
    }

    void add_LeakyReLU_back(const CFtensorArray& g, const CFtensorArray& x, const float alpha){
      assert(x.asize==asize);
      assert(g.asize==asize);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+(x.arr[i]<=0)*alpha)*g.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=((x.arrc[i]>0)+(x.arrc[i]<=0)*alpha)*g.arrc[i];
    }

#include "CFtensorArray_add_Mprod.hpp"


  public: // ---- I/O ----------------------------------------------------------------------------------------

    string str(const string indent="") const{
      if(device>0) return CFtensorArray(*this,0).str(indent);
      assert(device==0);
      ostringstream oss;

      if(k==1){
	  oss<<indent<<"[ ";
	  for(int j=0; j<dims[0]; j++)
	  oss<<"("<<arr[j]<<","<<arrc[j]<<") ";
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

    friend ostream& operator<<(ostream& stream, const CFtensorArray& x){
      stream<<x.str(); return stream;
    }

  };
  


#endif 


