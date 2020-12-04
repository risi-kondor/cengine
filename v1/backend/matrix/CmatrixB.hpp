#ifndef _CmatrixB
#define _CmatrixB

#include "Cobject.hpp"
#include "RscalarB.hpp"
#include "CscalarB.hpp"


namespace Cengine{

  class CmatrixB: public Cobject{
  public:

    int n0;
    int n1;
    int nbu;
    mutable int dev=0; 

  protected:

    int asize; 
    int bst;
    int cst;
    int tsize;
    int memsize;
    bool is_view=false;

    mutable float* arr=nullptr;
    mutable float* arrc;
    mutable float* arrg=nullptr;
    mutable float* arrgc;

  public:

    CmatrixB(){
      CMATRIXB_CREATE();
    }

    ~CmatrixB(){
      CMATRIXB_DESTROY();
    }

    string classname() const{
      return "CmatrixB";
    }

    string describe() const{
      if(nbu>=0) return "CmatrixB("+to_string(n0)+","+to_string(n1)+") ["+to_string(nbu)+"]";
      return "CmatrixB"+to_string(n0)+","+to_string(n1)+")";
    }



  public: // ---- Filled constructors -----------------------------------------------------------------------


    CmatrixB(const int _n0, const int _n1, const int _nb, const int _dev=0):
      n0(_n0), n1(_n1), nbu(_nb), dev(_dev){
      CMATRIXB_CREATE();
      make_strides();
      reallocate();
    }

    CmatrixB(const int _n0, const int _n1, const int _nb, const fill_raw& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,_nb,_dev){}
    CmatrixB(const int _n0, const int _n1, const fill_raw& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,-1,_dev){}

    CmatrixB(const int _n0, const int _n1, const int _nb, const fill_zero& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,_nb,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }
    CmatrixB(const int _n0, const int _n1, const fill_zero& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,-1,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    CmatrixB(const int _n0, const int _n1, const int _nb, const fill_ones& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,_nb,_dev){
      if(dev==0) std::fill(arr,arr+memsize,1.0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,1.0,memsize*sizeof(float)));
    }
    CmatrixB(const int _n0, const int _n1, const fill_ones& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,-1,_dev){
      if(dev==0) std::fill(arr,arr+memsize,1.0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,1.0,memsize*sizeof(float)));
    }

    CmatrixB(const int _n0, const int _n1, const int _nb, const fill_sequential& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,_nb,_dev){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=i;
      for(int i=0; i<asize; i++) arrc[i]=0;
      to_device(dev);
    }
    CmatrixB(const int _n0, const int _n1, const fill_sequential& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,-1,dummy,_dev){}

    CmatrixB(const int _n0, const int _n1, const int _nb, const fill_gaussian& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,_nb,_dev){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      for(int i=0; i<asize; i++) arrc[i]=distr(rndGen);
      to_device(dev);
    }
    CmatrixB(const int _n0, const int _n1, const fill_gaussian& dummy, const int _dev=0):
      CmatrixB(_n0,_n1,-1,dummy,_dev){}


    CmatrixB(const int _n0, const int _n1, 
      std::function<complex<float>(const int i, const int j)> fn, const int _dev=0):
      CmatrixB(_n0,_n1,-1,0){
      for(int i=0; i<n0; i++)
	for(int j=0; j<n1; j++)
	  set(i,j,fn(i,j));
      to_device(_dev);
    }
	  
    CmatrixB(const CmatrixB& x, 
      std::function<complex<float>(const int i, const int j, const complex<float>)> fn, const int _dev=0):
      CmatrixB(x.n0,x.n1,x.nbu,0){
      _pullin<CmatrixB> t(x);
      for(int i=0; i<n0; i++)
	for(int j=0; j<n1; j++)
	  set(i,j,fn(i,j,x.get(i,j)));
      to_device(_dev);
    }


  private: // ---- Allocation -------------------------------------------------------------------------------


    void make_strides(){
      asize=n0*n1;
      bst=roundup(asize,32);
      if(nbu==-1) cst=bst;
      else cst=nbu*bst;
      tsize=2*cst;
      memsize=tsize;
    }
    
    void reallocate(){
      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst;
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	//CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
	arrgc=arrg+cst;
      }
    }

    void reallocate(int _dev) const{
      if(_dev==0){
	arr=new float[memsize];
	arrc=arr+cst;
      }
      if(_dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	//CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
	arrgc=arrg+cst;
      }
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CmatrixB(const CmatrixB& x): 
      n0(x.n0), n1(x.n1), nbu(x.nbu), dev(x.dev), 
      asize(x.asize), bst(x.bst), cst(x.cst), tsize(x.tsize), memsize(x.memsize){
      CMATRIXB_CREATE();
      COPY_WARNING;
      reallocate();
      if(dev==0) std::copy(x.arr,x.arr+memsize,arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    CmatrixB(const CmatrixB& x, const nowarn_flag& dummy): 
      n0(x.n0), n1(x.n1), nbu(x.nbu), dev(x.dev), 
      asize(x.asize), bst(x.bst), cst(x.cst), tsize(x.tsize), memsize(x.memsize){
      reallocate();
      CMATRIXB_CREATE();
      if(dev==0) std::copy(x.arr,x.arr+memsize,arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    CmatrixB(CmatrixB&& x): 
      n0(x.n0), n1(x.n1), nbu(x.nbu), dev(x.dev), 
      asize(x.asize), bst(x.bst), cst(x.cst), tsize(x.tsize), memsize(x.memsize){
      CMATRIXB_CREATE();
      COPY_WARNING;
      is_view=x.is_view;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
    }

    CmatrixB& operator=(const CmatrixB& x){
      n0=x.n0; n1=x.n1; nbu=x.nbu; dev=x.dev; 
      asize=x.asize; bst=x.bst; cst=x.cst; tsize=x.tsize; memsize=x.memsize;
      if(!is_view) delete arr;
      if(!is_view && arrg) cudaFree(arrg); 
      reallocate();
      if(dev==0) std::copy(x.arr,x.arr+memsize,arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      return *this; 
    }

    CmatrixB& operator=(CmatrixB&& x){
      n0=x.n0; n1=x.n1; nbu=x.nbu; dev=x.dev; 
      asize=x.asize; bst=x.bst; cst=x.cst; tsize=x.tsize; memsize=x.memsize;
      if(!is_view) delete arr;
      if(!is_view && arrg) cudaFree(arrg); 
      is_view=x.is_view;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      return *this; 
    }

    CmatrixB* clone() const{
      return new CmatrixB(*this, nowarn);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    CmatrixB(const CmatrixB& x, const int dev): 
      n0(x.n0), n1(x.n1), nbu(x.nbu), dev(x.dev), 
      asize(x.asize), bst(x.bst), cst(x.cst), tsize(x.tsize), memsize(x.memsize){
      CMATRIXB_CREATE();
      COPY_WARNING;
      if(dev==0){
	if(x.dev==0) std::copy(x.arr,x.arr+memsize,arr);
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost)); 
      }
      if(dev==1){
	if(x.dev==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }

    const CmatrixB& to_device(const int _dev) const{
      if(dev==_dev) return *this;
      
      if(_dev==0){
 	delete[] arr;
	reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	if(!is_view && arrg) CUDA_SAFE(cudaFree(arrg));
	const_cast<CmatrixB*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }
      
      if(_dev>0){
	if(!is_view && arrg) CUDA_SAFE(cudaFree(arrg));
	reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<CmatrixB*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }

      return *this;
    }

    CmatrixB(const Gtensor<complex<float> >& x): 
      CmatrixB(x.dims[0],x.dims[1],fill::raw,0){
      int _dev=x.device; 
      x.to_device(0);
      for(int i=0; i<asize; i++){
	arr[i]=std::real(x.arr[i]);
	arrc[i]=std::imag(x.arr[i]);
      }
      to_device(_dev);
      x.to_device(_dev);
    }
    
    template<typename TYPE>
    operator Gtensor<complex<TYPE> >(){
      Gtensor<complex<TYPE> > R(dims(n0,n1),fill::raw);
      to_device(0);
      for(int i=0; i<asize; i++)
	R.arr[i]=complex<TYPE>(arr[i],arrc[i]);
      return R;
    }

    
  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }

    int get_device() const{
      return dev;
    }

    void resize(const int _n0, const int _n1){
      assert(n1*n0==_n1*_n0);
      n0=_n0;
      n1=_n1;
    }

    float re(const int i, const int j) const{
      return arr[i*n1+j];
    }

    float im(const int i, const int j) const{
      return arrc[i*n1+j];
    }

    float& re(const int i, const int j){
      return arr[i*n1+j];
    }

    float& im(const int i, const int j){
      return arrc[i*n1+j];
    }

    complex<float> val(const int i, const int j) const{
      return complex<float>(arr[i*n1+j],arrc[i*n1+j]);
    }

    complex<float> get(const int i, const int j) const{
      return complex<float>(arr[i*n1+j],arrc[i*n1+j]);
    }

    complex<float> operator()(const int i, const int j) const{
      return complex<float>(arr[i*n1+j],arrc[i*n1+j]);
    }

    void set(const int i, const int j, const complex<float> x){
      int t=i*n1+j;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i, const int j, const complex<float> x){
      int t=i*n1+j;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }





  public: // ---- Operations ---------------------------------------------------------------------------------


    void zero(){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    CmatrixB* conj() const{
      if(dev==0){
	CmatrixB* R=new CmatrixB(n0,n1,fill::raw,0);
	std::copy(arr,arr+asize,R->arr);
	for(int i=0; i<asize; i++) R->arrc[i]=-arrc[i];
	return R;
      }
      CmatrixB* R=new CmatrixB(n0,n1,fill::zero,dev);
      const float alpha = 1.0;
      const float malpha = -1.0;
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, arrg, 1, R->arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &malpha, arrgc, 1, R->arrgc, 1));
      return R;
    }

    CmatrixB* transp() const{
      if(dev==0){
	CmatrixB* R=new CmatrixB(n1,n0,fill::raw,0);
	for(int i=0; i<n1; i++)
	  for(int j=0; j<n0; j++){
	    R->arr[i*n0+j]=arr[j*n1+i];
	    R->arrc[i*n0+j]=arrc[j*n1+i];
	  }
	return R;
      }
      CmatrixB* R=new CmatrixB(n1,n0,fill::zero,dev);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &alpha,arrg,n1,&beta,R->arrg,n0,R->arrg,n0));
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &alpha,arrgc,n1,&beta,R->arrgc,n0,R->arrgc,n0));
      return R;
    }

    CmatrixB* herm() const{
      if(dev==0){
	CmatrixB* R=new CmatrixB(n1,n0,fill::raw,0);
	for(int i=0; i<n1; i++)
	  for(int j=0; j<n0; j++){
	    R->arr[i*n0+j]=arr[j*n1+i];
	    R->arrc[i*n0+j]=-arrc[j*n1+i];
	  }
	return R;
      }
      CmatrixB* R=new CmatrixB(n1,n0,fill::zero,dev);
      const float alpha = 1.0;
      const float malpha = -1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &alpha,arrg,n1,&beta,R->arrg,n0,R->arrg,n0));
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &malpha,arrgc,n1,&beta,R->arrgc,n0,R->arrgc,n0));
      return R;
    }

    //CmatrixB* divide_cols(const CmatrixB& N) const{
    //return new CmatrixB(CFmatrix::divide_cols(N));
    //}

    //CmatrixB* normalize_cols() const{
    //return new CmatrixB(CFmatrix::normalize_cols());
    //}


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add(const CmatrixB& x){
      assert(asize==x.asize);
      _tmpdev<CmatrixB>(dev,x);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
      }
    }


    void add_conj(const CmatrixB& x){
      assert(asize==x.asize);
      _tmpdev<CmatrixB>(dev,x);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	const float malpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &malpha, x.arrgc, 1, arrgc, 1));
      }
    }

    
    void add_transp(const CmatrixB& x) const{
      assert(asize==x.asize);
      if(dev==0){
	for(int i=0; i<n1; i++)
	  for(int j=0; j<n0; j++){
	    arr[i*n0+j]+=x.arr[j*n1+i];
	    arrc[i*n0+j]+=x.arrc[j*n1+i];
	  }
	return;
      }
      const float alpha = 1.0;
      const float beta = 1.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &alpha,x.arrg,n1,&beta,arrg,n0,arrg,n0));
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &alpha,x.arrgc,n1,&beta,arrgc,n0,arrgc,n0));
    }


    void add_herm(const CmatrixB& x) const{
      assert(asize==x.asize);
      if(dev==0){
	for(int i=0; i<n1; i++)
	  for(int j=0; j<n0; j++){
	    arr[i*n0+j]+=x.arr[j*n1+i];
	    arrc[i*n0+j]-=x.arrc[j*n1+i];
	  }
	return;
      }
      const float alpha = 1.0;
      const float malpha = -1.0;
      const float beta = 1.0;
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &alpha,x.arrg,n1,&beta,arrg,n0,arrg,n0));
      CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n0,n1,
	  &malpha,x.arrgc,n1,&beta,arrgc,n0,arrgc,n0));
    }


    void subtract(const CmatrixB& x){
      assert(asize==x.asize);
      _tmpdev<CmatrixB>(dev,x);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
      }
    }


  public: // ---- Add times const ----------------------------------------------------------------------------

    
    void add(const CmatrixB& x, const float c){
      assert(asize==x.asize);
      if(dev!=1 || x.dev!=1){
	to_device(0);
	x.to_device(0);
      }
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*c;
	return;
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
      }
    }

    void add(const CmatrixB& x, const complex<float> c){
      assert(asize==x.asize);
      float cr=std::real(c);
      float ci=std::imag(c);
      if(dev!=1 || x.dev!=1){
	to_device(0);
	x.to_device(0);
      }
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
	return;
      }
      if(dev==1){
	const float mci=-ci; 
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
      }
    }


    void add_conj(const CmatrixB& x, const complex<float> c){
      assert(asize==x.asize);
      float cr=std::real(c);
      float ci=std::imag(c);
      if(dev!=1 || x.dev!=1){
	to_device(0);
	x.to_device(0);
      }
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr+x.arrc[i]*ci;
	for(int i=0; i<asize; i++) arrc[i]+=-x.arrc[i]*cr+x.arr[i]*ci;
	return;
      }
      if(dev==1){
	const float mcr=-cr; 
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrgc, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
	CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mcr, x.arrgc, 1, arrgc, 1));
      }
    }


  public: // ---- Add times scalar ----------------------------------------------------------------------------


    void add(const CmatrixB& x, const RscalarB y){
      add(x,y.val);
    }

    void add(const CmatrixB& x, const CscalarB y){
      add(x,y.val);
    }

    void add_x_times_yc(const CmatrixB& x, const CscalarB y){
      add(x,std::conj(y.val));
    }

    void add_xc_times_y(const CmatrixB& x, const CscalarB y){
      add_conj(x,y.val);
    }


  public: // ---- Add times matrix ----------------------------------------------------------------------------


#include "CmatrixB_add_mprod.hpp"

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    //string str(const string indent="") const{
    //return Gmatrix<complex<float> >(*this);
    //}

   
  };


  inline CmatrixB& asCmatrixB(Cobject* x, const char* s){
    return downcast<CmatrixB>(x,s);
  }

  inline CmatrixB& asCmatrixB(Cnode* x, const char* s){
    return downcast<CmatrixB>(x,s);
  }
  
  inline CmatrixB& asCmatrixB(Cnode& x, const char* s){
    return downcast<CmatrixB>(x,s);
  }


#define CMATRIXB(x) asCmatrixB(x,__PRETTY_FUNCTION__) 


}

#endif

