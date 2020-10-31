#ifndef _CtensorBpack
#define _CtensorBpack

#include "CtensorB.hpp"
#include "Cnode.hpp"

namespace Cengine{

  class CtensorBpack{
  public:

    Gdims dims; 
    int nbu=-1;
    mutable int device=0;

    vector<CtensorB*> pack;
    mutable float** parr=nullptr;
    mutable float** parrc=nullptr;
    mutable bool parr_valid=false;

    int N;
    bool temp=false;
    float* temp_base;
    
    CtensorBpack(const Gdims& _dims, const int _nbu=-1, const int dev=0):
      dims(_dims), nbu(_nbu), device(dev){}

    CtensorBpack(const CtensorB& x):
      dims(x.dims), nbu(x.nbu), device(x.device){}

    CtensorBpack(const vector<Cnode*>& x, const int s){
      const int N=x.size();
      assert(N>0);
      assert(dynamic_cast<CtensorB*>(x[0]->op->inputs[s]->obj));
      device=dynamic_cast<CtensorB*>(x[0]->op->inputs[s]->obj)->device;
      pack.resize(N);
      for(int i=0; i<N; i++){
	pack[i]=dynamic_cast<CtensorB*>(x[i]->op->inputs[s]->obj);
	pack[i]->to_device(device);
      }
    }

    ~CtensorBpack(){
      if(parr) CUDA_SAFE(cudaFree(parr));
      if(parrc) CUDA_SAFE(cudaFree(parrc));
      if(temp) CUDA_SAFE(cudaFree(temp_base));
    }


  public: // -------------------------------------------------------------------------------------------------


    CtensorBpack(const int N, const Gdims& _dims, const int _nbu, const fill_raw& dummy, const int dev=0):
      dims(_dims), nbu(_nbu), device(dev){
      assert(dev==1);
      
    }

    CtensorBpack(const int _N, const CtensorB& model, const fill_raw& dummy, const int dev=1):
      dims(model.dims), nbu(model.nbu), device(dev), N(_N){
      assert(dev==1);
      int cst=model.cst;
      int memsize=model.memsize;
      float* temp_base;
      //int cst=roundup(model->cst*sizeof(float),128);
      //int memsize=roundup(model->memsize*sizeof(float),128);
      CUDA_SAFE(cudaMalloc((void **)&temp_base, memsize*N));

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



  public: // -------------------------------------------------------------------------------------------------


    CtensorBpack(const CtensorBpack& x)=delete;

    CtensorBpack& operator=(const CtensorBpack& x)=delete;


  public: // -------------------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }


    Gdims get_dims() const{
      return dims; 
    }


    void push_back(CtensorB* x){
      assert(!temp);
      assert(x->dims==dims);
      assert(x->nbu==nbu);
      pack.push_back(x);
      parr_valid=false;
    }


    void push_back(CtensorB& x){
      assert(!temp);
      assert(x.dims==dims);
      assert(x.nbu==nbu);
      pack.push_back(&x);
      parr_valid=false;
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
      assert(!temp);
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
      assert(!temp);
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
    void sum_into_cu(const CtensorB& R, const cudaStream_t& stream);
#endif 

    void copy(const CtensorBpack& x){
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
    
    void add(const CtensorBpack& x){
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

    void sum_into(CtensorB& R){
      assert(temp);

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
    
    #include "CtensorBpack_add_Mprod.hpp"


  };

}

#endif
