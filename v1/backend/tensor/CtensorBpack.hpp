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
    float** parr=nullptr;
    float** parrc=nullptr;
    mutable bool parr_valid=false;
    
    CtensorBpack(const Gdims& _dims, const int _nbu=-1, const int dev=0):
      dims(_dims), nbu(_nbu), device(dev){}

    CtensorBpack(const CtensorB& x):
      dims(x.dims), nbu(x.nbu), device(x.device){}

    CtensorBpack(const vector<Cnode*>& x, const int s){
      const int N=x.size();
      assert(N>0);
      assert(dynamic_cast<CtensorB*>(x[0]->op->inputs[s]->obj));
      pack.resize(N);
      for(int i=0; i<N; i++)
	pack[i]=dynamic_cast<CtensorB*>(x[i]->op->inputs[s]->obj);
    }


    ~CtensorBpack(){
      if(parr) CUDA_SAFE(cudaFree(parr));
      if(parrc) CUDA_SAFE(cudaFree(parr));
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


    float** get_parr() const{
      if(!parr || !parr_valid) renew_parr();
      return parr;
    }


    float** get_parrc() const{
      if(!parrc || !parr_valid) renew_parr();
      return parrc;
    }


    void renew_parr() const{

      if(parr) CUDA_SAFE(cudaFree(parr));

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
      if(_dev.id()==device) return; 
      parr_valid=false; 
      for(auto p: pack)
	p->to_device(_dev);
      device=_dev.id();
    }



  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    /*
    void add(const CtensorBpack& x){
      const int N=pack.size();
      assert(x.pack.size()==N);

      if(device==0){
	for(int i=0; i<N; i++)
	  pack[i]->add(*x.pack[i]);
      }else{
	if(!parr_valid) renew_parr();
	if(!x.parr_valid) x.renew_parr();
	CUBLAS_SAFE(gemmBatched(genet_cublas,CUBLAS_OP_N,CUBLAS_OP_N);)
      }

    }
    */
 
    #include "CtensorBpack_add_Mprod.hpp"


  };

}

#endif
