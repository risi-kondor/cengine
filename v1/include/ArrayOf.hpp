#ifndef _ArrayOf
#define _ArrayOf

#include "Cengine_base.hpp"
#include "Gdims.hpp"


namespace Cengine{

  template <typename OBJ>
  class ArrayOf{
  public:

    int k;
    Gdims dims;
    vector<int> strides;
    int asize;
    int device=0; 

    OBJ** arr;

    ArrayOf(){
      k=0;
      dims={};
      asize=0;
      arr=new OBJ*[0];
    }

    ~ArrayOf(){
      //cout<<"del"<<endl; 
      for(int i=0; i<asize; i++) 
	delete arr[i];
      delete[] arr;
    }


  public: // ---- Filled constructors -------------------------------------------------------------------------

    
    ArrayOf(const Gdims& _dims, const device_id& dev=0): 
      k(_dims.size()), dims(_dims), strides(_dims.size()){
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      if(dev.id()==0){
	arr=new OBJ*[asize];
	for(int i=0; i<asize; i++)
	  arr[i]=new OBJ();
	device=0;
      }
      if(dev.id()==1){
	//CUDA_SAFE(cudaMalloc((void **)&arrg, asize*sizeof(TYPE)));
	//device=1;
      }
    }

    ArrayOf(std::function<OBJ(const int)> fn, const Gdims& _dims): 
      k(_dims.size()), dims(_dims), strides(_dims.size()){
      assert(k==1);
      allocate();
      for(int i=0; i<dims[0]; i++)
	arr[i]=new OBJ(fn(i));
    }
      
    template<typename ARG1>
    ArrayOf(const Gdims& _dims, const ARG1& arg1): 
      k(_dims.size()), dims(_dims), strides(_dims.size()){
      allocate();
      for(int i=0; i<asize; i++)
	arr[i]=new OBJ(arg1);
    }

    template<typename ARG1, typename ARG2>
    ArrayOf(const Gdims& _dims, const ARG1& arg1, const ARG2& arg2): 
      k(_dims.size()), dims(_dims), strides(_dims.size()){
      allocate();
      for(int i=0; i<asize; i++)
	arr[i]=new OBJ(arg1,arg2);
    }

    template<typename ARG1, typename ARG2, typename ARG3>
    ArrayOf(const Gdims& _dims, const ARG1& arg1, const ARG2& arg2, const ARG3& arg3): 
      k(_dims.size()), dims(_dims), strides(_dims.size()){
      allocate();
      for(int i=0; i<asize; i++)
	arr[i]=new OBJ(arg1,arg2,arg3);
    }

    template<typename ARG1, typename ARG2, typename ARG3, typename ARG4>
    ArrayOf(const Gdims& _dims, const ARG1& arg1, const ARG2& arg2, const ARG3& arg3, const ARG4& arg4): 
      k(_dims.size()), dims(_dims), strides(_dims.size()){
      allocate();
      for(int i=0; i<asize; i++)
	arr[i]=new OBJ(arg1,arg2,arg3,arg4);
    }

  private:

    void allocate(){
     strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      arr=new OBJ*[asize];
    }

    void reallocate(const Gdims& _dims){
      dims=_dims;
      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      delete[] arr;
      arr=new OBJ*[asize];
    }      


  public: // ---- Copying -------------------------------------------------------------------------------------


    ArrayOf(const ArrayOf<OBJ>& x): 
      ArrayOf<OBJ>(x.dims){
      for(int i=0; i<asize; i++)
	arr[i]=new OBJ(*x.arr[i]);
    }

    ArrayOf(ArrayOf<OBJ>&& x): 
      ArrayOf<OBJ>(x.dims){
      for(int i=0; i<asize; i++)
	arr[i]=x.arr[i];
      delete x.arr;
      x.asize=0;
    }

    ArrayOf<OBJ>& operator=(const ArrayOf<OBJ>& x){
      //for(int i=0; i<asize; i++)
      //delete arr[i];
      delete[] arr;
      reallocate(x.dims);
      for(int i=0; i<asize; i++)
	arr[i]=new OBJ(*x.arr[i]);
      return *this;
    }

    ArrayOf<OBJ>& operator=(ArrayOf<OBJ>&& x){
      delete[] arr;
      reallocate(x.dims);
      for(int i=0; i<asize; i++)
	arr[i]=x.arr[i];
      delete x.arr;
      x.arr=nullptr;
      x.asize=0;
      return *this;
    }

    

  public: // ---- Access --------------------------------------------------------------------------------------

    OBJ operator()(const int i) const{
      assert(k==1);
      return *arr[i];
    }

    OBJ& operator()(const int i){
      assert(k==1);
      return *arr[i];
    }

    OBJ operator()(const int i, const int j) const{
      assert(k==2);
      return *arr[i*strides[0]+j];
    }

    OBJ& operator()(const int i, const int j){
      assert(k==2);
      return *arr[i*strides[0]+j];
    }

    OBJ operator()(const int i, const int j, const int _k) const{
      assert(k==3);
      return *arr[i*strides[0]+j*strides[1]+_k];
    }

    OBJ& operator()(const int i, const int j, const int _k){
      assert(k==3);
      return *arr[i*strides[0]+j*strides[1]+_k];
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------

    /*
    ArrayOf(Bifstream& ifs){
      reallocate(ifs);
      for(int i=0; i<asize; i++) 
	arr[i]=new OBJ(ifs);
    }
    */

  };



}

#endif
