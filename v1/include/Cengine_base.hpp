#ifndef _Cengine_base
#define _Cengine_base

#include <assert.h>
#include <pthread.h>

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <complex>
#include <iostream>
#include <unordered_map>
#include <random>
#include <functional> 
#include <thread>
#include <array>
#include <set>
#include <algorithm>

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

using namespace std; 


//#define COPY_WARNING
#define COPY_WARNING cout<<"\e[1mWarning:\e[0m "<<classname()<<" copied."<<endl;

//#define ASSIGN_WARNING
#define ASSIGN_WARNING cout<<"\e[1mWarning:\e[0m "<<classname()<<" assigned."<<endl;

//#define CONVERT_WARNING(a,b)
#define CONVERT_WARNING(a,b) cout<<"\e[1mWarning:\e[0m "<<a<<" converted to "<<b<<"."<<endl;

#define NOCUDA_ERROR cout<<"Error: Cengine was compiled without GPU support."<<endl;

#ifndef _WITH_CUDA
#define cudaMalloc(a,b) NOCUDA_ERROR
#define cudaMemcpy(a,b,c,d) NOCUDA_ERROR
#define cudaMemset(a,b,c) NOCUDA_ERROR
#define cudaFree(a) NOCUDA_ERROR
#endif

#define FCG_ASSERT(condition, message) \
    if (!(condition)) {cout<<message<<endl; assert ((condition)); exit(-1); }

#define FCG_UNIMPL() printf("Cengine error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);
#define FCG_NOTIMPL() printf("Cengine error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);
#define FCG_CPUONLY() if(device>0) {printf("Cengine error: CUDA code for \"%s\" not implemented.\n",__PRETTY_FUNCTION__); exit(-1);}

#define CENGINE_DEPRECATED() printf("Cengine warning: function \"%s\" is deprecated.\n",__PRETTY_FUNCTION__);
//#define FCG_DEPRECATED(message) printf("Warning: %s is deprecated.\n",(message));

#define FCG_WARNING(message) printf("Warning: %s.\n",(message));

#define GENET_UNIMPL() printf("Cengine error: function \"%s\" not implemented.\n",__PRETTY_FUNCTION__);

#define COUT(cmd) {CoutLock lk; cout<<cmd<<endl;}

#ifdef DEBUG_ENGINE_FLAG
#define DEBUG_ENGINE(cmd) cmd;
#define DEBUG_ENGINE2(cmd) {CoutLock lk; cout<<cmd<<endl;}
#else
#define DEBUG_ENGINE(cmd);
#define DEBUG_ENGINE2(cmd);
#endif 

#ifdef DEBUG_FLUSH_FLAG
#define DEBUG_FLUSH(cmd) cmd;
#define DEBUG_FLUSH2(cmd) {CoutLock lk; cout<<cmd<<endl;}
#else
#define DEBUG_FLUSH(cmd);
#define DEBUG_FLUSH2(cmd);
#endif 

//const int explicitL=3;
//const int inline_explicitL=5;

#define CG_CONST_MEM_SIZE 32276


#ifdef CENGINE_OBJ_COUNT
#define CNODE_CREATE() ::Cengine::Cnode_count++; //cout<<::Cengine::Cnode_count<<endl;
#define CNODE_DESTROY() ::Cengine::Cnode_count--; //cout<<::Cengine::Cnode_count<<endl;
#define CHANDLE_CREATE() ::Cengine::Chandle_count++; //cout<<::Cengine::Chandle_count<<endl;
#define CHANDLE_DESTROY() ::Cengine::Chandle_count--; //cout<<::Cengine::Chandle_count<<endl;
#define COPERATOR_CREATE() ::Cengine::Coperator_count++; //cout<<::Cengine::Coperator_count<<endl;
#define COPERATOR_DESTROY() ::Cengine::Coperator_count--; //cout<<::Cengine::Coperator_count<<endl;
#define RSCALARB_CREATE() ::Cengine::RscalarB_count++; //cout<<::Cengine::RscalarB_count<<endl;
#define RSCALARB_DESTROY() ::Cengine::RscalarB_count--; //cout<<::Cengine::RscalarB_count<<endl;
#define CSCALARB_CREATE() ::Cengine::CscalarB_count++; //cout<<::Cengine::CscalarB_count<<endl;
#define CSCALARB_DESTROY() ::Cengine::CscalarB_count--; //cout<<::Cengine::CscalarB_count<<endl;
#define CTENSORB_CREATE() ::Cengine::CtensorB_count++; // {CoutLock lk; cout<<"CtensorB("<<dims<<","<<nbu<<","<<device<<")"<<endl;} 
#define CTENSORB_DESTROY() ::Cengine::CtensorB_count--; //cout<<::Cengine::CtensorB_count<<endl;
#else
#define CNODE_CREATE();
#define CNODE_DESTROY();
#define CHANDLE_CREATE();
#define CHANDLE_DESTROY();
#define COPERATOR_CREATE();
#define COPERATOR_DESTROY();
#define RSCALARB_CREATE();
#define RSCALARB_DESTROY();
#define CSCALARB_CREATE();
#define CSCALARB_DESTROY();
#define CTENSORB_CREATE();
#define CTENSORB_DESTROY();
#endif 

#ifdef CENGINE_TRACEBACK_FLAG
#define CENGINE_TRACE(msg) {traceback(msg);}
#define CENGINE_ASSERT(condition)				\
  if(!(condition)) {traceback.dump(); assert(condition);}
#define CENGINE_DUMP_TRACE() traceback.dump();
#else
#define CENGINE_ASSERT(condition);
#define CENGINE_TRACE(msg);
#define CENGINE_DUMP_TRACE(); 
#endif



namespace Cengine{

  namespace engine{}; 


  // ---- Fill ----------------------------------------------------------------------------------------------


  struct fill_pattern{};
  struct fill_noalloc: public fill_pattern {fill_noalloc(){}};
  struct fill_raw: public fill_pattern {fill_raw(){}};
  struct fill_zero: public fill_pattern{fill_zero(){}};
  struct fill_fn: public fill_pattern{fill_fn(){}};
  struct fill_ones: public fill_pattern{fill_ones(){}};
  struct fill_sequential: public fill_pattern{fill_sequential(){}};
  struct fill_identity: public fill_pattern{fill_identity(){}};
  struct fill_uniform: public fill_pattern{fill_uniform(){}};
  struct fill_tensor: public fill_pattern{fill_tensor(){}};

  struct fill_gaussian: public fill_pattern{
  public:
    float c=1.0;
    fill_gaussian(){}
    explicit fill_gaussian(const float _c): c(_c){}
    fill_gaussian operator()(const float _c) const {return fill_gaussian(_c);}
  };

  struct fill_cgaussian: public fill_pattern{fill_cgaussian(){}};

  struct fill_bernoulli: public fill_pattern{
    double p=0.5;
    fill_bernoulli(){}
    fill_bernoulli(const double _p):p(_p){}
  };
  
  struct fill_symm_bernoulli: public fill_pattern{
    double p=0.5;
    fill_symm_bernoulli(){}
    fill_symm_bernoulli(const double _p):p(_p){}};
  template<typename TYPE> 
  struct fill_const: public fill_pattern{
    TYPE p=0;
    fill_const(){}
    fill_const(const TYPE _p):p(_p){}
  };
  struct fill_stack: public fill_pattern{fill_stack(){}};
  struct fill_cat: public fill_pattern{fill_cat(){}};

  namespace fill{
    static const fill_noalloc noalloc;
    static const fill_raw raw; // 0
    static const fill_zero zero; // 1
    static const fill_fn fn; 
    static const fill_ones ones; // 2 
    static const fill_sequential sequential; //3 
    static const fill_identity identity; //4 
    static const fill_uniform uniform; //5 
    static const fill_tensor tensor; //5 
    static const fill_bernoulli bernoulli; //6 
    static const fill_symm_bernoulli symm_bernoulli; //7
    static const fill_gaussian gaussian; //8
    static const fill_cgaussian cgaussian;
    static const fill_stack stack;
    static const fill_cat cat;
  }

  
  // ---- Channels and bundles ------------------------------------------------------------------------------


  class nchannels{
  public:
    const int val;
    nchannels(const int _val): val(_val){}
  };
  
  class channel{
  public:
    const int val;
    channel(const int _val): val(_val){}
  };
    

  class nbundle{
  public:
    const int val;
    nbundle(const int _val): val(_val){}
  };
  
  class bundle{
  public:
    const int val;
    bundle(const int _val): val(_val){}
  };


  // ---- Multithreading ------------------------------------------------------------------------------------


  class CoutLock{
  public:
    CoutLock(): lock(mx){}
    lock_guard<mutex> lock;
    static mutex mx;
  };


  // ---- Other flags ---------------------------------------------------------------------------------------


  struct view_flag{};

  namespace flag{
    static const view_flag view;
  }

  struct uninitialized_flag{};


  struct destroy_flag{};
  static const destroy_flag destroy;

  struct nowarn_flag{};
  static const nowarn_flag nowarn;

  /*
  template<typename RET, typename... TYPES>
  struct functn_wrapper{
    std::function<RET(TYPES...)>& fn;
    functn_wrapper(std::function<RET(TYPES...)>& _fn): fn(_fn){}
  };

  template<typename RET, typename... TYPES>
  functn_wrapper<RET,TYPES...>
  functn(std::function<RET(const TYPES...)>& fn){
    return functn_wrapper<RET,TYPES...>(fn);
  }
  */

  // --- Devices ---------------------------------------------------------------------------------------------


  struct device_id{
    int _id;
    device_id(const int x): _id(x){};
    int id() const {return _id;}
  };

  namespace device{
    static device_id CPU(0);
    static device_id GPU0(1);
  }    


  // --- Variadics -------------------------------------------------------------------------------------------


  template<class TYPE, typename... Args>
  vector<TYPE*> variadic_unroller(TYPE& x, Args&... args){
    vector<TYPE*> argv;
    variadic_unroller_sub(argv, x, args...);
    return argv;}

  template<class TYPE, typename... Args>
  void variadic_unroller_sub(vector<TYPE*>& argv, TYPE& x, Args&... args){
    argv.push_back(&x);
    variadic_unroller_sub(argv, args...);}

  template<class TYPE, typename... Args>
  void variadic_unroller_sub(vector<TYPE*>& argv, TYPE& x){
    argv.push_back(&x);}


  template<class TYPE, typename... Args>
  void const_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE& x){
    argv.push_back(&x);}

  template<class TYPE, typename... Args>
  void const_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE& x, Args&... args){
    argv.push_back(&x);
    const_variadic_unroller_sub(argv, args...);}

  template<class TYPE, typename... Args>
  vector<const TYPE*> const_variadic_unroller(const TYPE& x, Args&... args){
    vector<const TYPE*> argv;
    const_variadic_unroller_sub(argv, x, args...);
    return argv;}


  template<class TYPE, class TYPE2, typename... Args>
  void const_derived_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE2& x, Args&... args){
    argv.push_back(&x);
    const_derived_variadic_unroller_sub(argv, args...);
  }

  template<class TYPE, class TYPE2> 
  void const_derived_variadic_unroller_sub(vector<const TYPE*>& argv, const TYPE2& x){
    argv.push_back(&x);
  }

  template<class TYPE, typename... Args>
  vector<const TYPE*> const_derived_variadic_unroller(Args&... args){
    vector<const TYPE*> argv;
    const_derived_variadic_unroller_sub<TYPE>(argv, args...);
    return argv;
  }

  template <std::size_t I, typename T, typename... Ts>
  struct nth_element_impl{
    using type=typename nth_element_impl<I-1, Ts...>::type;
  };

  template <typename T, typename... Ts>
  struct nth_element_impl<0, T, Ts...>{
    using type=T;
  };


  // --- CONVENIENCE ---------------------------


  template<typename OBJ>
  int moveToDeviceIfAnyOnDevice(const vector<OBJ*>& x){
    int dev=0; for(auto p:x) if(p->device) {dev=1; break;}
    if(dev) for(auto p:x) const_cast<OBJ*>(p)->to_device();
    return dev;
  }

  template<typename OBJ>
  int moveValuesToDeviceIfAnyOnDevice(const vector<OBJ*>& x){
    int dev=0; for(auto p:x) if(p->value->device) {dev=1; break;}
    if(dev) for(auto p:x) const_cast<OBJ*>(p)->value->to_device();
    return dev;
  }

  template<typename OBJ>
  int moveToDeviceIfAnyOnDevice(const OBJ& x, const OBJ& y){
    int dev=0; if(x.device || y.device) dev=1;
    if(dev){
      const_cast<OBJ&>(x).to_device();
      const_cast<OBJ&>(y).to_device();}
    return dev;
  }

  template<typename OBJ>
  int moveToDeviceIfAnyOnDevice(const OBJ& x, const OBJ& y, const device_id& _dev){
    int dev=0; if(x.device || y.device) dev=1;
    if(dev){
      const_cast<OBJ&>(x).move_to(_dev);
      const_cast<OBJ&>(y).move_to(_dev);}
    return dev;
  }

  inline int roundup(const int x, const int s){
    return ((x-1)/s+1)*s;
  }

  inline int roundup(const int x){
    return ((x-1)/32+1)*32;
  }

  //template<typename TYPE1, typename TYPE2>
  //inline TYPE2 safe_cast(TYPE1 x){
  //assert(dynamic_cast<TYPE2>(x));
  //return static_cast<TYPE2>(x);
  //}

  //template<typename TYPE>
  //class CtensorA;

  template<typename TYPE1, typename TYPE2>
  vector<TYPE2> apply(const vector<TYPE1> v,std::function<TYPE2(TYPE1)> lambda){
    const int n=v.size();
    vector<TYPE2> w(n);
    for(int i=0; i<n; i++)
      w[i]=lambda(v[i]);
    return w;
  }


  template<typename TYPE1, typename TYPE2>
  bool multi_or(TYPE2& v, std::function<bool(TYPE1)> lambda){
    bool b=false; 
    for(auto p: v)
      if(lambda(p)){b=true; break;}
    return b;
  }


  // --------------------------------------


  template<class TYPE>
  class _view{
  public:
    const TYPE& obj;
    _view(const TYPE& _obj): obj(_obj){};
    operator TYPE() const{return obj;}
  };
    
  template<class TYPE>
  _view<TYPE> view(const TYPE& obj){return obj;}

     
  template<class TYPE>
  TYPE copy_of(const TYPE& obj){
    return TYPE(obj,nowarn);
  }


  template<class TYPE>
  class _like{
  public:
    const TYPE& obj;
    _like(const TYPE& _obj): obj(_obj){};
    operator TYPE() const{return obj;}
  };
    
  template<class TYPE>
  _like<TYPE> like(const TYPE& obj){return obj;}


  template<class TYPE>
  class SelfPointer{
  public:
    TYPE* ptr;
    SelfPointer(TYPE* x): ptr(x){}
    SelfPointer& operator=(TYPE* x){ptr=x; return *this;}

    //operator TYPE() {return *ptr;}
    operator TYPE*() const {return ptr;}
    TYPE* operator->(){return ptr;}
  };

}

template<typename TYPE>
inline void fastadd(const TYPE* source, TYPE* dest, const int n){
  for(int i=0; i<n; i++)
    *(dest+i)+=*(source+i);
}

template<typename TYPE>
inline TYPE ifthen(const bool p, const TYPE& x, const TYPE& y){
  if(p) return x; else return y;
}


template<class TYPE>
inline ostream& operator<<(ostream& stream, const vector<TYPE>& x){
  //stream<<"(";
  for(int i=0; i<x.size(); i++){
    stream<<x[i]; 
    if(i+1<x.size()) stream<<",";
  }
  //stream<<")";
  return stream;
}


template<typename TYPE>
inline void print(const TYPE& x){
  cout<<x.str()<<endl;
}

template<typename TYPE>
inline ostream& print(const string name, const TYPE& x){
  cout<<name<<"="<<x.str()<<endl;
  return cout; 
}

template<typename TYPE>
inline ostream& printl(const string name, const TYPE& x){
  cout<<name<<"="<<endl<<x.str()<<endl;
  return cout; 
}

template<typename TYPE>
inline void printv(const vector<TYPE>& x){
  cout<<"[ ";
  for(auto p:x) cout<<p<<" ";
  cout<<"]"<<endl;
}


//#include "Cobject.hpp"
//namespace Cengine{
//class Cnode; 
//}


// ---- CUDA STUFF ------------------------------------------------------------------------------------------


#ifdef _WITH_CUDA
#define IFCUDA(cmds) cmds 
#else 
#define IFCUDA(cmds) 
#endif


#ifdef _WITH_CUDA
#define CUDA_SAFE(err) __cudaSafeCall( err, __FILE__, __LINE__ );
inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  if(cudaSuccess!=err){
    fprintf(stderr,"cudaSafeCall() failed at %s:%i : %s\n",file,line,cudaGetErrorString(err));
    exit(-1);}
  return;
}
#else 
#define CUDA_SAFE(err) err 
#endif 


#ifdef _WITH_CUBLAS
#define CUBLAS_SAFE(expression) {			     \
    cublasStatus_t status= (expression);		     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "CuBLAS error on line " << __LINE__ << ": ";		\
    if(status==CUBLAS_STATUS_SUCCESS) fprintf(stderr,"CUBLAS SUCCESS"); \
    else if(status==CUBLAS_STATUS_NOT_INITIALIZED) \
        fprintf(stderr,"'CUBLAS_STATUS_NOT_INITIALIZED'"); \
    else if(status==CUBLAS_STATUS_ALLOC_FAILED)\
        fprintf(stderr,"'CUBLAS_STATUS_ALLOC_FAILED'");\
    else if(status==CUBLAS_STATUS_INVALID_VALUE)\
        fprintf(stderr,"'CUBLAS_STATUS_INVALID_VALUE'");\
    else if(status==CUBLAS_STATUS_ARCH_MISMATCH)\
        fprintf(stderr,"'CUBLAS_STATUS_ARCH_MISMATCH'");\
    else if(status==CUBLAS_STATUS_MAPPING_ERROR)\
        fprintf(stderr,"'CUBLAS_STATUS_MAPPING_ERROR'");\
    else if(status==CUBLAS_STATUS_EXECUTION_FAILED)\
        fprintf(stderr,"'CUBLAS_STATUS_EXECUTION_FAILED'");\
    else if(status==CUBLAS_STATUS_INTERNAL_ERROR)\
        fprintf(stderr,"'CUBLAS_STATUS_INTERNAL_ERROR'");\
    else						 \
      fprintf(stderr,"UNKNOWN CUBLAS ERROR");\
    std::exit(EXIT_FAILURE);				     \
    }                                                        \
  }
#else 
#define CUBLAS_SAFE(expression) //expression  
#endif 


#ifdef _WITH_CUDA
#define CUDNN_SAFE(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
#else 
#define CUDNN_SAFE(expression) expression  
#endif 


template<typename TYPE>
class pullin{
public:
  TYPE& obj;
  int device;
  pullin(TYPE& _obj): obj(_obj){
    device=obj.device;
    obj.to_device(0);
  }
  pullin(const TYPE& _obj): obj(const_cast<TYPE>(_obj)){
    device=obj.device;
    obj.to_device(0);
  }
  ~pullin(){
    obj.to_device(device);
  }
};


template<typename TYPE>
class pullin2{
public:
  TYPE& obj0;
  TYPE& obj1;
  int device0;
  int device1;
  pullin2(const TYPE& _obj0, const TYPE& _obj1): 
    obj0(const_cast<TYPE&>(_obj0)),
    obj1(const_cast<TYPE&>(_obj1)){
    device0=obj0.device;
    device1=obj1.device;
    obj0.to_device(0);
    obj1.to_device(0);
  }
  ~pullin2(){
    obj0.to_device(device0);
    obj1.to_device(device1);
  }
};

namespace Cengine{

  template<typename T, typename U>
  struct is_same : std::false_type { };
  
  template<typename T>
  struct is_same<T, T> : std::true_type { };
  
  template<typename T, typename U>
  constexpr bool eqTypes() { return is_same<T, U>::value; }
  
}


//template<typename TYPE>
//allGPUornone(TYPE& x, TYPE& y){
//if()
//}


#define SO3PART_CGPRODUCTOP_INDEX 1;
#define SO3PART_CGPRODUCTOPBACK0_INDEX 2;
#define SO3PART_CGPRODUCTOPBACK1_INDEX 3;


#endif


    //if(!dynamic_cast<const TYPE*>(&x)) printf("In function \"%s\" \n",__PRETTY_FUNCTION__);
