#ifndef _SO3_CGbank
#define _SO3_CGbank

#include <mutex>
#include "SO3_CGcoeffs.hpp"

#define CG_CMEM_DATA_OFFS 4096

#ifdef _WITH_CUDA
extern __device__ __constant__ unsigned char cg_cmem[]; 
#endif

namespace GEnet{

  class SO3_CGbank{
  private:

    unordered_map<CGindex,SO3_CGcoeffs<float>*> cgcoeffsf;
    unordered_map<CGindex,SO3_CGcoeffs<double>*> cgcoeffsd;
    unordered_map<CGindex,SO3_CGcoeffs<float>*> cgcoeffsfG;
    unordered_map<CGindex,SO3_CGcoeffs<double>*> cgcoeffsdG;
    unordered_map<CGindex,int> cgcoeffsfC;
    
    mutex safety_mx;
    int cmem_index_tail=0;
    int cmem_data_tail=CG_CMEM_DATA_OFFS;
  
  public:

    SO3_CGbank(){}
    
    SO3_CGbank(const SO3_CGbank& x)=delete;
    SO3_CGbank& operator=(const SO3_CGbank& x)=delete;
    
    ~SO3_CGbank(){
      for(auto p:cgcoeffsf) delete p.second;
      for(auto p:cgcoeffsd) delete p.second;
      for(auto p:cgcoeffsfG) delete p.second;
      for(auto p:cgcoeffsdG) delete p.second;
    }
    
    const SO3_CGcoeffs<float>& getf(const CGindex& ix, const device_id& dev=0){
      lock_guard<mutex> lock(safety_mx);
      if(dev.id()==0){
	auto it=cgcoeffsf.find(ix);
	if(it!=cgcoeffsf.end()) return *it->second;
	SO3_CGcoeffs<float>* r=new SO3_CGcoeffs<float>(ix);
	//lock_guard<mutex> lock(safety_mx);
	it=cgcoeffsf.find(ix);
	if(it!=cgcoeffsf.end()) return *it->second;
	cgcoeffsf[ix]=r;
	return *r;
      }else{
	auto it=cgcoeffsfG.find(ix);
	if(it!=cgcoeffsfG.end()) return *it->second;
	SO3_CGcoeffs<float>* r=new SO3_CGcoeffs<float>(getf(ix));
	r->to_device(dev);
	//lock_guard<mutex> lock(safety_mx);
	it=cgcoeffsfG.find(ix);
	if(it!=cgcoeffsfG.end()) return *it->second;
	cgcoeffsfG[ix]=r;
	return *r;
      }
    }

    const SO3_CGcoeffs<double>& getd(const CGindex& ix, const device_id& dev=0){
      if(dev.id()==0){
	auto it=cgcoeffsd.find(ix);
	if(it!=cgcoeffsd.end()) return *it->second;
	SO3_CGcoeffs<double>* r=new SO3_CGcoeffs<double>(ix);
	lock_guard<mutex> lock(safety_mx);
	it=cgcoeffsd.find(ix);
	if(it!=cgcoeffsd.end()) return *it->second;
	cgcoeffsd[ix]=r;
	return *r;
      }else{
	auto it=cgcoeffsdG.find(ix);
	if(it!=cgcoeffsdG.end()) return *it->second;
	SO3_CGcoeffs<double>* r=new SO3_CGcoeffs<double>(getd(ix));
	r->to_device(dev);
	lock_guard<mutex> lock(safety_mx);
	it=cgcoeffsdG.find(ix);
	if(it!=cgcoeffsdG.end()) return *it->second;
	cgcoeffsdG[ix]=r;
	return *r;
      }
    }

    int getfC(const int l1, const int l2, const int l){
      CGindex ix(l1,l2,l);
      auto it=cgcoeffsfC.find(ix);
      if(it!=cgcoeffsfC.end()) return it->second;
      const SO3_CGcoeffs<float>& coeffs=getf(ix);
#ifdef _WITH_CUDA
      //cout<<cmem_index_tail<<": "<<l1<<" "<<l2<<" "<<l<<endl;
      if(cmem_index_tail+4*sizeof(int)>CG_CMEM_DATA_OFFS){
	cerr<<"SO3_CGbank: no room to store index entry in constant memory."<<endl; exit(-1);}
      int ix_entry[4];
      ix_entry[0]=l1;
      ix_entry[1]=l2;
      ix_entry[2]=l;
      ix_entry[3]=cmem_data_tail;
      CUDA_SAFE(cudaMemcpyToSymbol(cg_cmem,reinterpret_cast<void*>(ix_entry),
	  4*sizeof(int),cmem_index_tail,cudaMemcpyHostToDevice));
      cmem_index_tail+=4*sizeof(int);
      cgcoeffsfC[ix]=cmem_data_tail; 
      if(cmem_data_tail+sizeof(float)*coeffs.asize>CG_CONST_MEM_SIZE){
	cerr<<"SO3_CGbank: no room to store CG matrix in constant memory."<<endl; exit(-1);}
      //cout<<l1<<l2<<l<<coeffs.arr[0]<<endl;
      CUDA_SAFE(cudaMemcpyToSymbol(cg_cmem,reinterpret_cast<void*>(coeffs.arr),
	  coeffs.asize*sizeof(float),cmem_data_tail,cudaMemcpyHostToDevice));
      int r=cmem_data_tail;
      cmem_data_tail+=sizeof(float)*coeffs.asize;
      return r;
#else
      NOCUDA_ERROR;
      return 0;
#endif 
    }

    template<class TYPE>
    const SO3_CGcoeffs<TYPE>& get(const int l1, const int l2, const int l);

    template<class TYPE>
    const SO3_CGcoeffs<TYPE>& getG(const int l1, const int l2, const int l);

  };


  
  template<>
  inline const SO3_CGcoeffs<float>& SO3_CGbank::get<float>(const int l1, const int l2, const int l){
    return getf(CGindex(l1,l2,l));}
  
  template<>
  inline const SO3_CGcoeffs<double>& SO3_CGbank::get<double>(const int l1, const int l2, const int l){
    return getd(CGindex(l1,l2,l));}
  
  template<>
  inline const SO3_CGcoeffs<float>& SO3_CGbank::getG<float>(const int l1, const int l2, const int l){
    return getf(CGindex(l1,l2,l),device_id(1));}
  
  template<>
  inline const SO3_CGcoeffs<double>& SO3_CGbank::getG<double>(const int l1, const int l2, const int l){
    return getd(CGindex(l1,l2,l),device_id(1));}
  
    

} 

#endif
