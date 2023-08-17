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
#ifndef _CtensorB
#define _CtensorB

#include "CFtensor.hpp"
#include "Cobject.hpp"
#include "RscalarB.hpp"
#include "CscalarB.hpp"
#include "CellAccessor.hpp"


namespace Cengine{

  class CtensorB: public Cobject, public CFtensor{
  public:

    Gdims dims; 
    int nbu=-1;

    CtensorB(){
      CTENSORB_CREATE();
    }

    ~CtensorB(){
      CTENSORB_DESTROY();
    }

    string classname() const{
      return "CtensorB";
    }

    string describe() const{
      if(nbu>=0) return "CtensorB"+dims.str()+"["+to_string(nbu)+"]";
      return "CtensorB"+dims.str();
    }



  public: // ---- Constructors ------------------------------------------------------------------------------

    
    CtensorB(const Gtensor<complex<float> >& x, const int dev=0): 
      CFtensor(x), dims(x.dims), nbu(-1){
      CTENSORB_CREATE();
    }

    CtensorB(const Gtensor<complex<float> >& x, const device_id& dev=0): 
      CFtensor(x), dims(x.dims), nbu(-1){
      CTENSORB_CREATE();
    }


  public: // ---- Filled constructors -----------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorB(const Gdims& _dims, const FILLTYPE& fill, const int dev=0):
      CFtensor(_dims,fill,dev), dims(_dims){
      CTENSORB_CREATE();
    }
	  
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorB(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int dev=0):
      CFtensor(_dims.prepend(_nbu),fill,dev), dims(_dims), nbu(_nbu){
      CTENSORB_CREATE();
    }
	  
    CtensorB(const Gdims& _dims, const int _nbu, const fill_gaussian& fill, const float c, const int dev=0):
      CFtensor(_dims.prepend(_nbu),fill,c,dev), dims(_dims), nbu(_nbu){
      CTENSORB_CREATE();
    }
	  
    CtensorB(const Gdims& _dims, const int _nbu, std::function<complex<float>(const int i, const int j)> fn, const int dev=0):
      CFtensor(_dims.prepend(_nbu),fill::raw), dims(_dims), nbu(_nbu){
      if(nbu==-1){
	for(int i=0; i<dims[0]; i++)
	  for(int j=0; j<dims[1]; j++)
	    CFtensor::set(i,j,fn(i,j));
      }else{
	for(int b=0; b<nbu; b++)
	  for(int i=0; i<dims[0]; i++)
	    for(int j=0; j<dims[1]; j++)
	      CFtensor::set(b,i,j);
      }
      if(dev>0) to_device(dev);
      CTENSORB_CREATE();
    }
	  
    CtensorB(const CtensorB& x, std::function<complex<float>(const complex<float>)> fn):
      CFtensor(x,fn), dims(x.dims){
      CTENSORB_CREATE();
    }

    CtensorB(const CtensorB& x, std::function<complex<float>(const int i, const int j, const complex<float>)> fn):
      CFtensor(x,fill::raw), dims(x.dims){
      assert(dims.size()==2);
      if(nbu==-1){
	for(int i=0; i<dims[0]; i++)
	  for(int j=0; j<dims[1]; j++)
	    CFtensor::set(i,j,fn(i,j,x.CFtensor::get(i,j)));
      }else{
	for(int b=0; b<nbu; b++)
	  for(int i=0; i<dims[0]; i++)
	    for(int j=0; j<dims[1]; j++)
	      CFtensor::set(b,i,j,fn(i,j,x.CFtensor::get(b,i,j)));
      }
      CTENSORB_CREATE();
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorB(const CtensorB& x): 
      CFtensor(x), dims(x.dims), nbu(x.nbu){
      COPY_WARNING;
      CTENSORB_CREATE();
    }

    CtensorB(const CtensorB& x, const nowarn_flag& dummy): 
      CFtensor(x,dummy), dims(x.dims), nbu(x.nbu){
      CTENSORB_CREATE();
    }

    CtensorB* clone() const{
      return new CtensorB(*this, nowarn);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CtensorB(const CFtensor& x):
      CFtensor(x), dims(x.dims), nbu(-1){
      CTENSORB_CREATE();
    }

    CtensorB(CFtensor&& x):
      CFtensor(std::move(x)), dims(x.dims), nbu(-1){
      CTENSORB_CREATE();
    }

    void to_device(const int _dev) const{
      CFtensor::to_device(_dev);
    }



  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }

    Gdims get_dims() const{
      return dims;
    }

    int get_device() const{
      return device;
    }

    CtensorAccessor<CtensorB> accessor(){
      return CtensorAccessor<CtensorB>(arr,arrc,strides);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorB* conj() const{
      return new CtensorB(CFtensor::conj());
    }

    CtensorB* transp() const{
      return new CtensorB(CFtensor::transp());
    }

    CtensorB* herm() const{
      return new CtensorB(CFtensor::herm());
    }


    CtensorB* divide_cols(const CtensorB& N) const{
      return new CtensorB(CFtensor::divide_cols(N));
    }

    CtensorB* normalize_cols() const{
      return new CtensorB(CFtensor::normalize_cols());
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_prod(const RscalarB& c, const CtensorB& A){
      if(c.nbu==-1){
	CFtensor::add(A,c.val);
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_prod(const CscalarB& c, const CtensorB& A){
      if(c.nbu==-1){
	CFtensor::add(A,c.val);
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_prod_cconj(const CscalarB& c, const CtensorB& A){
      if(c.nbu==-1){
	CFtensor::add(A,std::conj(c.val));
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_prod_c_times_conj(const CscalarB& c, const CtensorB& A){
      if(c.nbu==-1){
	CFtensor::add_conj(A,c.val);
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_inp_into(CscalarB& r, const CtensorB& A){
      if(nbu==-1){
	r.val+=inp(A);
      }else{
	FCG_UNIMPL();
      }
    }

    void add_element_into(CscalarB& r, const Gindex& ix){
      if(nbu==-1){
	r.val+=get(ix);
      }else{
	FCG_UNIMPL();
      }
    }

    void add_to_element(const Gindex& ix, CscalarB& r){
      assert(nbu==-1);
      if(nbu==-1){
	inc(ix,r.val);
      }else{
	FCG_UNIMPL();
      }
    }

    void mix_into(CscalarB& r, const CscalarB& x) const{
      to_device(0);
      assert(dims.size()==2);
      if(r.nbu==-1){
	assert(dims[0]==1);
	if(x.nbu==-1){
	  assert(dims[1]==1);
	  r.val+=complex<float>(arr[0],arrc[0])*x.val;
	  return; 
	}else{
	  assert(dims[1]==x.nbu);
	  for(int i=0; i<x.nbu; i++)
	    r.val+=complex<float>(arr[i],arrc[i])*x.arr[i];
	  return;
	}
      }else{
	assert(dims[0]==r.nbu);
	if(x.nbu==-1){
	  assert(dims[1]==1);
	  for(int i=0; i<r.nbu; i++)
	    r.arr[i]+=complex<float>(arr[i],arrc[i])*x.val;
	}else{
	  assert(dims[1]==x.nbu);
	  for(int i=0; i<r.nbu; i++){
	    complex<float> t=r.arr[i];
	    for(int j=0; j<x.nbu; j++)
	      t+=complex<float>(arr[i*x.nbu+j],arrc[i*x.nbu+j])*x.val;
	    r.arr[i]=t;
	  }
	}
      }
    }

    void mix_into(CtensorB& r, const CtensorB& x) const{
      to_device(0);
      assert(dims.size()==2);
      if(r.nbu==-1){
	assert(dims[0]==1);
	if(x.nbu==-1){
	  assert(dims[1]==1);
	  r.add(x,complex<float>(arr[0],arrc[0]));
	  return; 
	}else{
	  assert(dims[1]==x.nbu);
	  FCG_UNIMPL();
	  return;
	}
      }else{
	FCG_UNIMPL();
      }
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    //string str(const string indent="") const{
    //return Gtensor<complex<float> >(*this);
    //}

   
  };


  inline CtensorB& asCtensorB(Cobject* x, const char* s){
    return downcast<CtensorB>(x,s);
  }

  inline CtensorB& asCtensorB(Cnode* x, const char* s){
    return downcast<CtensorB>(x,s);
  }
  
  inline CtensorB& asCtensorB(Cnode& x, const char* s){
    return downcast<CtensorB>(x,s);
  }


#define CTENSORB(x) asCtensorB(x,__PRETTY_FUNCTION__) 


}

#endif

    /*
    CtensorB(const string filename, const device_id& dev=0):
      CFtensor(filename,dev){}

    int save(const string filename) const{
      CFtensor::save(filename);
      return 0;
    }

    CtensorB(Bifstream& ifs): 
      CFtensor(ifs){
    }

    void serialize(Bofstream& ofs){
      CFtensor::serialize(ofs);
    }
    */

  //inline CtensorB& asCtensorB(Cobject* x){
  //return downcast<CtensorB>(x,"");
  //}

  //inline CtensorB& asCtensorB(Cnode* x){
  //return downcast<CtensorB>(x,"");
  //}

  //inline CtensorB& asCtensorB(Cnode& x){
  //return downcast<CtensorB>(x,"");
  //}

