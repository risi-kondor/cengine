#ifndef _CtensorBarray
#define _CtensorBarray

#include "CFtensor.hpp"
#include "Cobject.hpp"
#include "RscalarB.hpp"
#include "CscalarB.hpp"


namespace Cengine{

  class CtensorBarray: public Cobject, public CFtensorArray{
  public:

    Gdims adims; 
    Gdims dims; 
    int nbu=-1;

    CtensorBarray(){
      CTENSORBARRAY_CREATE();
    }

    ~CtensorBarray(){
      CTENSORBARRAY_DESTROY();
    }

    string classname() const{
      return "CtensorBarray";
    }

    string describe() const{
      if(nbu>=0) return "CtensorBarray"+adims.str()+" "+dims.str()+"["+to_string(nbu)+"]";
      return "CtensorBarray"+adims.str()+" "+dims.str();
    }



  public: // ---- Constructors ------------------------------------------------------------------------------

    /*
    CtensorBarray(const Gtensor<complex<float> >& x, const int dev=0): 
      CFtensor(x), dims(x.dims), nbu(-1){
      CTENSORBARRAY_CREATE();
    }
    */

    /*
    CtensorBarray(const Gtensor<complex<float> >& x, const device_id& dev=0): 
      CFtensor(x), dims(x.dims), nbu(-1){
      CTENSORBARRAY_CREATE();
    }
    */


  public: // ---- Filled constructors -----------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorBarray(const Gdims& _adims, const Gdims& _dims, const FILLTYPE& fill, const int dev=0):
      CFtensorArray(_adims,_dims,fill,dev), adims(_adims), dims(_dims){
      CTENSORBARRAY_CREATE();
    }
	  
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorBarray(const Gdims& _adims, const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int dev=0):
      CFtensorArray(_adims,_dims.prepend(_nbu),fill,dev), adims(_adims), dims(_dims), nbu(_nbu){
      CTENSORBARRAY_CREATE();
    }
	  
    CtensorBarray(const Gdims& _adims, const Gdims& _dims, const int _nbu, const fill_gaussian& fill, const float c, const int dev=0):
      CFtensorArray(_adims,_dims.prepend(_nbu),fill,c,dev), adims(_adims), dims(_dims), nbu(_nbu){
      CTENSORBARRAY_CREATE();
    }

    /*
    CtensorBarray(const Gdims& _adims, const Gdims& _dims, const int _nbu, std::function<complex<float>(const int i, const int j)> fn, const int dev=0):
      CFtensor(_dims.prepend(_nbu),fill::raw), adims(_adims), dims(_dims), nbu(_nbu){
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
      CTENSORBARRAY_CREATE();
    }
    */
	
    /*
    CtensorBarray(const CtensorBarray& x, std::function<complex<float>(const complex<float>)> fn):
      CFtensor(x,fn), adims(x.adims), dims(x.dims){
      CTENSORBARRAY_CREATE();
    }
    */

    CtensorBarray(const CtensorBarray& x, std::function<complex<float>(const int i, const int j, const complex<float>)> fn):
      CFtensor(x,fill::raw), adims(x.adims), dims(x.dims){
      /*
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
      */
      CTENSORBARRAY_CREATE();
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorBarray(const CtensorBarray& x): 
      CFtensor(x), dims(x.dims), nbu(x.nbu){
      COPY_WARNING;
      CTENSORBARRAY_CREATE();
    }

    CtensorBarray(const CtensorBarray& x, const nowarn_flag& dummy): 
      CFtensor(x,dummy), dims(x.dims), nbu(x.nbu){
      CTENSORBARRAY_CREATE();
    }

    CtensorBarray* clone() const{
      return new CtensorBarray(*this, nowarn);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CtensorBarray(const CFtensor& x):
      CFtensor(x), dims(x.dims), nbu(-1){
      CTENSORBARRAY_CREATE();
    }

    CtensorBarray(CFtensor&& x):
      CFtensor(std::move(x)), dims(x.dims), nbu(-1){
      CTENSORBARRAY_CREATE();
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


  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorBarray* conj() const{
      return new CtensorBarray(CFtensor::conj());
    }

    CtensorBarray* transp() const{
      return new CtensorBarray(CFtensor::transp());
    }

    CtensorBarray* herm() const{
      return new CtensorBarray(CFtensor::herm());
    }


    CtensorBarray* divide_cols(const CtensorBarray& N) const{
      return new CtensorBarray(CFtensor::divide_cols(N));
    }

    CtensorBarray* normalize_cols() const{
      return new CtensorBarray(CFtensor::normalize_cols());
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_prod(const RscalarB& c, const CtensorBarray& A){
      if(c.nbu==-1){
	CFtensor::add(A,c.val);
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_prod(const CscalarB& c, const CtensorBarray& A){
      if(c.nbu==-1){
	CFtensor::add(A,c.val);
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_prod_cconj(const CscalarB& c, const CtensorBarray& A){
      if(c.nbu==-1){
	CFtensor::add(A,std::conj(c.val));
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_prod_c_times_conj(const CscalarB& c, const CtensorBarray& A){
      if(c.nbu==-1){
	CFtensor::add_conj(A,c.val);
      }else{
	FCG_UNIMPL();
      }
    }
 
    void add_inp_into(CscalarB& r, const CtensorBarray& A){
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

    void mix_into(CtensorBarray& r, const CtensorBarray& x) const{
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


  inline CtensorBarray& asCtensorBarray(Cobject* x, const char* s){
    return downcast<CtensorBarray>(x,s);
  }

  inline CtensorBarray& asCtensorBarray(Cnode* x, const char* s){
    return downcast<CtensorBarray>(x,s);
  }
  
  inline CtensorBarray& asCtensorBarray(Cnode& x, const char* s){
    return downcast<CtensorBarray>(x,s);
  }


#define CTENSORBARRAY(x) asCtensorBarray(x,__PRETTY_FUNCTION__) 


}

#endif

