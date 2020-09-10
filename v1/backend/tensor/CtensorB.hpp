#ifndef _CtensorB
#define _CtensorB

#include "CFtensor.hpp"
#include "Cobject.hpp"
#include "RscalarB.hpp"
#include "CscalarB.hpp"


namespace Cengine{

  class CtensorB: public Cobject, public CFtensor{
  public:

    Gdims dims; 
    int nbu=-1;

    CtensorB(){}

    string classname() const{
      return "CtensorB";
    }

    string describe() const{
      if(nbu>=0) return "CtensorB"+dims.str()+"["+to_string(nbu)+"]";
      return "CtensorB"+dims.str();
    }



  public: // ---- Constructors ------------------------------------------------------------------------------

    
    CtensorB(const Gtensor<complex<float> >& x, const device_id& dev=0): 
      CFtensor(x), dims(x.dims), nbu(-1){}


  public: // ---- Filled constructors -----------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorB(const Gdims& _dims, const FILLTYPE& fill, const device_id& dev=0):
      CFtensor(_dims,fill,dev), dims(_dims){}
	  
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorB(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const device_id& dev=0):
      CFtensor(_dims.prepend(_nbu),fill,dev), dims(_dims), nbu(_nbu){}
	  

  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorB(const CtensorB& x): 
      CFtensor(x), dims(x.dims), nbu(x.nbu){
      COPY_WARNING;
    }

    CtensorB(const CtensorB& x, const nowarn_flag& dummy): 
      CFtensor(x,dummy), dims(x.dims), nbu(x.nbu){}

    CtensorB* clone() const{
      return new CtensorB(*this, nowarn);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CtensorB(const CFtensor& x):
      CFtensor(x), dims(x.dims), nbu(-1){
    }

    CtensorB(CFtensor&& x):
      CFtensor(std::move(x)), dims(x.dims), nbu(-1){
    }


    void to_device(const device_id& _dev) const{
      CFtensor::to_device(_dev);
    }



  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }

    Gdims get_dims() const{
      return dims;
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
      //{CoutLock lk; cout<<r.val<<endl;}
      assert(nbu==-1);
      r.val+=inp(A);
      //{CoutLock lk; cout<<r.val<<endl;}
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


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

    string str(const string indent="") const{
      stringstream oss;
      return oss.str();
    }
   
  };


  inline CtensorB& asCtensorB(Cobject* x){
    assert(x); 
    if(!dynamic_cast<CtensorB*>(x))
      cerr<<"Cengine error: Cobject is of type "<<x->classname()<<" instead of CtensorB."<<endl;
    assert(dynamic_cast<CtensorB*>(x));
    return static_cast<CtensorB&>(*x);
  }

  inline CtensorB& asCtensorB(Cobject* x, const char* s){
    if(!x){CoutLock lk; cerr<<"Cengine error ("<<s<<"): CtensorB does not exist"<<endl;}
    assert(x); 
    if(!dynamic_cast<CtensorB*>(x))
      cerr<<"Cengine error: Cobject is of type "<<x->classname()<<" instead of CtensorB."<<endl;
    assert(dynamic_cast<CtensorB*>(x));
    return static_cast<CtensorB&>(*x);
  }

  inline CtensorB& asCtensorB(Cnode* x){
    assert(x->obj); 
    if(!dynamic_cast<CtensorB*>(x->obj))
      cerr<<"Cengine error: Cobject is of type "<<x->obj->classname()<<" instead of CtensorB."<<endl;
    assert(dynamic_cast<CtensorB*>(x->obj));
    return static_cast<CtensorB&>(*x->obj);
  }

  inline CtensorB& asCtensorB(Cnode* x, const char* s){
    if(!x->obj){CoutLock lk; cerr<<"Cengine error ("<<s<<"): CtensorB does not exist"<<endl;}
    assert(x->obj);
    if(!dynamic_cast<CtensorB*>(x->obj))
      cerr<<"Cengine error ("<<s<<"): Cobject is of type "<<x->obj->classname()<<" instead of CtensorB."<<endl;
    assert(dynamic_cast<CtensorB*>(x->obj));
    return static_cast<CtensorB&>(*x->obj);
  }

  inline CtensorB& asCtensorB(Cnode& x){
    assert(x.obj);
    if(!dynamic_cast<CtensorB*>(x.obj))
      cerr<<"Cengine error: Cobject is of type "<<x.obj->classname()<<" instead of CtensorB."<<endl;
    assert(dynamic_cast<CtensorB*>(x.obj));
    return static_cast<CtensorB&>(*x.obj);
  }

  inline CtensorB& asCtensorB(Cnode& x, const char* s){
    if(!x.obj){CoutLock lk; cerr<<"Cengine error ("<<s<<"): CtensorB does not exist"<<endl;}
    assert(x.obj);
    if(!dynamic_cast<CtensorB*>(x.obj))
      cerr<<"Cengine error: Cobject is of type "<<x.obj->classname()<<" instead of CtensorB."<<endl;
    assert(dynamic_cast<CtensorB*>(x.obj));
    return static_cast<CtensorB&>(*x.obj);
  }




}

#endif

