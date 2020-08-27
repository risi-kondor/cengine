#ifndef _CtensorObject
#define _CtensorObject

#include "Cengine_base.hpp"
#include "ExprTemplates.hpp"
#include "RscalarObject.hpp"
#include "CscalarObject.hpp"
#include "CtensorInterface.hpp"


namespace Cengine{


  class CtensorObject{
  public:

    Gdims dims;
    int nbu=-1;

    Chandle* hdl;

    ~CtensorObject(){
      delete hdl; 
    }

    CtensorObject(){}

    CtensorObject(Chandle* _hdl, const Gdims& _dims): 
      dims(_dims), hdl(_hdl){}

    CtensorObject(Chandle* _hdl, const Gdims& _dims, const int _nbu): 
      dims(_dims), hdl(_hdl), nbu(_nbu){}

    CtensorObject(const Gdims& _dims): dims(_dims){
      hdl=engine::new_ctensor(_dims,-1,0);
    }

    CtensorObject(const Gdims& _dims, const fill_raw& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_zero& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_zero(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_identity& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_identity(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_sequential& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_sequential(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_gaussian& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_gaussian(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd=-1, const int device=0): dims(_dims){
      hdl=engine::new_ctensor(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_raw& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_zero& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_zero(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_ones& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_ones(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_identity& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_identity(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_sequential& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_sequential(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_gaussian& fill, const int device=0): dims(_dims){
      hdl=engine::new_ctensor_gaussian(_dims,nbd,device);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    CtensorObject(const CtensorObject& x):
      hdl(engine::ctensor_copy(x.hdl)){}
      
    CtensorObject(CtensorObject&& x){
      hdl=x.hdl;
      x.hdl=nullptr;
    }

    CtensorObject& operator=(const CtensorObject& x){
      delete hdl;
      hdl=engine::ctensor_copy(x.hdl);
      return *this;
    }

    CtensorObject& operator=(CtensorObject&& x){
      delete hdl;
      hdl=x.hdl;
      x.hdl=nullptr;
      return *this;
    }
    
    //CtensorSeed* seed() const{
    //return new CtensorSeed(dims,nbu);
    //}

    CtensorObject(CtensorObject& x, const view_flag& flag):
      hdl(new_handle(x.hdl->node)){}
      

  public: // ---- Conversions --------------------------------------------------------------------------------

    
    CtensorObject(const Conjugate<CtensorObject>& x):
      CtensorObject(x.obj.conj()){}

    CtensorObject(const Transpose<CtensorObject>& x):
      CtensorObject(x.obj.transp()){}

    CtensorObject(const Hermitian<CtensorObject>& x):
      CtensorObject(x.obj.herm()){}

    CtensorObject(const Transpose<Conjugate<CtensorObject> >& x):
      CtensorObject(x.obj.obj.herm()){}

    CtensorObject(const Conjugate<Transpose<CtensorObject> >& x):
      CtensorObject(x.obj.obj.herm()){}


    CtensorObject(const Gtensor<complex<float> >& x, const fill_tensor& dummy, const int _device=0): 
      dims(x.dims){
      hdl=engine::new_ctensor_from_gtensor(x);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{ 
      return nbu;
    }

    int get_k() const{ 
      return dims.size();
    }

    Gdims get_dims() const{ 
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    /*
    int combined(const int a, const int b) const{
      return asCtensorB(hdl->node->obj).combined(a,b);
    }
    */

    void flush() const{
      engine::ctensor_get(hdl);
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      engine::ctensor_zero(hdl);
    }


  public: // ---- Not in-place operations --------------------------------------------------------------------


    CtensorObject conj() const{
      return CtensorObject(engine::ctensor_conj(hdl),dims);
    }

    CtensorObject transp() const{
      return CtensorObject(engine::ctensor_transp(hdl),dims);
    }

    CtensorObject herm() const{
      return CtensorObject(engine::ctensor_herm(hdl),dims);
    }

    CtensorObject plus(const CtensorObject& x){
      return CtensorObject(engine::ctensor_add(hdl,x.hdl),x.dims,nbu);
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CtensorObject& x){
      replace(hdl,engine::ctensor_add(hdl,x.hdl));
    }

    void add_conj(const CtensorObject& x){
      replace(hdl,engine::ctensor_add_conj(hdl,x.hdl));
    }

    void add_transp(const CtensorObject& x){
      replace(hdl,engine::ctensor_add_transp(hdl,x.hdl));
    }

    void add_herm(const CtensorObject& x){
      replace(hdl,engine::ctensor_add_herm(hdl,x.hdl));
    }

    void subtract(const CtensorObject& x){
      replace(hdl,engine::ctensor_add(hdl,x.hdl));
    }

    void add(const CtensorObject& x, const float c){
      replace(hdl,engine::ctensor_add_times_real(hdl,x.hdl,c));
    }

    void add(const CtensorObject& x, const complex<float> c){
      replace(hdl,engine::ctensor_add_times_complex(hdl,x.hdl,c));
    }

    void add(const CtensorObject& x, const RscalarObject& c){
      replace(hdl,engine::ctensor_add_prod_rA(hdl,c.hdl,x.hdl));
    }

    void add(const CtensorObject& x, const CscalarObject& c){
      replace(hdl,engine::ctensor_add_prod_cA(hdl,c.hdl,x.hdl));
    }

    void add_cconj(const CtensorObject& x, const CscalarObject& c){
      replace(hdl,engine::ctensor_add_prod_cc_A(hdl,c.hdl,x.hdl));
    }
   
    void add_conj(const CtensorObject& x, const CscalarObject& c){
      replace(hdl,engine::ctensor_add_prod_c_Ac(hdl,c.hdl,x.hdl));
    }

    
    void add_plus(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add(hdl,x.hdl));
      replace(hdl,engine::ctensor_add(hdl,y.hdl));
    }

    void add_minus(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add(hdl,x.hdl));
      replace(hdl,engine::ctensor_subtract(hdl,y.hdl));
    }


    void add_Mprod(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod(hdl,x.hdl,y.hdl));
    }

    void add_Mprod_AT(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod_AT(hdl,x.hdl,y.hdl));
    }

    void add_Mprod_TA(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod_TA(hdl,x.hdl,y.hdl));
    }

    void add_Mprod_AC(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod_AC(hdl,x.hdl,y.hdl));
    }

    void add_Mprod_TC(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod_TC(hdl,x.hdl,y.hdl));
    }

    void add_Mprod_AH(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod_AH(hdl,x.hdl,y.hdl));
    }

    void add_Mprod_HA(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,engine::ctensor_add_Mprod_HA(hdl,x.hdl,y.hdl));
    }

    
    void add_ReLU(const CtensorObject& x, const float c=0){
      replace(hdl,engine::ctensor_add_ReLU(hdl,x.hdl,c));
    }

    void add_ReLU_back(const CtensorObject& g, const CtensorObject& x, const float c=0){
      replace(hdl,engine::ctensor_add_ReLU_back(hdl,g.hdl,x.hdl,c));
    }

    
  public: // ---- Into operations ----------------------------------------------------------------------------


    void inp_into(const CtensorObject& y, CscalarObject& R) const{
      replace(R.hdl,engine::ctensor_add_inp(R.hdl,hdl,y.hdl));
    }

    void norm2_into(CscalarObject& R) const{
      replace(R.hdl,engine::ctensor_add_inp(R.hdl,hdl,hdl));
    }

    void add_norm2_back(const CscalarObject& g, const CtensorObject& x){
      add(x,g);
      add_conj(x,g);
    }


  public: // ---- Operators ---------------------------------------------------------------------------------


  CtensorObject& operator+=(const CtensorObject& y){
    add(y);
    return *this;
  }

  CtensorObject& operator-=(const CtensorObject& y){
    subtract(y);
    return *this;
  }

  CtensorObject operator+(const CtensorObject& y){
    CtensorObject R(*this);
    R.add(y);
    return R;
  }

  CtensorObject operator-(const CtensorObject& y){
    CtensorObject R(*this);
    R.subtract(y);
    return R;
  }

  CtensorObject operator*(const CscalarObject& c){
    CtensorObject R(dims,nbu,fill::zero);
    R.add(*this,c);
    return R;
  }

  CtensorObject operator*(const CtensorObject& y){
    int I=dims.combined(0,dims.k()-1);
    int J=y.dims.combined(1,y.dims.k());
    CtensorObject R({I,J},fill::zero);
    R.add_Mprod(*this,y);
    return R;
  }

  CtensorObject operator*(const Transpose<CtensorObject>& y){
    int I=dims.combined(0,dims.k()-1);
    int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
    CtensorObject R({I,J},fill::zero);
    R.add_Mprod_AT(*this,y.obj);
    return R;
  }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::CtensorObject";
    }

    string str(const string indent="") const{
      Gtensor<complex<float> > R=engine::ctensor_get(hdl);
      return R.str();
    }

    friend ostream& operator<<(ostream& stream, const CtensorObject& x){
      stream<<x.str(); return stream;}

  };


  // ---- Functions ------------------------------------------------------------------------------------------


  Transpose<CtensorObject> transp(const CtensorObject& x){
    return Transpose<CtensorObject>(x);
  }

  Conjugate<CtensorObject> conj(const CtensorObject& x){
    return Conjugate<CtensorObject>(x);
  }

  Hermitian<CtensorObject> herm(const CtensorObject& x){
    return x;
  }

  CtensorObject operator*(const Transpose<CtensorObject>& x, const CtensorObject& y){
    int I=x.obj.dims.combined(1,x.obj.dims.k());
    int J=y.dims.combined(1,y.dims.k());
    CtensorObject R({I,J},fill::zero);
    R.add_Mprod_TA(x.obj,y);
    return R;
  }

  CscalarObject norm2(const CtensorObject& x){
    CscalarObject r(x.nbu,fill::zero);
    x.norm2_into(r);
    return r;
  }

  CtensorObject ReLU(const CtensorObject& x, const float c=0){
    CtensorObject R(x.dims,x.nbu,fill::zero);
    R.add_ReLU(x,c);
    return R;
  }

  CscalarObject inp(const CtensorObject& x, const CtensorObject& y){
    CscalarObject r(x.nbu,fill::zero);
    x.inp_into(y,r);
    return r;
  }



}


#endif


    //public: // ---- Filled constructors ------------------------------------------------------------------------
  /*
  inline CtensorObject& asCtensor(Dobject* x){
    assert(x); 
    if(!dynamic_cast<CtensorObject*>(x))
      cerr<<"GEnet error: Dobject is of type "<<x->classname()<<" instead of CtensorObject."<<endl;
    assert(dynamic_cast<CtensorObject*>(x));
    return static_cast<CtensorObject&>(*x);
  }

  inline CtensorObject& asCtensor(Dobject& x){
    if(!dynamic_cast<CtensorObject*>(&x))
      cerr<<"GEnet error: Dobject is of type "<<x.classname()<<" instead of CtensorObject."<<endl;
    assert(dynamic_cast<CtensorObject*>(&x));
    return static_cast<CtensorObject&>(x);
  }

  inline CtensorObject& asCtensor(Dnode* x){
    assert(x->obj); 
    if(!dynamic_cast<CtensorObject*>(x->obj))
      cerr<<"GEnet error: Dobject is of type "<<x->obj->classname()<<" instead of CtensorObject."<<endl;
    assert(dynamic_cast<CtensorObject*>(x->obj));
    return static_cast<CtensorObject&>(*x->obj);
  }

  inline CtensorObject& asCtensor(Dnode& x){
    if(!dynamic_cast<CtensorObject*>(x.obj))
      cerr<<"GEnet error: Dobject is of type "<<x.obj->classname()<<" instead of CtensorObject."<<endl;
    assert(dynamic_cast<CtensorObject*>(x.obj));
    return static_cast<CtensorObject&>(*x.obj);
  }
  */
  /*
  inline CtensorObject CtensorSeed::spawn(const fill_zero& fill){
    //if(nch<0) return new SO3partB(l,n,fill::zero,device);
    return CtensorObject(dims,nbu,fill::zero,device);
    }
  */

  /*
  class CtensorObject;

  class CtensorSeed{
  public:
    
    Gdims dims;
    int nbu=-1;
    int device; 

    CtensorSeed(const Gdims& _dims, const int _nbu, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){}

    CtensorObject spawn(const fill_zero& fill);

  };
  */
