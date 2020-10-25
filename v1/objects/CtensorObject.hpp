#ifndef _CtensorObject
#define _CtensorObject

#include "Cengine_base.hpp"
#include "ExprTemplates.hpp"
#include "RscalarObject.hpp"
#include "CscalarObject.hpp"
//#include "CtensorInterface.hpp"


namespace Cengine{


  class CtensorObject{
  public:

    Gdims dims;
    int nbu=-1;
    int device=0; 

    Chandle* hdl=nullptr;

    ~CtensorObject(){
      delete hdl; 
    }

    CtensorObject(){}

    CtensorObject(Chandle* _hdl, const Gdims& _dims): 
      dims(_dims), hdl(_hdl){}

    CtensorObject(Chandle* _hdl, const Gdims& _dims, const int _nbu): 
      dims(_dims), hdl(_hdl), nbu(_nbu){}

    CtensorObject(const Gdims& _dims): dims(_dims){
      hdl=Cengine_engine->push<new_ctensor_op>(_dims,-1,0);
    }

    CtensorObject(const Gdims& _dims, const fill_raw& fill, const int _device=0): 
      dims(_dims), device(_device){
      hdl=Cengine_engine->push<new_ctensor_op>(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_zero& fill, const int _device=0): 
      dims(_dims), device(_device){
      hdl=Cengine_engine->push<new_ctensor_zero_op>(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_identity& fill, const int _device=0): 
      dims(_dims), device(_device){
      hdl=Cengine_engine->push<new_ctensor_identity_op>(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_sequential& fill, const int _device=0): 
      dims(_dims), device(_device){
      hdl=Cengine_engine->push<new_ctensor_sequential_op>(_dims,-1,device);
    }

    CtensorObject(const Gdims& _dims, const fill_gaussian& fill, const int _device=0): 
      dims(_dims), device(_device){
      hdl=Cengine_engine->push<new_ctensor_gaussian_op>(_dims,-1,fill.c,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd=-1, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_op>(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_raw& fill, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_op>(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_zero& fill, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_zero_op>(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_ones& fill, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_ones_op>(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_identity& fill, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_identity_op>(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_sequential& fill, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_sequential_op>(_dims,nbd,device);
    }

    CtensorObject(const Gdims& _dims, const int nbd, const fill_gaussian& fill, const int _device=0): 
      dims(_dims), nbu(nbd), device(_device){
      hdl=Cengine_engine->push<new_ctensor_gaussian_op>(_dims,nbd,fill.c,device);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    CtensorObject(const CtensorObject& x):
      dims(x.dims),
      nbu(x.nbu),
      device(x.device),
      hdl(Cengine_engine->push<ctensor_copy_op>(x.hdl)){}
    
    CtensorObject(CtensorObject&& x):
      dims(std::move(x.dims)),
      nbu(x.nbu),
      device(x.device)
    {
      hdl=x.hdl;
      x.hdl=nullptr;
    }

    CtensorObject& operator=(const CtensorObject& x){
      dims=x.dims;
      nbu=x.nbu;
      device=x.device;
      delete hdl;
      hdl=Cengine_engine->push<ctensor_copy_op>(x.hdl);
      return *this;
    }

    CtensorObject& operator=(CtensorObject&& x){
      //cout<<"a"<<endl; 
      dims=x.dims;
      nbu=x.nbu;
      device=x.device;
      delete hdl;
      hdl=x.hdl;
      x.hdl=nullptr;
      return *this;
    }
    
    //CtensorSeed* seed() const{
    //return new CtensorSeed(dims,nbu);
    //}

    CtensorObject(CtensorObject& x, const view_flag& flag):
      dims(x.dims), nbu(x.nbu), hdl(new_handle(x.hdl->node)){}
      

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
      hdl=Cengine_engine->push<new_ctensor_from_gtensor_op>(x);
      //hdl=engine::new_ctensor_from_gtensor(x);
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
      ctensor_get(hdl);
    }

    CtensorObject& to_device(const device_id& _dev){
      device=_dev.id();
      replace(hdl,Cengine_engine->push<ctensor_to_device_op>(hdl,_dev.id()));
      return *this; 
    }

    CtensorObject& to(const device_id& _dev){
      device=_dev.id();
      replace(hdl,Cengine_engine->push<ctensor_to_device_op>(hdl,_dev.id()));
      return *this; 
    }

    CtensorObject& to(const int dev){
      device=dev;
      replace(hdl,Cengine_engine->push<ctensor_to_device_op>(hdl,device));
      return *this; 
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      replace(hdl,Cengine_engine->push<ctensor_zero_op>(hdl));
    }


  public: // ---- Not in-place operations --------------------------------------------------------------------


    CtensorObject conj() const{
      return CtensorObject(Cengine_engine->push<ctensor_conj_op>(hdl),dims,nbu);
    }

    CtensorObject transp() const{
      return CtensorObject(Cengine_engine->push<ctensor_transp_op>(hdl),dims,nbu); //TODO
    }

    CtensorObject herm() const{
      return CtensorObject(Cengine_engine->push<ctensor_herm_op>(hdl),dims,nbu);
    }

    CtensorObject plus(const CtensorObject& x){
      return CtensorObject(Cengine_engine->push<ctensor_add_op>(hdl,x.hdl),dims,nbu);
    }

    CscalarObject mix(const CscalarObject& x){
      assert(dims.size()==2);
      CscalarObject r(dims[0],fill::zero);
      Cengine_engine->push<cscalar_mix_op>(r.hdl,hdl,x.hdl);
      return r;
    }

    CtensorObject mix(const CtensorObject& x){
      assert(dims.size()==2);
      CtensorObject r(dims[0],fill::zero);
      Cengine_engine->push<ctensor_mix_op>(r.hdl,hdl,x.hdl);
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CtensorObject& x){
      replace(hdl,Cengine_engine->push<ctensor_add_op>(hdl,x.hdl));
    }

    void add_conj(const CtensorObject& x){
      replace(hdl,Cengine_engine->push<ctensor_add_conj_op>(hdl,x.hdl));
    }

    void add_transp(const CtensorObject& x){
      replace(hdl,Cengine_engine->push<ctensor_add_transp_op>(hdl,x.hdl));
    }

    void add_herm(const CtensorObject& x){
      replace(hdl,Cengine_engine->push<ctensor_add_herm_op>(hdl,x.hdl));
    }

    void subtract(const CtensorObject& x){
      replace(hdl,Cengine_engine->push<ctensor_subtract_op>(hdl,x.hdl));
    }

    void add(const CtensorObject& x, const float c){
      replace(hdl,Cengine_engine->push<ctensor_add_times_real_op>(hdl,x.hdl,c));
    }

    void add(const CtensorObject& x, const complex<float> c){
      replace(hdl,Cengine_engine->push<ctensor_add_times_complex_op>(hdl,x.hdl,c));
    }

    void add(const CtensorObject& x, const RscalarObject& c){
      replace(hdl,Cengine_engine->push<ctensor_add_prod_rA_op>(hdl,c.hdl,x.hdl));
    }

    void add(const CtensorObject& x, const CscalarObject& c){
      replace(hdl,Cengine_engine->push<ctensor_add_prod_cA_op>(hdl,c.hdl,x.hdl));
    }

    void add_cconj(const CtensorObject& x, const CscalarObject& c){
      replace(hdl,Cengine_engine->push<ctensor_add_prod_cc_A_op>(hdl,c.hdl,x.hdl));
    }
   
    void add_conj(const CtensorObject& x, const CscalarObject& c){
      replace(hdl,Cengine_engine->push<ctensor_add_prod_c_Ac_op>(hdl,c.hdl,x.hdl));
    }

    
    void add_plus(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_op>(hdl,x.hdl));
      replace(hdl,Cengine_engine->push<ctensor_add_op>(hdl,y.hdl));
    }

    void add_minus(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_op>(hdl,x.hdl));
      replace(hdl,Cengine_engine->push<ctensor_subtract_op>(hdl,y.hdl));
    }


    void add_Mprod(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<0,0> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_Mprod_AT(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<2,0> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_Mprod_TA(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<1,0> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_Mprod_AC(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<0,2> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_Mprod_TC(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<1,2> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_Mprod_AH(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<2,2> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }

    void add_Mprod_HA(const CtensorObject& x, const CtensorObject& y){
      replace(hdl,Cengine_engine->push<ctensor_add_Mprod_op<1,1> >(hdl,x.hdl,y.hdl,x.dims,y.dims));
    }


    void add_column_norms(const CtensorObject& x){
      replace(hdl,Cengine_engine->push<ctensor_add_col_norms_op>(hdl,x.hdl));
    }


    void add_ReLU(const CtensorObject& x, const float c=0){
      replace(hdl,Cengine_engine->push<ctensor_add_ReLU_op>(hdl,x.hdl,c));
    }

    void add_ReLU_back(const CtensorObject& g, const CtensorObject& x, const float c=0){
      replace(hdl,Cengine_engine->push<ctensor_add_ReLU_back_op>(hdl,g.hdl,x.hdl,c));
    }

    
  public: // ---- Into operations ----------------------------------------------------------------------------


    void inp_into(const CtensorObject& y, CscalarObject& R) const{
      replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,y.hdl));
    }

    void norm2_into(CscalarObject& R) const{
      replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,hdl));
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
      CtensorObject R(dims,nbu,fill::zero,device);
      R.add(*this,c);
      return R;
    }

    CtensorObject operator*(const CtensorObject& y){
      int I=dims.combined(0,dims.k()-1);
      int J=y.dims.combined(1,y.dims.k());
      CtensorObject R({I,J},fill::zero,device);
      R.add_Mprod(*this,y);
      return R;
    }

    CtensorObject operator*(const Transpose<CtensorObject>& y){
      int I=dims.combined(0,dims.k()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
      CtensorObject R({I,J},fill::zero,device);
      R.add_Mprod_AT(*this,y.obj);
      return R;
    }

    CtensorObject column_norms() const{
      assert(dims.size()>=2);
      CtensorObject R(dims.remove(dims.size()-2),nbu,fill::zero,device);
      R.add_column_norms(*this);
      return R;
    }

    CtensorObject divide_columns(const CtensorObject& N){
      return CtensorObject(Cengine_engine->push<ctensor_divide_cols_op>(hdl,N.hdl),dims,nbu);
    }
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::CtensorObject";
    }

    string str(const string indent="") const{
      Gtensor<complex<float> > R=ctensor_get(hdl);
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

 
