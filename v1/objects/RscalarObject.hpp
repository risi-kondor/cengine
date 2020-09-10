#ifndef _RscalarObject
#define _RscalarObject

#include "Cengine_base.hpp"
#include "RscalarB_ops.hpp"
//#include "RscalarInterface.hpp"

extern Cengine::Cengine* Cscalar_engine;


namespace Cengine{

  class RscalarObject{
  public:

    int nbu=-1; 

    Chandle* hdl;

    ~RscalarObject(){
    }


  public: // ---- Filled constructors ------------------------------------------------------------------------


    RscalarObject(){
      hdl=(*Cengine_engine)(new new_rscalar_op(-1,0));
    }

    RscalarObject(Chandle* _hdl): hdl(_hdl){}


    RscalarObject(const fill_raw& fill, const int device=0){
      hdl=(*Cengine_engine)(new new_rscalar_op(-1,device));
    }

    RscalarObject(const fill_zero& fill, const int device=0){
      hdl=(*Cengine_engine)(new new_rscalar_zero_op(-1,device));
    }

    RscalarObject(const fill_gaussian& fill, const int device=0){
      hdl=(*Cengine_engine)(new new_rscalar_zero_op(-1,device));
    }

    RscalarObject(const int x){
      hdl=(*Cengine_engine)(new new_rscalar_set_op(-1,x,0));
    }

    RscalarObject(const float x){
      hdl=(*Cengine_engine)(new new_rscalar_set_op(-1,x,0));
    }

    RscalarObject(const double x){
      hdl=(*Cengine_engine)(new new_rscalar_set_op(-1,x,0));
    }

    RscalarObject(const float x, const int device){
      hdl=(*Cengine_engine)(new new_rscalar_set_op(-1,x,device));
    }

    RscalarObject(const int nbd, const fill_raw& fill, const int device=0){
      hdl=(*Cengine_engine)(new new_rscalar_op(nbd,device));
    }

    RscalarObject(const int nbd, const fill_zero& fill, const int device=0){
      hdl=(*Cengine_engine)(new new_rscalar_zero_op(nbd,device));
    }

    RscalarObject(const int nbd, const fill_gaussian& fill, const int device=0){
      hdl=(*Cengine_engine)(new new_rscalar_gaussian_op(nbd,device));
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    RscalarObject(const RscalarObject& x):
      hdl((*Cengine_engine)(new rscalar_copy_op(nodeof(x.hdl)))){}
      
    RscalarObject(RscalarObject&& x){
      hdl=x.hdl;
      x.hdl=nullptr;
    }

    RscalarObject& operator=(const RscalarObject& x){
      delete hdl;
      //hdl=engine::rscalar_copy(x.hdl);
      hdl=(*Cengine_engine)(new rscalar_copy_op(nodeof(x.hdl)));
      return *this;
    }

    RscalarObject& operator=(RscalarObject&& x){
      delete hdl;
      hdl=x.hdl;
      x.hdl=nullptr;
      return *this;
    }
    

  public: // ---- Conversions --------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int getnbu() const{ // TODO make sure it exists!! 
      return asRscalarB(hdl->node->obj).nbu;
    }

    float val() const{
      //return engine::rscalar_get(hdl)[0];
      Cengine_engine->flush(hdl->node);
      vector<float> R=asRscalarB(hdl->node->obj);
      return R[0];
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      replace(hdl,(*Cengine_engine)(new rscalar_set_zero_op(nodeof(hdl))));
    }

    void zero(){
      replace(hdl,(*Cengine_engine)(new rscalar_set_zero_op(nodeof(hdl))));
    }


  public: // ---- Non-inplace operations ---------------------------------------------------------------------


    RscalarObject plus(const RscalarObject& x){
      return RscalarObject((*Cengine_engine)(new rscalar_add_op(nodeof(hdl),nodeof(x.hdl))));
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const RscalarObject& x){
      replace(hdl,Cengine_engine->push<rscalar_add_op>(hdl,x.hdl));
    }

    void add(const RscalarObject& x, const float c){
      replace(hdl,(*Cengine_engine)(new rscalar_add_times_real_op(nodeof(hdl),nodeof(x.hdl),c)));
    }

    void add_conj(const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_add_op(nodeof(hdl),nodeof(x.hdl))));
    }

    void subtract(const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_subtract_op(nodeof(hdl),nodeof(x.hdl))));
    }

    void add_minus(const RscalarObject& x, const RscalarObject& y){
      replace(hdl,(*Cengine_engine)(new rscalar_add_op(nodeof(hdl),nodeof(x.hdl))));
      replace(hdl,(*Cengine_engine)(new rscalar_subtract_op(nodeof(hdl),nodeof(y.hdl))));
    }

    void add_prod(const RscalarObject& x, const RscalarObject& y){
      replace(hdl,(*Cengine_engine)(new rscalar_add_prod_op(nodeof(hdl),nodeof(x.hdl),nodeof(y.hdl))));
    }

    void add_prodc1(const RscalarObject& x, const RscalarObject& y){
      replace(hdl,(*Cengine_engine)(new rscalar_add_prod_op(nodeof(hdl),nodeof(x.hdl),nodeof(y.hdl))));
    }

    void add_div(const RscalarObject& x, const RscalarObject& y){
      replace(hdl,(*Cengine_engine)(new rscalar_add_div_op(nodeof(hdl),nodeof(x.hdl),nodeof(y.hdl))));
    }

    void add_div_back0(const RscalarObject& g, const RscalarObject& y){
      replace(hdl,(*Cengine_engine)(new rscalar_add_div_back0_op(nodeof(hdl),nodeof(g.hdl),nodeof(y.hdl))));
    }

    void add_div_back1(const RscalarObject& g, const RscalarObject& x, const RscalarObject& y){
      replace(hdl,(*Cengine_engine)(new rscalar_add_div_back1_op(nodeof(hdl),nodeof(g.hdl),nodeof(x.hdl),nodeof(y.hdl))));
    }

    void add_abs(const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_add_abs_op(nodeof(hdl),nodeof(x.hdl))));
    }

    void add_abs_back(const RscalarObject& g, const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_add_abs_back_op(nodeof(hdl),nodeof(g.hdl),nodeof(x.hdl))));
    }

    void add_norm2_back(const RscalarObject& g, const RscalarObject& x){
      add_prod(g,x);
      add_prod(g,x);
    }

    void add_pow(const RscalarObject& x, const float p, const float c=1.0){
      replace(hdl,(*Cengine_engine)(new rscalar_add_pow_op(nodeof(hdl),nodeof(x.hdl),p,c)));
    }

    void add_exp(const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_add_exp_op(nodeof(hdl),nodeof(x.hdl))));
    }

    void add_ReLU(const RscalarObject& x, const float c=0){
      replace(hdl,(*Cengine_engine)(new rscalar_add_ReLU_op(nodeof(hdl),nodeof(x.hdl),c)));
    }

    void add_ReLU_back(const RscalarObject& g, const RscalarObject& x, const float c=0){
      replace(hdl,(*Cengine_engine)(new rscalar_add_ReLU_back_op(nodeof(hdl),nodeof(g.hdl),nodeof(x.hdl),c)));
    }

    void add_sigmoid(const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_add_sigmoid_op(nodeof(hdl),nodeof(x.hdl))));
    }

    void add_sigmoid_back(const RscalarObject& g, const RscalarObject& x){
      replace(hdl,(*Cengine_engine)(new rscalar_add_sigmoid_back_op(nodeof(hdl),nodeof(g.hdl),nodeof(x.hdl))));
    }


  public: 

    void inp_into(const RscalarObject& y, RscalarObject& R) const{
      R.add_prod(*this,y);
    }

    void norm2_into(RscalarObject& R) const{
      R.add_prod(*this,*this);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string classname() const{
      return "GEnet::RscalarObject";
    }

    string str(const string indent="") const{
      Cengine_engine->flush(hdl->node);
      vector<float> R=asRscalarB(hdl->node->obj);

      ostringstream oss;
      oss<<"[ ";
      for(int i=0; i<R.size(); i++)
	oss<<R[i]<<" ";
      oss<<"]";
      return oss.str();
    }
    

    friend ostream& operator<<(ostream& stream, const RscalarObject& x){
      stream<<x.str(); return stream;}

  };


}

#endif


  /*
  inline RscalarObject& asRscalar(Dobject* x){
    assert(x); 
    if(!dynamic_cast<RscalarObject*>(x))
      cerr<<"GEnet error: Dobject is of type "<<x->classname()<<" instead of RscalarObject."<<endl;
    assert(dynamic_cast<RscalarObject*>(x));
    return static_cast<RscalarObject&>(*x);
  }

  inline RscalarObject& asRscalar(Dobject& x){
    if(!dynamic_cast<RscalarObject*>(&x))
      cerr<<"GEnet error: Dobject is of type "<<x.classname()<<" instead of RscalarObject."<<endl;
    assert(dynamic_cast<RscalarObject*>(&x));
    return static_cast<RscalarObject&>(x);
  }

  inline RscalarObject& asRscalar(Dnode* x){
    assert(x->obj); 
    if(!dynamic_cast<RscalarObject*>(x->obj))
      cerr<<"GEnet error: Dobject is of type "<<x->obj->classname()<<" instead of RscalarObject."<<endl;
    assert(dynamic_cast<RscalarObject*>(x->obj));
    return static_cast<RscalarObject&>(*x->obj);
  }

  inline RscalarObject& asRscalar(Dnode& x){
    if(!dynamic_cast<RscalarObject*>(x.obj))
      cerr<<"GEnet error: Dobject is of type "<<x.obj->classname()<<" instead of RscalarObject."<<endl;
    assert(dynamic_cast<RscalarObject*>(x.obj));
    return static_cast<RscalarObject&>(*x.obj);
  }
  */
