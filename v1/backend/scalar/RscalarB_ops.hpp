#ifndef _RscalarB_ops
#define _RscalarB_ops

#include "RscalarB.hpp"

namespace Cengine{


  class new_rscalar_op: public Coperator{
  public:

    int nbu;
    int device;

    new_rscalar_op(const int _nbu=-1, const int _device=0):
      nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new RscalarB(nbu,fill::raw,device);
    }

    string str() const{
      return "rscalar()";
    }

  };


  class new_rscalar_zero_op: public Coperator{
  public:

    int nbu;
    int device;

    new_rscalar_zero_op(const int _nbu=-1, const int _device=0):
      nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new RscalarB(nbu,fill::zero,device);
    }

    string str() const{
      return "rscalar_zero()";
    }

  };


  class new_rscalar_gaussian_op: public Coperator{
  public:

    int nbu;
    int device;
    float c=1.0;

    new_rscalar_gaussian_op(const int _nbu=-1, const int _device=0):
      nbu(_nbu), device(_device){
    }

    new_rscalar_gaussian_op(const int _nbu, const float _c, const int _device=0):
      nbu(_nbu), device(_device), c(_c){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new RscalarB(nbu,fill::gaussian,c,device);
    }
    
    string str() const{
      return "rscalar_gaussian()";
    }

  };


  class new_rscalar_set_op: public Coperator{
  public:

    int nbu;
    int device;
    float c;

    new_rscalar_set_op(const int _nbu, const float _c, const int _device=0):
      nbu(_nbu), device(_device), c(_c){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new RscalarB(nbu,c,device);
    }

    string str() const{
      ostringstream oss;
      oss<<"rscalar_set("<<c<<")";
      return oss.str(); 
    }

  };


  class rscalar_copy_op: public Coperator{
  public:

    rscalar_copy_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new RscalarB(asRscalarB(inputs[0],__PRETTY_FUNCTION__));
    }
    
    string str() const{
      return "rscalar_copy"+inp_str();
    }

  };


  // ---- Not in-place operators  ----------------------------------------------------------------------------

  

  // ---- In-place operators  --------------------------------------------------------------------------------

  
  class rscalar_set_zero_op: public Coperator, public InPlaceOperator{
  public:

    rscalar_set_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      asRscalarB(owner,__PRETTY_FUNCTION__).zero();
    }

    string str() const{
      return "rscalar_set_zero"+inp_str();
    }

  };
  

  // ---- Cumulative operators  ------------------------------------------------------------------------------


  class rscalar_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add"+inp_str();
    }

  };
  

  class rscalar_add_times_real_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c;

    rscalar_add_times_real_op(Cnode* r, Cnode* x, float _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add(asRscalarB(inputs[1],__PRETTY_FUNCTION__),c);
    }

    string str() const{
      return "rscalar_add_times_real"+inp_str();
    }

  };
  

  class rscalar_subtract_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_subtract_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).subtract(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_subtract"+inp_str();
    }

  };

  class rscalar_add_prod_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_prod_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_prod(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_prod"+inp_str();
    }

  };

  
  class rscalar_add_div_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_div_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_div(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_div"+inp_str();
    }

  };

  
  class rscalar_add_div_back0_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_div_back0_op(Cnode* r, Cnode* g, Cnode* y):
      Coperator(r,g,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_div_back0(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_div_back0"+inp_str();
    }

  };

  
  class rscalar_add_div_back1_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_div_back1_op(Cnode* r, Cnode* g, Cnode* x, Cnode* y):
      Coperator(r,g,x,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).
	add_div_back1(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__),asRscalarB(inputs[3],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_div_back1"+inp_str();
    }

  };

  
  class rscalar_add_pow_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float p;
    float c;

    rscalar_add_pow_op(Cnode* r, Cnode* x, float _p, float _c):
      Coperator(r,x), p(_p), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_pow(asRscalarB(inputs[1],__PRETTY_FUNCTION__),p,c);
    }

    string str() const{
      return "rscalar_add_pow"+inp_str();
    }

  };
  

  class rscalar_add_pow_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float p;
    float c;

    rscalar_add_pow_back_op(Cnode* r, Cnode* g, Cnode* x, float _p, float _c):
      Coperator(r,g,x), p(_p), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_pow_back(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__),p,c);
    }

    string str() const{
      return "rscalar_add_pow_back"+inp_str();
    }

  };
  

  class rscalar_add_exp_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_exp_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_exp(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_exp"+inp_str();
    }

  };
  

  class rscalar_add_abs_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_abs_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_abs(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_abs"+inp_str();
    }

  };
  

  class rscalar_add_abs_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_abs_back_op(Cnode* r, Cnode*g, Cnode* x):
      Coperator(r,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_abs_back(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_abs_back"+inp_str();
    }

  };
  

  class rscalar_add_ReLU_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    rscalar_add_ReLU_op(Cnode* r, Cnode* x, float _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_ReLU(asRscalarB(inputs[1],__PRETTY_FUNCTION__),c);
    }

    string str() const{
      return "rscalar_add_ReLU"+inp_str();
    }

  };
  

  class rscalar_add_ReLU_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    rscalar_add_ReLU_back_op(Cnode* r, Cnode* g, Cnode* x, float _c):
      Coperator(r,g,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_ReLU_back(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__),c);
    }

    string str() const{
      return "rscalar_add_ReLU_back"+inp_str();
    }

  };
  

  class rscalar_add_sigmoid_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_sigmoid_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_sigmoid(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_sigmoid"+inp_str();
    }

  };
  

  class rscalar_add_sigmoid_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    rscalar_add_sigmoid_back_op(Cnode* r, Cnode* g, Cnode* x):
      Coperator(r,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asRscalarB(owner,__PRETTY_FUNCTION__).add_sigmoid_back(asRscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "rscalar_add_sigmoid_back"+inp_str();
    }
    
  };


}

#endif

