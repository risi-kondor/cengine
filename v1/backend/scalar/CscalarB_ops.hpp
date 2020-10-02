#ifndef _CscalarB_ops
#define _CscalarB_ops

#include "CscalarB.hpp"
#include "Coperators.hpp"


namespace Cengine{


  class new_cscalar_op: public Coperator{
  public:

    int nbu;
    int device;

    new_cscalar_op(const int _nbu=-1, const int _device=0):
      nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CscalarB(nbu,fill::raw,device);
    }

    string str() const{
      return "cscalar()";
    }

  };


  class new_cscalar_zero_op: public Coperator{
  public:

    int nbu;
    int device;

    new_cscalar_zero_op(const int _nbu=-1, const int _device=0):
      nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CscalarB(nbu,fill::zero,device);
    }

    string str() const{
      return "cscalar_zero()";
    }

  };


  class new_cscalar_gaussian_op: public Coperator{
  public:

    int nbu;
    int device;

    new_cscalar_gaussian_op(const int _nbu=-1, const int _device=0):
      nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CscalarB(nbu,fill::gaussian,device);
    }
    
    string str() const{
      return "cscalar_gaussian()";
    }

  };


  class new_cscalar_set_op: public Coperator{
  public:

    int nbu;
    int device;
    complex<float> c;

    new_cscalar_set_op(const int _nbu, const complex<float> _c, const int _device=0):
      nbu(_nbu), device(_device), c(_c){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CscalarB(nbu,c,device);
    }

    string str() const{
      ostringstream oss;
      oss<<"new_cscalar_set("<<c<<")";
      return oss.str(); 
    }

  };


  class cscalar_copy_op: public Coperator{
  public:

    cscalar_copy_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CscalarB(asCscalarB(inputs[0],__PRETTY_FUNCTION__));
    }
    
    string str() const{
      return "cscalar_copy"+inp_str();
    }

  };


  // ---- Not in-place operators  ----------------------------------------------------------------------------


  class cscalar_conj_op: public Coperator{
  public:

    cscalar_conj_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asCscalarB(inputs[0],__PRETTY_FUNCTION__).conj();
    }

    string str() const{
      return "cscalar_conj"+inp_str();
    }

  };
  

  class cscalar_get_real_op: public Coperator{
  public:

    cscalar_get_real_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asCscalarB(inputs[0],__PRETTY_FUNCTION__).real();
    }

    string str() const{
      return "cscalar_get_real"+inp_str();
    }

  };
  

  class cscalar_get_imag_op: public Coperator{
  public:

    cscalar_get_imag_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asCscalarB(inputs[0],__PRETTY_FUNCTION__).imag();
    }

    string str() const{
      return "cscalar_get_imag"+inp_str();
    }

  };
  

  // ---- In-place operators  --------------------------------------------------------------------------------

  
  class cscalar_set_zero_op: public Coperator, public InPlaceOperator{
  public:

    cscalar_set_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      asCscalarB(owner,__PRETTY_FUNCTION__).zero();
    }

    string str() const{
      return "cscalar_set_zero"+inp_str();
    }

  };

  
  class cscalar_set_real_op: public Coperator, public InPlaceOperator{
  public:

    cscalar_set_real_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).set_real(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_set_real"+inp_str();
    }

  };


  class cscalar_set_imag_op: public Coperator, public InPlaceOperator{
  public:

    cscalar_set_imag_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).set_imag(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_set_imag"+inp_str();
    }

  };


  // ---- Cumulative operators  ------------------------------------------------------------------------------


  class cscalar_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add(asCscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add"+inp_str();
    }

  };
  

  class cscalar_add_to_real_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 

    //cscalar_add_to_real_op(Cnode* r, Cnode* x):
    //Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_to_real(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_to_real"+inp_str();
    }

  };
  

  class cscalar_add_to_imag_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 

    //cscalar_add_to_imag_op(Cnode* r, Cnode* x):
    //Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_to_imag(asRscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_to_imag"+inp_str();
    }

  };
  

  class cscalar_add_conj_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 

    //cscalar_add_conj_op(Cnode* r, Cnode* x):
    //Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_conj(asCscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_conj"+inp_str();
    }

  };
  

  class cscalar_add_times_real_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c;

    cscalar_add_times_real_op(Cnode* r, Cnode* x, float _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add(asCscalarB(inputs[1],__PRETTY_FUNCTION__),c);
      //owner->computed=true; 
    }

    string str() const{
      return "cscalar_add_times_real"+inp_str();
    }

  };
  

  class cscalar_add_times_complex_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    complex<float> c;

    cscalar_add_times_complex_op(Cnode* r, Cnode* x, complex<float> _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add(asCscalarB(inputs[1],__PRETTY_FUNCTION__),c);
      //owner->computed=true; 
    }

    string str() const{
      return "cscalar_add_times_complex"+inp_str();
    }

  };
  

  class cscalar_subtract_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 

    //cscalar_subtract_op(Cnode* r, Cnode* x):
    //Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).subtract(asCscalarB(inputs[1],__PRETTY_FUNCTION__));
      //owner->computed=true; 
    }

    string str() const{
      return "cscalar_subtract"+inp_str();
    }

  };

  class cscalar_add_prod_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_prod(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_prod"+inp_str();
    }

  };


  
  class cscalar_add_prod2_op: public CumulativeOp3<CscalarB,CscalarB,CscalarB>{
  public:

    using CumulativeOp3::CumulativeOp3; 

    void exec(CscalarB& r, const CscalarB& x, const CscalarB& y){
      output().add_prod(x,y);
    }

    string str() const{
      return "cscalar_add_prod2"+inp_str();
    }

  };

  
  class cscalar_add_prodc_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 
    
    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_prodc(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_prodc"+inp_str();
    }

  };
    

  class cscalar_add_prodcc_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    using Coperator::Coperator; 
    
    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_prodcc(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_prodcc"+inp_str();
    }

  };
    

  class cscalar_add_prod_r_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_prod_r_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_prod(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asRscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_prod_r"+inp_str();
    }

  };
    

  class cscalar_add_div_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_div_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_div(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_div"+inp_str();
    }

  };

  
  class cscalar_add_div_back0_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_div_back0_op(Cnode* r, Cnode* g, Cnode* y):
      Coperator(r,g,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_div_back0(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_div_b0"+inp_str();
    }

  };

  
  class cscalar_add_div_back1_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_div_back1_op(Cnode* r, Cnode* g, Cnode* x, Cnode* y):
      Coperator(r,g,x,y){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_div_back1(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__),asCscalarB(inputs[3],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_div_b1"+inp_str();
    }

  };

  
  class cscalar_add_pow_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float p;
    complex<float> c;

    cscalar_add_pow_op(Cnode* r, Cnode* x, float _p, complex<float> _c):
      Coperator(r,x), p(_p), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_pow(asCscalarB(inputs[1],__PRETTY_FUNCTION__),p,c);
    }

    string str() const{
      return "cscalar_add_pow"+inp_str();
    }

  };
  

  class cscalar_add_pow_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float p;
    complex<float> c;

    cscalar_add_pow_back_op(Cnode* r, Cnode* g, Cnode* x, float _p, complex<float> _c):
      Coperator(r,g,x), p(_p), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_pow_back(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__),p,c);
    }

    string str() const{
      return "cscalar_add_pow_back"+inp_str();
    }

  };
  

  class cscalar_add_exp_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_exp_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_exp(asCscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_exp"+inp_str();
    }

  };
  

  class cscalar_add_abs_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_abs_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_abs(asCscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_abs"+inp_str();
    }

  };
  

  class cscalar_add_abs_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_abs_back_op(Cnode* r, Cnode*g, Cnode* x):
      Coperator(r,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_abs_back(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_abs_back"+inp_str();
    }

  };
  

  class cscalar_add_ReLU_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    cscalar_add_ReLU_op(Cnode* r, Cnode* x, float _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_ReLU(asCscalarB(inputs[1],__PRETTY_FUNCTION__),c);
    }

    string str() const{
      return "cscalar_add_ReLU"+inp_str();
    }

  };
  

  class cscalar_add_ReLU_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    cscalar_add_ReLU_back_op(Cnode* r, Cnode* g, Cnode* x, float _c):
      Coperator(r,g,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_ReLU_back(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__),c);
    }

    string str() const{
      return "cscalar_add_ReLU_back"+inp_str();
    }
    
  };


  class cscalar_add_sigmoid_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_sigmoid_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_sigmoid(asCscalarB(inputs[1],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_sigmoid"+inp_str();
    }

  };
  

  class cscalar_add_sigmoid_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cscalar_add_sigmoid_back_op(Cnode* r, Cnode* g, Cnode* x):
      Coperator(r,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCscalarB(owner,__PRETTY_FUNCTION__).add_sigmoid_back(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCscalarB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "cscalar_add_sigmoid_back"+inp_str();
    }
    
  };


}

#endif

