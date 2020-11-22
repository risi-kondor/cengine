#ifndef _CtensorB_ops
#define _CtensorB_ops

#include "CtensorB.hpp"


namespace Cengine{


  // ---- Not in-place operators  ----------------------------------------------------------------------------


  class ctensor_conj_op: public Coperator{
  public:

    ctensor_conj_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).conj();
    }

    string str() const{
      return "ctensor_conj"+inp_str();
    }

  };
  

  class ctensor_transp_op: public Coperator{
  public:

    ctensor_transp_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).transp();
    }

    string str() const{
      return "ctensor_transp"+inp_str();
    }

  };
  

  class ctensor_herm_op: public Coperator{
  public:

    ctensor_herm_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).herm();
    }

    string str() const{
      return "ctensor_herm"+inp_str();
    }

  };
  

  // ---- Normalization  -------------------------------------------------------------------------------------


  class ctensor_add_col_norms_op: public CumulativeOp2<CtensorB,CtensorB>{
  public:

    using CumulativeOp2::CumulativeOp2;

    virtual void exec(CtensorB& R, const CtensorB& x){
      R.add_col_norms(x);
    }

    string str() const{
      return "ctensor_add_col_norms"+inp_str();
    }

  };


  class ctensor_add_col_norms_back_op: public CumulativeOp4<CtensorB,CtensorB,CtensorB,CtensorB>{
  public:

    using CumulativeOp4::CumulativeOp4;

    virtual void exec(CtensorB& R, const CtensorB& g, const CtensorB& x, const CtensorB& n){
      R.add_col_norms_back(g,x,n);
    }

    string str() const{
      return "ctensor_add_col_norms_back"+inp_str();
    }

  };


  class ctensor_divide_cols_op: public Coperator{
  public:

    ctensor_divide_cols_op(Cnode* r, Cnode* n):
      Coperator(r,n){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CTENSORB(inputs[0]).divide_cols(CTENSORB(inputs[1]));
    }

    string str() const{
      return "ctensor_divide_cols"+inp_str();
    }

  };


  class ctensor_add_divide_cols_op: public CumulativeOp3<CtensorB,CtensorB,CtensorB>{
  public:

    using CumulativeOp3::CumulativeOp3;

    virtual void exec(CtensorB& R, const CtensorB& x, const CtensorB& n){
      R.add_divide_cols(x,n);
    }

    string str() const{
      return "ctensor_add_divide_cols"+inp_str();
    }

  };


  class ctensor_add_divide_cols_back0_op: public CumulativeOp3<CtensorB,CtensorB,CtensorB>{
  public:

    using CumulativeOp3::CumulativeOp3;

    virtual void exec(CtensorB& R, const CtensorB& g, const CtensorB& n){
      R.add_divide_cols_back0(g,n);
    }

    string str() const{
      return "ctensor_add_divide_cols_back0"+inp_str();
    }

  };

  class ctensor_add_divide_cols_back1_op: public CumulativeOp4<CtensorB,CtensorB,CtensorB,CtensorB>{
  public:

    using CumulativeOp4::CumulativeOp4;

    virtual void exec(CtensorB& R, const CtensorB& g, const CtensorB& x, const CtensorB& n){
      R.add_divide_cols_back1(g,x,n);
    }

    string str() const{
      return "ctensor_add_divide_cols_back1"+inp_str();
    }

  };




  // ---- In-place operators  --------------------------------------------------------------------------------

  
  class ctensor_zero_op: public Coperator, public InPlaceOperator{ // DEPRECATED 
  public:

    ctensor_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner,__PRETTY_FUNCTION__).zero();
    }

    string str() const{
      return "ctensor_zero"+inp_str();
    }

  };
  
  
  class ctensor_set_zero_op: public Coperator, public InPlaceOperator{
  public:

    ctensor_set_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner,__PRETTY_FUNCTION__).zero();
    }

    string str() const{
      return "ctensor_zero"+inp_str();
    }

  };


  // ---- Access ---------------------------------------------------------------------------------------------

  
  class ctensor_get_element_op: public Coperator{
  public:

    Gindex ix;

    ctensor_get_element_op(Cnode* x, const Gindex& _ix):
      Coperator(x), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      const CtensorB& x=CTENSORB(inputs[0]);
      owner->obj=new CscalarB(x.nbu,x(ix),x.device);
    }

    string str() const{
      return "ctensor_get_element"+inp_str(ix);
    }

  };


  class ctensor_set_element_op: public Coperator, public InPlaceOperator{
  public:

    Gindex ix;

    ctensor_set_element_op(Cnode* r, Cnode* x, const Gindex& _ix):
      Coperator(r,x), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner,__PRETTY_FUNCTION__).set(ix,asCscalarB(inputs[1],__PRETTY_FUNCTION__).val);
    }

    string str() const{
      return "ctensor_set_element"+inp_str(ix);
    }

  };


  class ctensor_set_chunk_op: public Coperator, public InPlaceOperator{
  public:

    int ix;
    int offs;

    ctensor_set_chunk_op(Cnode* r, Cnode* x, const int _ix, const int _offs):
      Coperator(r,x), ix(_ix), offs(_offs){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).set_chunk(CTENSORB(inputs[1]),ix,offs);
    }

    string str() const{
      return "ctensor_set_chunk"+inp_str(ix,offs);
    }

  };


  class ctensor_to_device_op: public Coperator, public InPlaceOperator{
  public:

    int dev;

    ctensor_to_device_op(Cnode* r, const int _dev):
      Coperator(r), dev(_dev){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).to_device(dev);
    }

    string str() const{
      return "ctensor_to_device"+inp_str(dev);
    }

  };


  
}

#endif 
