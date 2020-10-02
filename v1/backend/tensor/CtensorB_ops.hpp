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
  
  
}

#endif 
  
  /*
  class ctensor_add_col_norms_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_col_norms_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      owner->obj=inputs[0]->obj;
      CTENSORB(inputs[0]).add_col_norms(CTENSORB(inputs[1]));
    }

    string str() const{
      return "ctensor_normalize_cols"+inp_str();
    }

  };
  */

  /*
  class ctensor_add_normalize_cols_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_normalize_cols_back_op(Cnode* xg, Cnode* g, Cnode* x):
      Coperator(xg,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      //owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).add_normalize_cols_back(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "ctensor_add_normalize_cols_back"+inp_str();
    }

  };
  */
  /*
  class ctensor_add_normalize_cols_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_normalize_cols_back_op(Cnode* xg, Cnode* g, Cnode* x):
      Coperator(xg,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      //owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).add_normalize_cols_back(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "ctensor_add_normalize_cols_back"+inp_str();
    }

  };
  */
  /*
  class ctensor_add_normalize_cols_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_normalize_cols_back_op(Cnode* xg, Cnode* g, Cnode* x):
      Coperator(xg,g,x){}

    virtual void exec(){
      assert(!owner->obj);
      //owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).add_normalize_cols_back(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "ctensor_add_normalize_cols_back"+inp_str();
    }

  };
  */
