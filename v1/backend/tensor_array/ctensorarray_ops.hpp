#ifndef _ctensorarray_ops
#define _ctensorarray_ops

#include "CtensorArrayB.hpp"


namespace Cengine{


  // ---- Not in-place operators  ----------------------------------------------------------------------------


  class ctensorarray_conj_op: public Coperator{
  public:

    ctensorarray_conj_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CTENSORARRAYB(inputs[0]).conj();
    }

    string str() const{
      return "ctensorarray_conj"+inp_str();
    }

  };
  

  /*
  class ctensorarray_transp_op: public Coperator{
  public:

    ctensorarray_transp_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CTENSORARRAYB(inputs[0]).transp();
    }

    string str() const{
      return "ctensorarray_transp"+inp_str();
    }

  };
  */
  

  /*
  class ctensorarray_herm_op: public Coperator{
  public:

    ctensorarray_herm_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CTENSORARRAYB(inputs[0]).herm();
    }

    string str() const{
      return "ctensorarray_herm"+inp_str();
    }

  };
  */


  // ---- Normalization  -------------------------------------------------------------------------------------

  /*
  class ctensorarray_add_col_norms_op: public CumulativeOp2<CtensorB,CtensorB>{
  public:

    using CumulativeOp2::CumulativeOp2;

    virtual void exec(CtensorB& R, const CtensorB& x){
      R.add_col_norms(x);
    }

    string str() const{
      return "ctensorarray_add_col_norms"+inp_str();
    }

  };


  class ctensorarray_add_col_norms_back_op: public CumulativeOp4<CtensorB,CtensorB,CtensorB,CtensorB>{
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
  */



  // ---- In-place operators  --------------------------------------------------------------------------------

  
  class ctensorarray_set_zero_op: public Coperator, public InPlaceOperator{
  public:

    ctensorarray_set_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).zero();
    }

    string str() const{
      return "ctensor_zero"+inp_str();
    }

  };


  // ---- Access ---------------------------------------------------------------------------------------------

  
  class ctensorarray_add_cell_into_op: public Coperator{
  public:

    Gindex ix;

    ctensorarray_add_cell_into_op(Cnode* r, Cnode* x, const Gindex& _ix):
      Coperator(r,x), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(inputs[0]).add_cell_into(CSCALARARRAYB(inputs[1]).val);
    }

    string str() const{
      return "ctensor_set_element"+inp_str(ix);
    }

  };

  /*
  class ctensorarray_set_element_op: public Coperator, public InPlaceOperator{
  public:

    Gindex ix;

    ctensorarray_set_element_op(Cnode* r, Cnode* x, const Gindex& _ix):
      Coperator(r,x), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).set(ix,CSCALARARRAYB(inputs[1]).val);
    }

    string str() const{
      return "ctensor_set_element"+inp_str(ix);
    }

  };
  */

  /*
  class ctensor_set_chunk_op: public Coperator, public InPlaceOperator{
  public:

    int ix;
    int offs;

    ctensor_set_chunk_op(Cnode* r, Cnode* x, const int _ix, const int _offs):
      Coperator(r,x), ix(_ix), offs(_offs){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).set_chunk(CTENSORARRAYB(inputs[1]),ix,offs);
    }

    string str() const{
      return "ctensor_set_chunk"+inp_str(ix,offs);
    }

  };
  */


  class ctensorarray_to_device_op: public Coperator, public InPlaceOperator{
  public:

    int dev;

    ctensorarray_to_device_op(Cnode* r, const int _dev):
      Coperator(r), dev(_dev){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).to_device(dev);
    }

    string str() const{
      return "ctensorarray_to_device"+inp_str(dev);
    }

  };


  
}

#endif 
