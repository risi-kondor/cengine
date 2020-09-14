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


  class ctensor_normalize_cols_op: public Coperator{
  public:

    ctensor_normalize_cols_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      //owner->obj=asCtensorB(inputs[0],__PRETTY_FUNCTION__).normalize_cols();
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_normalize_cols"+inp_str();
    }

  };


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


  // ---- In-place operators  --------------------------------------------------------------------------------

  
  class ctensor_zero_op: public Coperator, public InPlaceOperator{
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
  
  
}

#endif 
  
