#ifndef _ctensorarray_arr_ops
#define _ctensorarray_arr_ops

#include "CtensorArrayB.hpp"
#include "ctensor_signature.hpp"


namespace Cengine{


  class ctensorarray_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add(CTENSORARRAYB(inputs[1]));
    }


  public:

    string str() const{
      return "ctensorarray_add"+inp_str();
    }

    static string classname(){
      return "ctensorarray_add_op";
    }
    
  };
  


  class ctensorarray_add_prod_c_A_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_prod_c_A_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      //CTENSORARRAYB(owner).add_prod(CSCALARARRAYB(inputs[1]),CTENSORARRAYB(inputs[2]));
    }


  public:

    string str() const{
      return "ctensorarray_add"+inp_str();
    }

    static string classname(){
      return "ctensorarray_add_prod_c_A_op";
    }

  };
  

}

#endif 

