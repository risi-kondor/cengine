/*
 * This file is part of Cengine, an asynchronous C++/CUDA compute engine. 
 *  
 * Copyright (c) 2020- Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */
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

