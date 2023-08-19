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
#ifndef _CengineHelpers
#define _CengineHelpers

#include "Coperator.hpp"


namespace Cengine{


  class diamond_op: public Coperator{
  public:

    diamond_op(Cnode* x, Cnode* y):
      Coperator(x,y){}

    virtual void exec(){
      owner->obj=inputs[0]->obj;
      inputs[0]->is_view=true; 
    }

    string str() const{
      return "diamond"+inp_str();
    }
    
  };




}
#endif

