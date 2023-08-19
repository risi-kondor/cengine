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
#ifndef _Batcher
#define _Batcher

#include "Cengine_base.hpp"


namespace Cengine{


  class Batcher{
  public:

    //Cengine* engine;

    virtual ~Batcher(){}

    virtual void push(Coperator* op)=0;
    virtual void release(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;

    virtual void release()=0; 
    virtual int flush()=0; 
    virtual int npending() const=0; 

  };


}


#endif 
