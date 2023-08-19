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
#ifndef _Rbatcher_base
#define _Rbatcher_base

namespace Cengine{

  class Rbatcher_base{
  public:

    virtual ~Rbatcher_base(){}

    int id;
    virtual void push(Cnode* node)=0;
    //virtual void new_gang(Cnode* node){}
    virtual void release(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;

    virtual void release()=0; 
    virtual int flush()=0; 
    virtual int npending() const=0; 

    
  };

}

#endif 
