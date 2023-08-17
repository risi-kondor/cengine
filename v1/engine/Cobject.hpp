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
#ifndef _Cobject
#define _Cobject

#include "Cengine_base.hpp"


namespace Cengine{

  class Cobject{
  public:

    virtual ~Cobject(){}


  public:

    virtual int get_device() const{
      return 0;
    }

    virtual string classname() const{
      return "";
    }

    virtual string describe() const{
      return "";
    }

  };


  template<typename TYPE>
  inline TYPE& downcast(Cobject* x, const char* s){
    if(!x){
      CoutLock lk; cerr<<"\e[1mCengine error\e[0m ("<<s<<"): object does not exist"<<endl;
      exit(-1);
    }
    if(!dynamic_cast<TYPE*>(x)){
      CoutLock lk; 
      cerr<<"\e[1mCengine error\e[0m ("<<s<<"): Cobject is of type "<<x->classname()<<" instead of TYPE."<<endl;
      exit(-1);
    }
    return static_cast<TYPE&>(*x);
  }


  template<typename TYPE>
  inline TYPE& downcast(Cobject& x, const char* s){
    if(!dynamic_cast<TYPE&>(x)){
      CoutLock lk; 
      cerr<<"\e[1mCengine error\e[0m ("<<s<<"): Cobject is of type "<<x.classname()<<" instead of TYPE."<<endl;
      exit(-1);
    }
    return static_cast<TYPE&>(x);
  }


}

#endif
