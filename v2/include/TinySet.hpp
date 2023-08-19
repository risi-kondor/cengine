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
#ifndef _TinySet
#define _TinySet

#include "Cengine_base.hpp"

namespace Cengine{

  template<typename TYPE>
  class TinySet{
  public:

    vector<TYPE> v;

  public:
    
    int size() const{
      int t=0;
      for(auto p:v) if(p!=nullptr) t++;
      return t;
    }

    bool find(const TYPE x) const{
      for(auto p:v) if(p==x) return true;
      return false;
    }

    bool insert(TYPE x){
      if(find(x)) return false;
      v.push_back(x);
      return true;
    }

    bool erase(const TYPE x){
      for(auto& p:v)
	if(p==x){
	  p=nullptr;
	  return true;
	}
      return false;
    }

    void clear(){
      v.clear();
    }


  public:

    void map(std::function<void(TYPE x)> lambda) const{
      for(auto p:v) 
	if(p) lambda(p);
    }

  };

}

#endif
