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
#ifndef _ctensor_signature
#define _ctensor_signature

#include "Cengine_base.hpp"


namespace Cengine{

  class ctensor_signature{
  public:

    Gdims dims1;

    ctensor_signature(const Gdims& _dims1): 
      dims1(_dims1){
    }

    bool operator==(const ctensor_signature& x) const{
      return (dims1==x.dims1);}

    string str() const{
      return "("+dims1.str()+")";}

  };
  
}


namespace std{

  template<>
  struct hash<::Cengine::ctensor_signature>{
  public:
    size_t operator()(const ::Cengine::ctensor_signature& ix) const{
      size_t t=(hash<::Cengine::Gdims>()(ix.dims1)<<1);
      return t;
    }
  };

}


#endif
