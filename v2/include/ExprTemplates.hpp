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
#ifndef _ExprTemplates
#define _ExprTemplates


namespace Cengine{

  template<typename OBJ>
  class Transpose{
  public:
    const OBJ& obj;
    explicit Transpose(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Conjugate{
  public:
    const OBJ& obj;
    explicit Conjugate(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Hermitian{
  public:
    const OBJ& obj;
    explicit Hermitian(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Broadcast{
  public:
    const OBJ& obj;
    explicit Broadcast(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Scatter{
  public:
    const OBJ& obj;
    explicit Scatter(const OBJ& _obj):obj(_obj){}
  };



  template<typename OBJ>
  Broadcast<OBJ> broadcast(const OBJ& x){
    return Broadcast<OBJ>(x);
  }

  template<typename OBJ>
  Scatter<OBJ> scatter(const OBJ& x){
    return Scatter<OBJ>(x);
  }


}

#endif
