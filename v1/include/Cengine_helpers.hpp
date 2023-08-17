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
#ifndef _Cengine_helpers
#define _Cengine_helpers

namespace Cengine{

  // ---- Convenience functions ------------------------------------------------------------------------------


  template<typename TYPE>
  void stdadd(const TYPE* beg, const TYPE* end, TYPE* dest){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]+=beg[i];
  }


  // ---- Transport ------------------------------------------------------------------------------------------


  template<typename TYPE>
  class pullin{
  public:
    TYPE& obj;
    int device;
    pullin(TYPE& _obj): obj(_obj){
      device=obj.device;
      obj.to_device(0);
    }
    pullin(const TYPE& _obj): 
      obj(const_cast<TYPE&>(_obj)){
      device=obj.device;
      obj.to_device(0);
    }
    ~pullin(){
      obj.to_device(device);
    }
  };


  template<typename TYPE>
  class _pullin{
  public:
    TYPE& obj;
    int dev;
    _pullin(TYPE& _obj): obj(_obj){
      dev=obj.dev;
      obj.to_device(0);
    }
    _pullin(const TYPE& _obj): 
      obj(const_cast<TYPE&>(_obj)){
      dev=obj.dev;
      obj.to_device(0);
    }
    ~_pullin(){
      obj.to_device(dev);
    }
  };


  template<typename TYPE>
  class pullin2{
  public:
    TYPE& obj0;
    TYPE& obj1;
    int device0;
    int device1;
    pullin2(const TYPE& _obj0, const TYPE& _obj1): 
      obj0(const_cast<TYPE&>(_obj0)),
      obj1(const_cast<TYPE&>(_obj1)){
      device0=obj0.device;
      device1=obj1.device;
      obj0.to_device(0);
      obj1.to_device(0);
    }
    ~pullin2(){
      obj0.to_device(device0);
      obj1.to_device(device1);
    }
  };


  template<typename TYPE>
  class tmpdev{
  public:
    TYPE& obj;
    int device;
    tmpdev(const int dev, TYPE& _obj): obj(_obj){
      device=obj.device;
      obj.to_device(dev);
    }
    tmpdev(const int dev, const TYPE& _obj): obj(const_cast<TYPE&>(_obj)){
      device=obj.device;
      obj.to_device(dev);
    }
    ~tmpdev(){
      obj.to_device(device);
    }
  };

  template<typename TYPE>
  class _tmpdev{
  public:
    TYPE& obj;
    int dev;
    _tmpdev(const int _dev, TYPE& _obj): obj(_obj){
      dev=obj.dev;
      obj.to_device(_dev);
    }
    _tmpdev(const int _dev, const TYPE& _obj): obj(const_cast<TYPE&>(_obj)){
      dev=obj.dev;
      obj.to_device(_dev);
    }
    ~_tmpdev(){
      obj.to_device(dev);
    }
  };

  template<typename TYPE>
  class tmpdev2{
  public:
    TYPE& obj0;
    TYPE& obj1;
    int device0;
    int device1;
    tmpdev2(const int dev, TYPE& _obj0, TYPE& _obj1): obj0(_obj0), obj1(_obj1){
      device0=obj0.device;
      device1=obj1.device;
      obj0.to_device(dev);
      obj1.to_device(dev);
    }
    tmpdev2(const int dev, const TYPE& _obj0, const TYPE& _obj1): 
      obj0(const_cast<TYPE&>(_obj0)), 
      obj1(const_cast<TYPE&>(_obj1)){
      device0=obj0.device;
      device1=obj1.device;
      obj0.to_device(dev);
      obj1.to_device(dev);
    }
    ~tmpdev2(){
      obj0.to_device(device0);
      obj1.to_device(device1);
    }
  };




}



#endif 

