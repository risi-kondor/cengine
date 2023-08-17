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
#ifndef _cmatrix_ops
#define _cmatrix_ops

#include "CmatrixB.hpp"

namespace Cengine{


  template<typename FILLTYPE>
  class new_cmatrix_op: public Coperator{
  public:

    int n0; 
    int n1;
    int nbu;
    int dev;

    new_cmatrix_op(const int _n0, const int _n1, const int _nbu=-1, const int _dev=0):
      n0(_n0), n1(_n1), nbu(_nbu), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CmatrixB(n0,n1,nbu,FILLTYPE(),dev);
    }

    string str() const{
      return "new_cmatrix("+to_string(n0)+","+to_string(n1)+")";
    }

  };


  class cmatrix_copy_op: public Coperator{
  public:

    cmatrix_copy_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CmatrixB(CMATRIXB(inputs[0]));
    }

    string str() const{
      return "cmatrix_copy("+inputs[0]->ident()+")";
    }

  };


  class new_cmatrix_from_gtensor_op: public Coperator{
  public:

    Gtensor<complex<float> > x;
    int nbu;
    int dev;

    new_cmatrix_from_gtensor_op(const Gtensor<complex<float> >& _x, const int _nbu=-1, const int _dev=0):
      x(_x,nowarn), nbu(_nbu), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CmatrixB(x,dev);
    }

    string str() const{
      return "cmatrix()";
    }

  };


  class cmatrix_create_op: public Coperator{
  public:

    int n0;
    int n1;
    int nbu;

    std::function<complex<float>(const int, const int)> fn; 

    cmatrix_create_op(const int _n0, const int _n1, 
      std::function<complex<float>(const int, const int)> _fn):
      fn(_fn), n0(_n0), n1(_n1), nbu(-1){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CmatrixB(n0,n1,fn);
    }
    
    string str() const{
      return "cmatrix_create"+inp_str();
    }

  };


  class cmatrix_apply_op: public Coperator{
  public:

    std::function<complex<float>(const int, const int, const complex<float>)> fn; 

    cmatrix_apply_op(Cnode* x, std::function<complex<float>(const int, const int, const complex<float>)> _fn):
      Coperator(x), fn(_fn){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CmatrixB(CMATRIXB(inputs[0]),fn);
    }
    
    string str() const{
      return "cmatrix_apply"+inp_str();
    }

  };


  // ---- In-place operators  --------------------------------------------------------------------------------

  
  class cmatrix_set_zero_op: public Coperator, public InPlaceOperator{
  public:

    cmatrix_set_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).zero();
    }

    string str() const{
      return "cmatrix_zero"+inp_str();
    }

  };


  class cmatrix_to_device_op: public Coperator, public InPlaceOperator{
  public:

    int dev;

    cmatrix_to_device_op(Cnode* r, const int _dev):
      Coperator(r), dev(_dev){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).to_device(dev);
    }

    string str() const{
      return "cmatrix_to_device"+inp_str(dev);
    }

  };


  // ---- Not in-place operators  ----------------------------------------------------------------------------


  class cmatrix_conj_op: public Coperator{
  public:

    cmatrix_conj_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CMATRIXB(inputs[0]).conj();
    }

    string str() const{
      return "cmatrix_conj"+inp_str();
    }

  };
  

  class cmatrix_transp_op: public Coperator{
  public:

    cmatrix_transp_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CMATRIXB(inputs[0]).transp();
    }

    string str() const{
      return "cmatrix_transp"+inp_str();
    }

  };
  

  class cmatrix_herm_op: public Coperator{
  public:

    cmatrix_herm_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CMATRIXB(inputs[0]).herm();
    }

    string str() const{
      return "cmatrix_herm"+inp_str();
    }

  };
  





}


#endif 

