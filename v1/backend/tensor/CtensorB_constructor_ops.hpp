#ifndef _CtensorB_constructor_ops
#define _CtensorB_constructor_ops

#include "CtensorB.hpp"


namespace Cengine{


  class new_ctensor_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;

    new_ctensor_op(const Gdims& _dims, const int _nbu=-1, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(dims,nbu,fill::raw,device);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor"+dims.str();
    }

  };


  class new_ctensor_zero_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;

    new_ctensor_zero_op(const Gdims& _dims, const int _nbu=-1, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(dims,nbu,fill::zero,device);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_zero"+dims.str();
    }

  };


  class new_ctensor_ones_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;

    new_ctensor_ones_op(const Gdims& _dims, const int _nbu=-1, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(dims,nbu,fill::ones,device);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_ones"+dims.str();
    }

  };


  class new_ctensor_identity_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;

    new_ctensor_identity_op(const Gdims& _dims, const int _nbu=-1, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(dims,nbu,fill::identity,device);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_identity"+dims.str();
    }

  };


  class new_ctensor_sequential_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;

    new_ctensor_sequential_op(const Gdims& _dims, const int _nbu=-1, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(dims,nbu,fill::sequential,device);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_sequential"+dims.str();
    }

  };


  class new_ctensor_gaussian_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;

    new_ctensor_gaussian_op(const Gdims& _dims, const int _nbu=-1, const int _device=0):
      dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(dims,nbu,fill::gaussian,device);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_gaussian"+dims.str();
    }
    
  };


  class new_ctensor_from_gtensor_op: public Coperator{
  public:

    Gtensor<complex<float> > x;
    int nbu;
    int device;

    new_ctensor_from_gtensor_op(const Gtensor<complex<float> >& _x, const int _nbu=-1, const int _device=0):
      x(_x,nowarn), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(x,device);
    }

    string str() const{
      return "ctensor()";
    }

  };


  class ctensor_copy_op: public Coperator{
  public:

    ctensor_copy_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(asCtensorB(inputs[0]),nowarn);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_copy("+inputs[0]->ident()+")";
    }
    
  };


}

#endif 
