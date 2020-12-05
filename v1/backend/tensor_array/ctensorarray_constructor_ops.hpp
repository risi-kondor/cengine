#ifndef _ctensorarray_constructor_ops
#define _ctensorarray_constructor_ops

#include "CtensorArrayB.hpp"

namespace Cengine{


  template<typename FILLTYPE>
  class new_ctensorarray_op: public Coperator{
  public:

    Gdims adims;
    Gdims dims;
    int nbu;
    int device;

    new_ctensorarray_op(const Gdims& _adims, const Gdims& _dims, const int _nbu=-1, const int _device=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims,dims,nbu,FILLTYPE(),device);
    }

    string str() const{
      return "new_ctensorarray"+adims.str()+dims.str();
    }

  };


  class new_ctensorarray_zero_op: public Coperator{
  public:

    Gdims adims;
    Gdims dims;
    int nbu;
    int device;

    new_ctensorarray_zero_op(const Gdims& _adims, const Gdims& _dims, const int _nbu=-1, const int _device=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims,dims,nbu,fill::zero,device);
    }

    string str() const{
      return "ctensorarray_zero"+adims.str()+dims.str();
    }

  };


  class new_ctensorarray_ones_op: public Coperator{
  public:

    Gdims adims;
    Gdims dims;
    int nbu;
    int device;

    new_ctensorarray_ones_op(const Gdims& _adims, const Gdims& _dims, const int _nbu=-1, const int _device=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims,dims,nbu,fill::ones,device);
    }

    string str() const{
      return "ctensorarray_ones"+adims.str()+dims.str();
    }

  };


  /*
  class new_ctensorarray_identity_op: public Coperator{
  public:

    Gdims adims;
    Gdims dims;
    int nbu;
    int device;

    new_ctensorarray_identity_op(const Gdims& _adims, const Gdims& _dims, const int _nbu=-1, const int _device=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims, dims,nbu,fill::identity,device);
    }

    string str() const{
      return "ctensorarray_identity"+adims.str()+dims.str();
    }

  };
  */


  class new_ctensorarray_sequential_op: public Coperator{
  public:

    Gdims adims;
    Gdims dims;
    int nbu;
    int device;

    new_ctensorarray_sequential_op(const Gdims& _adims, const Gdims& _dims, const int _nbu=-1, const int _device=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims,dims,nbu,fill::sequential,device);
    }

    string str() const{
      return "ctensorarray_sequential"+adims.str()+dims.str();
    }

  };


  class new_ctensorarray_gaussian_op: public Coperator{
  public:

    Gdims adims;
    Gdims dims;
    int nbu;
    int device;
    float c=1.0;

    new_ctensorarray_gaussian_op(const Gdims& _adims, const Gdims& _dims, const int _nbu=-1, const int _device=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_device){
    }

    new_ctensorarray_gaussian_op(const Gdims& _adims, const Gdims& _dims, const int _nbu, const float _c, const int _device=0):
      dims(_dims), nbu(_nbu), c(_c), device(_device){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims,dims,nbu,fill::gaussian,c,device);
    }

    string str() const{
      return "ctensorarray_gaussian"+adims.str()+dims.str();
    }
    
  };


  class ctensorarray_copy_op: public Coperator{
  public:

    ctensorarray_copy_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(CTENSORARRAYB(inputs[0]),nowarn);
    }

    string str() const{
      return "ctensorarray_copy("+inputs[0]->ident()+")";
    }
    
  };


  /*
  class new_ctensorarray_fn2_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int device;
    std::function<complex<float>(const int, const int)> fn; 

    new_ctensorarray_fn2_op(const Gdims& _adims, const Gdims& _dims, const int _nbu, 
      function<complex<float>(const int, const int)> _fn, const int _dev=0):
      adims(_adims), dims(_dims), nbu(_nbu), device(_dev), fn(_fn){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorArrayB(adims,dims,nbu,fn,device);
    }

    string str() const{
      return "ctensorarray_fn2"+adims.str()+dims.str();
    }

  };
  */

  /*
  class ctensorarray_apply_op: public Coperator{
  public:

    std::function<complex<float>(const complex<float>)> fn; 

    ctensorarray_apply_op(Cnode* x, std::function<complex<float>(const complex<float>)> _fn):
      Coperator(x), fn(_fn){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(CTENSORARRAYB(inputs[0]),fn);
    }
    
    string str() const{
      return "ctensorarray_apply"+inp_str();
    }

  };
  */

  /*
  class ctensorarray_apply2_op: public Coperator{
  public:

    std::function<complex<float>(const int, const int, const complex<float>)> fn; 

    ctensorarray_apply2_op(Cnode* x, std::function<complex<float>(const int, const int, const complex<float>)> _fn):
      Coperator(x), fn(_fn){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new CtensorB(CTENSORARRAYB(inputs[0]),fn);
    }
    
    string str() const{
      return "ctensorarray_apply"+inp_str();
    }

  };
  */

}

#endif 
