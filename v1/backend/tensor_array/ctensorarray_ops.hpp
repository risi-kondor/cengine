#ifndef _ctensorarray_ops
#define _ctensorarray_ops

#include "CtensorArrayB.hpp"

namespace Cengine {

// ---- Not in-place operators
// ----------------------------------------------------------------------------

class ctensorarray_conj_op : public Coperator {
 public:
  ctensorarray_conj_op(Cnode* x) : Coperator(x) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = CTENSORARRAYB(inputs[0]).conj();
  }

  string str() const { return "ctensorarray_conj" + inp_str(); }
};

/*
class ctensorarray_transp_op: public Coperator{
public:

  ctensorarray_transp_op(Cnode* x):
    Coperator(x){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=CTENSORARRAYB(inputs[0]).transp();
  }

  string str() const{
    return "ctensorarray_transp"+inp_str();
  }

};
*/

/*
class ctensorarray_herm_op: public Coperator{
public:

  ctensorarray_herm_op(Cnode* x):
    Coperator(x){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=CTENSORARRAYB(inputs[0]).herm();
  }

  string str() const{
    return "ctensorarray_herm"+inp_str();
  }

};
*/

// ---- Normalization
// -------------------------------------------------------------------------------------

/*
class ctensorarray_add_col_norms_op: public CumulativeOp2<CtensorB,CtensorB>{
public:

  using CumulativeOp2::CumulativeOp2;

  virtual void exec(CtensorB& R, const CtensorB& x){
    R.add_col_norms(x);
  }

  string str() const{
    return "ctensorarray_add_col_norms"+inp_str();
  }

};


class ctensorarray_add_col_norms_back_op: public
CumulativeOp4<CtensorB,CtensorB,CtensorB,CtensorB>{ public:

  using CumulativeOp4::CumulativeOp4;

  virtual void exec(CtensorB& R, const CtensorB& g, const CtensorB& x, const
CtensorB& n){ R.add_col_norms_back(g,x,n);
  }

  string str() const{
    return "ctensor_add_col_norms_back"+inp_str();
  }

};


class ctensor_divide_cols_op: public Coperator{
public:

  ctensor_divide_cols_op(Cnode* r, Cnode* n):
    Coperator(r,n){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=CTENSORB(inputs[0]).divide_cols(CTENSORB(inputs[1]));
  }

  string str() const{
    return "ctensor_divide_cols"+inp_str();
  }

};


class ctensor_add_divide_cols_op: public
CumulativeOp3<CtensorB,CtensorB,CtensorB>{ public:

  using CumulativeOp3::CumulativeOp3;

  virtual void exec(CtensorB& R, const CtensorB& x, const CtensorB& n){
    R.add_divide_cols(x,n);
  }

  string str() const{
    return "ctensor_add_divide_cols"+inp_str();
  }

};


class ctensor_add_divide_cols_back0_op: public
CumulativeOp3<CtensorB,CtensorB,CtensorB>{ public:

  using CumulativeOp3::CumulativeOp3;

  virtual void exec(CtensorB& R, const CtensorB& g, const CtensorB& n){
    R.add_divide_cols_back0(g,n);
  }

  string str() const{
    return "ctensor_add_divide_cols_back0"+inp_str();
  }

};

class ctensor_add_divide_cols_back1_op: public
CumulativeOp4<CtensorB,CtensorB,CtensorB,CtensorB>{ public:

  using CumulativeOp4::CumulativeOp4;

  virtual void exec(CtensorB& R, const CtensorB& g, const CtensorB& x, const
CtensorB& n){ R.add_divide_cols_back1(g,x,n);
  }

  string str() const{
    return "ctensor_add_divide_cols_back1"+inp_str();
  }

};
*/

// ---- In-place operators
// --------------------------------------------------------------------------------

class ctensorarray_set_zero_op : public Coperator, public InPlaceOperator {
 public:
  ctensorarray_set_zero_op(Cnode* r) : Coperator(r) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).set_zero();
  }

  string str() const { return "ctensorarray_set_zero" + inp_str(); }
};

// ---- Access
// ---------------------------------------------------------------------------------------------

class ctensorarray_reshape_op : public Coperator, public InPlaceOperator {
 public:
  Gdims dims;

  ctensorarray_reshape_op(Cnode* x, const Gdims& _dims)
      : Coperator(x), dims(_dims) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).reshape_array(dims);
  }

  string str() const { return "ctensorarray_reshape" + inp_str(dims); }
};

class ctensorarray_get_cell_op : public Coperator {
 public:
  Gindex ix;

  ctensorarray_get_cell_op(Cnode* x, const Gindex& _ix)
      : Coperator(x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    CtensorArrayB& x = CTENSORARRAYB(inputs[0]);
    owner->obj = new CtensorB(x.dims, x.nbu, fill::raw, x.device);
    x.copy_cell_into(CTENSORB(owner), ix);
  }

  string str() const { return "ctensorarray_get_cell" + inp_str(ix); }
};

class ctensorarray_copy_cell_into_ctensor_op : public Coperator,
                                               public InPlaceOperator {
 public:
  Gindex ix;

  ctensorarray_copy_cell_into_ctensor_op(Cnode* r, Cnode* x, const Gindex& _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(inputs[1]).copy_cell_into(CTENSORB(owner), ix);
  }

  string str() const {
    return "ctensorarray_copy_cell_into_ctensor" + inp_str(ix);
  }
};

class ctensorarray_add_cell_into_ctensor_op : public Coperator,
                                              public CumulativeOperator,
                                              public InPlaceOperator {
 public:
  Gindex ix;

  ctensorarray_add_cell_into_ctensor_op(Cnode* r, Cnode* x, const Gindex& _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(inputs[1]).add_cell_into(CTENSORB(owner), ix);
  }

  string str() const { return "ctensor_add_cell_into_ctensor" + inp_str(ix); }
};

class ctensorarray_copy_ctensor_into_cell_op : public Coperator,
                                               public InPlaceOperator {
 public:
  Gindex ix;

  ctensorarray_copy_ctensor_into_cell_op(Cnode* r, Cnode* x, const Gindex& _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).copy_into_cell(ix, CTENSORB(inputs[1]));
  }

  string str() const {
    return "ctensorarray_copy_ctensor_into_cell" + inp_str(ix);
  }
};

class ctensorarray_add_ctensor_into_cell_op : public Coperator,
                                              public InPlaceOperator {
 public:
  Gindex ix;

  ctensorarray_add_ctensor_into_cell_op(Cnode* r, Cnode* x, const Gindex& _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).add_into_cell(ix, CTENSORB(inputs[1]));
  }

  string str() const {
    return "ctensorarray_add_ctensor_into_cell" + inp_str(ix);
  }
};

class ctensorarray_to_device_op : public Coperator, public InPlaceOperator {
 public:
  int dev;

  ctensorarray_to_device_op(Cnode* r, const int _dev)
      : Coperator(r), dev(_dev) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).to_device(dev);
  }

  string str() const { return "ctensorarray_to_device" + inp_str(dev); }
};

// ---- Broadcasting and reductions
// ------------------------------------------------------------------------

class ctensorarray_broadcast_op : public Coperator, public InPlaceOperator {
 public:
  int ix;

  ctensorarray_broadcast_op(Cnode* r, Cnode* x, const int _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).broadcast(ix, CTENSORARRAYB(inputs[1]));
  }

  string str() const { return "ctensorarray_broadcast" + inp_str(ix); }
};

class ctensorarray_add_broadcast_op : public Coperator, public InPlaceOperator {
 public:
  int ix;

  ctensorarray_add_broadcast_op(Cnode* r, Cnode* x, const int _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(owner).add_broadcast(ix, CTENSORARRAYB(inputs[1]));
  }

  string str() const { return "ctensorarray_add_broadcast" + inp_str(ix); }
};

class ctensorarray_add_collapse_op : public Coperator, public InPlaceOperator {
 public:
  int ix;

  ctensorarray_add_collapse_op(Cnode* r, Cnode* x, const int _ix)
      : Coperator(r, x), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORARRAYB(inputs[1]).collapse_add_into(CTENSORARRAYB(owner), ix);
  }

  string str() const { return "ctensorarray_add_collapse" + inp_str(ix); }
};

}  // namespace Cengine

#endif
