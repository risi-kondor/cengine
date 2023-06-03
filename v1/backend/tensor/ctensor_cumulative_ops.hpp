#ifndef _CtensorB_cumulative_ops
#define _CtensorB_cumulative_ops

#include "CtensorB.hpp"

namespace Cengine {

/*
class ctensor_add_op: public Coperator, public CumulativeOperator, public
InPlaceOperator{ public:

  ctensor_add_op(Cnode* r, Cnode* x):
    Coperator(r,x){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner,__PRETTY_FUNCTION__).add(asCtensorB(inputs[1],__PRETTY_FUNCTION__));
  }

  string str() const{
    return "ctensor_add"+inp_str();
  }

};
*/

class ctensor_add_conj_op : public Coperator,
                            public CumulativeOperator,
                            public InPlaceOperator {
 public:
  ctensor_add_conj_op(Cnode* r, Cnode* x) : Coperator(r, x) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_conj(asCtensorB(inputs[1], __PRETTY_FUNCTION__));
  }

  string str() const { return "ctensor_add_conj" + inp_str(); }
};

class ctensor_add_transp_op : public Coperator,
                              public CumulativeOperator,
                              public InPlaceOperator {
 public:
  ctensor_add_transp_op(Cnode* r, Cnode* x) : Coperator(r, x) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_transp(asCtensorB(inputs[1], __PRETTY_FUNCTION__));
  }

  string str() const { return "ctensor_add_transp" + inp_str(); }
};

class ctensor_add_herm_op : public Coperator,
                            public CumulativeOperator,
                            public InPlaceOperator {
 public:
  ctensor_add_herm_op(Cnode* r, Cnode* x) : Coperator(r, x) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_herm(asCtensorB(inputs[1], __PRETTY_FUNCTION__));
  }

  string str() const { return "ctensor_add_herm" + inp_str(); }
};

class ctensor_add_sum_op : public Coperator,
                           public CumulativeOperator,
                           public InPlaceOperator {
 public:
  ctensor_add_sum_op(Cnode* r, const vector<Cnode*> x) : Coperator(r, x) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    vector<CFtensor*> v(inputs.size() - 1);
    for (int i = 0; i < inputs.size() - 1; i++) {
      v[i] = &asCtensorB(inputs[i + 1], __PRETTY_FUNCTION__);
    }
    asCtensorB(owner, __PRETTY_FUNCTION__).add_sum(v);
  }

  string str() const { return "add_ctensor_sum" + inp_str(); }
};

class ctensor_add_to_slice_op : public Coperator,
                                public CumulativeOperator,
                                public InPlaceOperator {
 public:
  int ix;
  int offs;

  ctensor_add_to_slice_op(Cnode* r, Cnode* x, const int _ix, const int _offs)
      : Coperator(r, x), ix(_ix), offs(_offs) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_to_slice(asCtensorB(inputs[1], __PRETTY_FUNCTION__), ix, offs);
  }

  string str() const { return "ctensor_add_to_slice" + inp_str(ix, offs); }
};

class ctensor_add_to_chunk_op : public Coperator,
                                public CumulativeOperator,
                                public InPlaceOperator {
 public:
  int ix;
  int offs;

  ctensor_add_to_chunk_op(Cnode* r, Cnode* x, const int _ix, const int _offs)
      : Coperator(r, x), ix(_ix), offs(_offs) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_to_chunk(asCtensorB(inputs[1], __PRETTY_FUNCTION__), ix, offs);
  }

  string str() const { return "ctensor_add_to_chunk" + inp_str(ix, offs); }
};

class ctensor_add_to_slices_op : public Coperator,
                                 public CumulativeOperator,
                                 public InPlaceOperator {
 public:
  int ix;

  ctensor_add_to_slices_op(Cnode* r, vector<Cnode*> v, const int _ix)
      : Coperator(r, v), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    vector<const CFtensor*> v(inputs.size() - 1);
    for (int i = 0; i < inputs.size() - 1; i++) {
      v[i] = &asCtensorB(inputs[i + 1], __PRETTY_FUNCTION__);
    }
    asCtensorB(owner, __PRETTY_FUNCTION__).add_to_slices(v, ix);
  }

  string str() const { return "ctensor_add_to_slices"; }
};

class ctensor_add_slice_op : public Coperator,
                             public CumulativeOperator,
                             public InPlaceOperator {
 public:
  int ix;
  int offs;

  ctensor_add_slice_op(Cnode* r, Cnode* x, const int _ix, const int _offs)
      : Coperator(r, x), ix(_ix), offs(_offs) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_slice(asCtensorB(inputs[1], __PRETTY_FUNCTION__), ix, offs);
  }

  string str() const { return "ctensor_add_slice" + inp_str(ix, offs); }
};

class ctensor_add_chunk_op : public Coperator,
                             public CumulativeOperator,
                             public InPlaceOperator {
 public:
  int ix;
  int offs;
  int n;

  ctensor_add_chunk_op(Cnode* r,
                       Cnode* x,
                       const int _ix,
                       const int _offs,
                       const int _n)
      : Coperator(r, x), ix(_ix), offs(_offs), n(_n) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_chunk(asCtensorB(inputs[1], __PRETTY_FUNCTION__), ix, offs, n);
  }

  string str() const { return "ctensor_add_chunk" + inp_str(ix, offs, n); }
};

// ---- Subtract
// -------------------------------------------------------------------------------------------

class ctensor_subtract_op : public Coperator,
                            public CumulativeOperator,
                            public InPlaceOperator {
 public:
  ctensor_subtract_op(Cnode* r, Cnode* x) : Coperator(r, x) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORB(owner).subtract(CTENSORB(inputs[1]));
  }

  string str() const { return "ctensor_subtract" + inp_str(); }
};

// ---- Products
// -------------------------------------------------------------------------------------------

class ctensor_add_times_real_op : public Coperator,
                                  public CumulativeOperator,
                                  public InPlaceOperator {
 public:
  float c;

  ctensor_add_times_real_op(Cnode* r, Cnode* A, float _c)
      : Coperator(r, A), c(_c) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add(asCtensorB(inputs[1], __PRETTY_FUNCTION__), c);
  }

  string str() const { return "ctensor_add_times_real" + inp_str(); }
};

class ctensor_add_times_complex_op : public Coperator,
                                     public CumulativeOperator,
                                     public InPlaceOperator {
 public:
  complex<float> c;

  ctensor_add_times_complex_op(Cnode* r, Cnode* A, complex<float> _c)
      : Coperator(r, A), c(_c) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add(asCtensorB(inputs[1], __PRETTY_FUNCTION__), c);
  }

  string str() const { return "ctensor_add_teims_complex" + inp_str(); }
};

class ctensor_add_prod_r_A_op : public Coperator,
                                public CumulativeOperator,
                                public InPlaceOperator {
 public:
  ctensor_add_prod_r_A_op(Cnode* r, Cnode* c, Cnode* A) : Coperator(r, c, A) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_prod(asRscalarB(inputs[1], __PRETTY_FUNCTION__),
                  asCtensorB(inputs[2], __PRETTY_FUNCTION__));
  }

  int batcher_id() const { return 100; }

  string str() const { return "ctensor_add_prod_r_A" + inp_str(); }
};

/*
class ctensor_add_prod_c_A_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_prod_c_A_op(Cnode* r, Cnode* c, Cnode* A):
    Coperator(r,c,A){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner,__PRETTY_FUNCTION__).add_prod(asCscalarB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
  }

  string str() const{
    return "ctensor_add_prod_cA"+inp_str();
  }

};
*/

class ctensor_add_prod_cc_A_op : public Coperator,
                                 public CumulativeOperator,
                                 public InPlaceOperator {
 public:
  ctensor_add_prod_cc_A_op(Cnode* r, Cnode* c, Cnode* A) : Coperator(r, c, A) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_prod_cconj(asCscalarB(inputs[1], __PRETTY_FUNCTION__),
                        asCtensorB(inputs[2], __PRETTY_FUNCTION__));
  }

  string str() const { return "ctensor_add_prod_cc_A" + inp_str(); }
};

class ctensor_add_prod_c_Ac_op : public Coperator,
                                 public CumulativeOperator,
                                 public InPlaceOperator {
 public:
  ctensor_add_prod_c_Ac_op(Cnode* r, Cnode* c, Cnode* A) : Coperator(r, c, A) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORB(owner).add_prod_c_times_conj(CSCALARB(inputs[1]),
                                          CTENSORB(inputs[2]));
  }

  string str() const { return "ctensor_add_prod_c_Ac" + inp_str(); }
};

/*
class ctensor_add_Mprod_AT_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_Mprod_AT_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner).add_Mprod_AT<0>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    //owner->computed=true;
  }

  string str() const{
    return "ctensor_add_Mprod_AT"+inp_str();
  }

};
*/

/*
class ctensor_add_Mprod_TA_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_Mprod_TA_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner).add_Mprod_TA<0>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    //owner->computed=true;
  }

  string str() const{
    return "ctensor_add_Mprod_TA"+inp_str();
  }

};
*/

/*
class ctensor_add_Mprod_AC_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_Mprod_AC_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner).add_Mprod<2>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    //owner->computed=true;
  }

  string str() const{
    return "ctensor_add_Mprod_AC"+inp_str();
  }

};
*/

/*
class ctensor_add_Mprod_TC_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_Mprod_TC_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner).add_Mprod_TA<2>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    //owner->computed=true;
  }

  string str() const{
    return "ctensor_add_Mprod_TC"+inp_str();
  }

};
*/

/*
class ctensor_add_Mprod_AH_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_Mprod_AH_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner).add_Mprod_AT<2>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
  }

  string str() const{
    return "ctensor_add_Mprod_AH"+inp_str();
  }

};
*/

/*
class ctensor_add_Mprod_HA_op: public Coperator, public CumulativeOperator,
public InPlaceOperator{ public:

  ctensor_add_Mprod_HA_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(owner).add_Mprod_TA<1>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
  }

  string str() const{
    return "ctensor_add_Mprod_HA"+inp_str();
  }

};
*/

class ctensor_add_ReLU_op : public Coperator,
                            public CumulativeOperator,
                            public InPlaceOperator {
 public:
  float c = 0;

  ctensor_add_ReLU_op(Cnode* r, Cnode* x, float _c) : Coperator(r, x), c(_c) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_LeakyReLU(asCtensorB(inputs[1], __PRETTY_FUNCTION__), c);
  }

  string str() const { return "ctensor_add_ReLU" + inp_str(); }
};

class ctensor_add_ReLU_back_op : public Coperator,
                                 public CumulativeOperator,
                                 public InPlaceOperator {
 public:
  float c = 0;

  ctensor_add_ReLU_back_op(Cnode* r, Cnode* g, Cnode* x, float _c)
      : Coperator(r, g, x), c(_c) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_LeakyReLU_back(asCtensorB(inputs[1], __PRETTY_FUNCTION__),
                            asCtensorB(inputs[2], __PRETTY_FUNCTION__), c);
  }

  string str() const { return "ctensor_add_ReLU_back" + inp_str(); }
};

/*
class ctensor_add_inp_op: public Coperator, public CumulativeOperator, public
InPlaceOperator{ public:

  ctensor_add_inp_op(Cnode* R, Cnode* A, Cnode* B):
    Coperator(R,A,B){}

  virtual void exec(){
    assert(!owner->obj);
    owner->obj=inputs[0]->obj;
    asCtensorB(inputs[1],__PRETTY_FUNCTION__).add_inp_into(asCscalarB(owner,__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
  }

  string str() const{
    return "ctensor_add_inp"+inp_str();
  }

};
*/

class ctensor_add_element_op : public Coperator,
                               public CumulativeOperator,
                               public InPlaceOperator {
 public:
  Gindex ix;

  ctensor_add_element_op(Cnode* r, Cnode* A, const Gindex& _ix)
      : Coperator(r, A), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(inputs[1], __PRETTY_FUNCTION__)
        .add_element_into(asCscalarB(owner, __PRETTY_FUNCTION__), ix);
  }

  string str() const { return "ctensor_add_element" + inp_str(ix); }
};

class ctensor_add_to_element_op : public Coperator,
                                  public CumulativeOperator,
                                  public InPlaceOperator {
 public:
  Gindex ix;

  ctensor_add_to_element_op(Cnode* R, Cnode* a, const Gindex& _ix)
      : Coperator(R, a), ix(_ix) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    asCtensorB(owner, __PRETTY_FUNCTION__)
        .add_to_element(ix, asCscalarB(inputs[1], __PRETTY_FUNCTION__));
  }

  string str() const { return "ctensor_add_element" + inp_str(ix); }
};

}  // namespace Cengine

#endif
