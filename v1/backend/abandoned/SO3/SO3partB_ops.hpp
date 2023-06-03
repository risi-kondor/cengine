#ifndef _SO3partB_ops
#define _SO3partB_ops

#include "CtensorB.hpp"
#include "SO3partB.hpp"

namespace GEnet {

// ---- Constructors
// --------------------------------------------------------------------------------------

class new_SO3part_from_Gtensor_op : public Coperator {
 public:
  Gtensor<complex<float>> x;
  int nbu;
  int device;

  new_SO3part_from_Gtensor_op(const Gtensor<complex<float>>& _x,
                              const int _device = 0)
      : x(_x, nowarn), device(_device) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = new SO3partB(x, device);
  }

  string str() const { return "new_SO3part()"; }
};

class new_spharm_op : public Coperator {
 public:
  int l;
  float x, y, z;
  int nbu;
  int device;

  new_spharm_op(const int _l,
                const float _x,
                const float _y,
                const float _z,
                const int _nbu = -1,
                const int _device = 0)
      : l(_l), x(_x), y(_y), z(_z), nbu(_nbu), device(_device) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = new SO3partB(l, x, y, z, nbu, device);
  }

  string str() const {
    return "spharm(" + to_string(l) + "," + to_string(x) + "," + to_string(y) +
           "," + to_string(z) + ")";
  }
};

// ---- In-place operators
// --------------------------------------------------------------------------------

class SO3part_add_CGproduct_op : public Coperator,
                                 public CumulativeOperator,
                                 public InPlaceOperator {
 public:
  int offs;

  SO3part_add_CGproduct_op(Cnode* r, Cnode* x, Cnode* y, const int _offs)
      : Coperator(r, x, y), offs(_offs) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    static_cast<SO3partB&>(asCtensorB(owner))
        .add_CGproduct(asCtensorB(inputs[1]), asCtensorB(inputs[2]), offs);
    // owner->computed=true;
  }

  string str() const { return "SO3part_add_CGproduct" + inp_str(); }
};

class SO3part_add_CGproduct_back0_op : public Coperator,
                                       public CumulativeOperator,
                                       public InPlaceOperator {
 public:
  int offs;

  SO3part_add_CGproduct_back0_op(Cnode* r, Cnode* g, Cnode* y, const int _offs)
      : Coperator(r, g, y), offs(_offs) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    static_cast<SO3partB&>(asCtensorB(owner))
        .add_CGproduct_back0(asCtensorB(inputs[1]), asCtensorB(inputs[2]),
                             offs);
    // owner->computed=true;
  }

  string str() const { return "SO3part_add_CGproduct_back0" + inp_str(); }
};

class SO3part_add_CGproduct_back1_op : public Coperator,
                                       public CumulativeOperator,
                                       public InPlaceOperator {
 public:
  int offs;

  SO3part_add_CGproduct_back1_op(Cnode* r, Cnode* g, Cnode* x, const int _offs)
      : Coperator(r, g, x), offs(_offs) {}

  virtual void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    static_cast<SO3partB&>(asCtensorB(owner))
        .add_CGproduct_back1(asCtensorB(inputs[1]), asCtensorB(inputs[2]),
                             offs);
    // owner->computed=true;
  }

  string str() const { return "SO3part_add_CGproduct_back1" + inp_str(); }
};

}  // namespace GEnet

#endif
