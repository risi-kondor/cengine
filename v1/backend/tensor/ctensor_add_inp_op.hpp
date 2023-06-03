#ifndef _ctensor_add_inp_op
#define _ctensor_add_inp_op

#include "CscalarBreducer.hpp"
#include "CtensorBpack.hpp"

namespace Cengine {

class ctensor_add_inp_op : public Coperator,
                           public CumulativeOperator,
                           public InPlaceOperator,
                           public RbatchedOperator {
 public:
  Gdims dims;

  ctensor_add_inp_op(Cnode* R, Cnode* A, Cnode* B, const Gdims& _dims)
      : Coperator(R, A, B), dims(_dims) {}

  void exec() {
    assert(!owner->obj);
    owner->obj = inputs[0]->obj;
    CTENSORB(inputs[1]).add_inp_into(CSCALARB(owner), CTENSORB(inputs[2]));
  }

  void rbatched_exec(const vector<Cnode*>& nodes) {
    const int N = nodes.size();
    int dev = CTENSORB(nodes[0]->op->inputs[0]).device;
    assert(dev == 1);

    if (dev == 1) {
      CscalarBreducer R(N, CSCALARB(owner));
      CtensorBpack A(nodes, 1);
      CtensorBpack B(nodes, 2);
      A.add_inp_into(R, B);
    }
  }

 public:
  string str() const { return "ctensor_add_inp" + inp_str(); }

  static string classname() { return "ctensor_add_inp_op"; }

  static int _rbatcher_id;
  void set_rbatcher_id(const int i) { _rbatcher_id = i; }
  int rbatcher_id() const { return _rbatcher_id; }
  string rbatcher_name() const {
    return "ctensor_add_inp<" + rsignature().str() + ">";
  }
  ctensor_signature rsignature() const { return ctensor_signature(dims); }
  Rbatcher_base* spawn_rbatcher(BasicCnodeEngine* engine) const {
    return new MetaRbatcher<ctensor_add_inp_op, ctensor_signature, Rbatcher>(
        engine);
  }
};

}  // namespace Cengine

#endif
