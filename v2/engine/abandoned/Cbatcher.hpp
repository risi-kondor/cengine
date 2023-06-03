#ifndef _Cbatcher
#define _Cbatcher

// #include ""

namespace Cengine {

class Cbatcher {
 public:
  set<Cnode*> waiting;
  vector<Cnode*> ready;

 public:
  void push(Coperator* op) {}
};

}  // namespace Cengine

#endif
