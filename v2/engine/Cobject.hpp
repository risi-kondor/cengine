#ifndef _Cobject
#define _Cobject

#include "Cengine_base.hpp"

namespace Cengine {

class Cobject {
 public:
  virtual ~Cobject() {}

 public:
  virtual int get_device() const { return 0; }

  virtual string classname() const { return ""; }

  virtual string describe() const { return ""; }
};

template <typename TYPE>
inline TYPE& downcast(Cobject* x, const char* s) {
  if (!x) {
    CoutLock lk;
    cerr << "\e[1mCengine error\e[0m (" << s << "): object does not exist"
         << endl;
    exit(-1);
  }
  if (!dynamic_cast<TYPE*>(x)) {
    CoutLock lk;
    cerr << "\e[1mCengine error\e[0m (" << s << "): Cobject is of type "
         << x->classname() << " instead of TYPE." << endl;
    exit(-1);
  }
  return static_cast<TYPE&>(*x);
}

template <typename TYPE>
inline TYPE& downcast(Cobject& x, const char* s) {
  if (!dynamic_cast<TYPE&>(x)) {
    CoutLock lk;
    cerr << "\e[1mCengine error\e[0m (" << s << "): Cobject is of type "
         << x.classname() << " instead of TYPE." << endl;
    exit(-1);
  }
  return static_cast<TYPE&>(x);
}

}  // namespace Cengine

#endif
