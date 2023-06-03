#ifndef _ctensor_signature
#define _ctensor_signature

#include "Cengine_base.hpp"

namespace Cengine {

class ctensor_signature {
 public:
  Gdims dims1;

  ctensor_signature(const Gdims& _dims1) : dims1(_dims1) {}

  bool operator==(const ctensor_signature& x) const {
    return (dims1 == x.dims1);
  }

  string str() const { return "(" + dims1.str() + ")"; }
};

}  // namespace Cengine

namespace std {

template <>
struct hash<::Cengine::ctensor_signature> {
 public:
  size_t operator()(const ::Cengine::ctensor_signature& ix) const {
    size_t t = (hash<::Cengine::Gdims>()(ix.dims1) << 1);
    return t;
  }
};

}  // namespace std

#endif
