#ifndef _ctensor_Mprod_signature
#define _ctensor_Mprod_signature

#include "Cengine_base.hpp"

namespace Cengine {

class ctensor_Mprod_signature {
 public:
  Gdims dims1;
  Gdims dims2;

  ctensor_Mprod_signature(const Gdims& _dims1, const Gdims& _dims2)
      : dims1(_dims1), dims2(_dims2) {}

  bool operator==(const ctensor_Mprod_signature& x) const {
    return (dims1 == x.dims1) && (dims2 == x.dims2);
  }

  string str() const { return "(" + dims1.str() + dims2.str() + ")"; }
};

}  // namespace Cengine

namespace std {

template <>
struct hash<Cengine::ctensor_Mprod_signature> {
 public:
  size_t operator()(const Cengine::ctensor_Mprod_signature& ix) const {
    size_t t = ((hash<Cengine::Gdims>()(ix.dims1) << 1) ^
                hash<Cengine::Gdims>()(ix.dims2) << 1);
    return t;
  }
};

}  // namespace std

#endif
