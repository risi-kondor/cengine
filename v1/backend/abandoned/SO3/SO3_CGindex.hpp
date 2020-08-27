#ifndef _CGindex
#define _CGindex

#include "GEnet_base.hpp"

namespace GEnet{

  class CGindex{
  public:

    int l1,l2,l;

    CGindex(const int _l1, const int _l2, const int _l): l1(_l1), l2(_l2), l(_l){
      assert(l1>=0); assert(l2>=0); assert(l>=0);
      assert(l<=l1+l2); assert(l>=abs(l1-l2));
    }

    bool operator==(const CGindex& x) const{
      return (l1==x.l1)&&(l2==x.l2)&&(l==x.l);}

    string str() const{
      return "("+to_string(l1)+","+to_string(l2)+","+to_string(l)+")";}

  };

} 


namespace std{
template<>
struct hash<GEnet::CGindex>{
public:
  size_t operator()(const GEnet::CGindex& ix) const{
    return ((hash<int>()(ix.l1)<<1)^hash<int>()(ix.l2)<<1)^hash<int>()(ix.l);
  }
};
}

#endif 
