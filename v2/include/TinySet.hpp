#ifndef _TinySet
#define _TinySet

#include "Cengine_base.hpp"

namespace Cengine{

  template<typename TYPE>
  class TinySet{
  public:

    vector<TYPE> v;

  public:
    
    int size() const{
      int t=0;
      for(auto p:v) if(p!=nullptr) t++;
      return t;
    }

    bool find(const TYPE x) const{
      for(auto p:v) if(p==x) return true;
      return false;
    }

    bool insert(TYPE x){
      if(find(x)) return false;
      v.push_back(x);
      return true;
    }

    bool erase(const TYPE x){
      for(auto& p:v)
	if(p==x){
	  p=nullptr;
	  return true;
	}
      return false;
    }

    void clear(){
      v.clear();
    }


  public:

    void map(std::function<void(TYPE x)> lambda) const{
      for(auto p:v) 
	if(p) lambda(p);
    }

  };

}

#endif
