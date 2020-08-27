#ifndef _Gindex
#define _Gindex

#include "Cengine_base.hpp"
#include "Gdims.hpp"


namespace Cengine{
    
  class Gindex: public vector<int>{
  public:

    Gindex(){}

    Gindex(const fill_zero& dummy){
    }

    Gindex(const initializer_list<int>& list): vector<int>(list){}

    Gindex(const int i0):
      Gindex({i0}){}

    Gindex(const int i0, const int i1): vector<int>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    Gindex(int a, const Gdims& dims): 
      vector<int>(dims.size()){
      for(int i=size()-1; i>=0; i--){
	(*this)[i]=a%dims[i];
	a=a/dims[i];
      }
    }

  public:
    
    string str() const{
      string str="("; 
      int k=size();
      for(int i=0; i<k; i++){
	str+=std::to_string((*this)[i]);
	if(i<k-1) str+=",";
      }
      return str+")";
    }

    string str_bare() const{
      string str; 
      int k=size();
      for(int i=0; i<k; i++){
	str+=std::to_string((*this)[i]);
	if(i<k-1) str+=",";
      }
      return str;
    }



  friend ostream& operator<<(ostream& stream, const Gindex& v){
    stream<<v.str(); return stream;}

  };



}


namespace std{
  template<>
  struct hash<Cengine::Gindex>{
  public:
    size_t operator()(const Cengine::Gindex& ix) const{
      size_t t=hash<int>()(ix[0]);
      for(int i=1; i<ix.size(); i++) t=(t<<1)^hash<int>()(ix[i]);
      return t;
    }
  };
}




#endif

