#ifndef _Bifstream
#define _Bifstream

#include <fstream>
#include "Cengine_base.hpp"


namespace Cengine{

  class Bifstream: public ifstream{
  public:
    
    Bifstream(const string filename){
      open(filename,ios::binary);
    }
    
    ~Bifstream(){close();}
   
    
  public:

    template<typename TYPE>
    TYPE get(){
      TYPE t;
      ifstream::read(reinterpret_cast<char*>(&t),sizeof(TYPE)); 
      return t;
    }
   
    Bifstream& read(int& x){
      ifstream::read(reinterpret_cast<char*>(&x),sizeof(int)); 
      return *this;
    }

    Bifstream& read(float& x){
      ifstream::read(reinterpret_cast<char*>(&x),sizeof(float)); 
      return *this;
    }

    Bifstream& read(double& x){
      ifstream::read(reinterpret_cast<char*>(&x),sizeof(double)); 
      return *this;
    }

    template<class TYPE>
    Bifstream& read_array(TYPE*& x){
      int n; ifstream::read(reinterpret_cast<char*>(&n),sizeof(int)); 
      ifstream::read(reinterpret_cast<char*>(x),n*sizeof(TYPE)); 
      return *this;
    }

    template<class TYPE>
    vector<TYPE*> read_vector_ofp(){
      int n; ifstream::read(reinterpret_cast<char*>(&n),sizeof(int));
      vector<TYPE*> R(n);
      for(int i=0; i<n; i++)
	R[i]=new TYPE(*this);
      return R;
    }

    template<class TYPE>
    vector<TYPE> read_vector_of(){
      int n; ifstream::read(reinterpret_cast<char*>(&n),sizeof(int));
      vector<TYPE> R(n);
      for(int i=0; i<n; i++)
	R[i]=TYPE(*this);
      return R;
    }


  };

}


#endif


