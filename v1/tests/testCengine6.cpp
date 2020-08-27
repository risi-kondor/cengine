#include "Cengine_base.cpp"

#include "CtensorObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;
typedef CtensorObject Ctensor;


int main(int argc, char** argv){

  Ctensor A({4,4},fill::identity);
  //cout<<"A="<<A<<endl;

  Ctensor B({4,4},fill::gaussian);
  //cout<<"B="<<B<<endl;

  vector<Ctensor*> v; 
  for(int i=0; i<5; i++){
    //v.push_back(new Ctensor(A));
    A+=B;
  }

  cout<<"A="<<endl<<A<<endl;

}
