#include "Cengine_base.cpp"
#include "CtensorObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;
typedef CtensorObject Ctensor;


int main(int argc, char** argv){

  Cscalar c(2);
  cout<<"c="<<c<<endl;

  Ctensor A({4,4},fill::identity);
  cout<<"A="<<A<<endl;

  Ctensor B({4,4},fill::gaussian);
  cout<<"B="<<B<<endl;

  B.add(A,c);

  cout<<"B="<<B<<endl;

  cout<<A*B<<endl; 

  cout<<"transp(B)="<<endl<<transp(B)<<endl; 

}
