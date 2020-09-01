#define DEBUG_ENGINE_FLAG 
//#define ENGINE_PRIORITY

#include "ArrayOf.hpp"
#include "Cengine_base.cpp"
#include "CtensorObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;
typedef CtensorObject Ctensor;


int main(int argc, char** argv){

  Cscalar c(2);
  cout<<"c="<<c<<endl;

  const int N=10;

  ArrayOf<Ctensor> A({N},Gdims({4,4}),fill::gaussian);
  ArrayOf<Ctensor> B({N},Gdims({4,4}),fill::gaussian);

  ArrayOf<Ctensor> C(Gdims({N}));

  for(int i=0; i<N; i++)
    C(i)=A(i)*B(i);

  cout<<C(3)<<endl; 

  Cengine_engine->batching=false;
  cout<<A(3)*B(3)<<endl; 

}

  /*
  Ctensor A({4,4},fill::identity);
  cout<<"A="<<A<<endl;

  Ctensor B({4,4},fill::gaussian);
  cout<<"B="<<B<<endl;

  B.add(A,c);

  cout<<"B="<<B<<endl;

  A*B;
  A*B;
  cout<<A*B<<endl; 

  cout<<"transp(B)="<<endl<<transp(B)<<endl; 
  */


