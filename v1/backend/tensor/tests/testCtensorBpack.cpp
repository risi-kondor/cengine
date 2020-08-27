#include "GEnet_base.cpp"

#include "CtensorFunctions.hpp"
#include "CtensorBpack.hpp"

using namespace GEnet;

typedef CscalarObj Cscalar;
typedef CtensorObj Ctensor;


int main(int argc, char** argv){
  //GEnet_engine=new Cengine();

  CtensorB A({4,4},fill::identity);
  cout<<"A="<<endl<<A<<endl<<endl;

  CtensorB B({4,4},fill::gaussian);
  cout<<"B="<<endl<<B<<endl<<endl;

  CtensorB C({4,4},fill::zero);
  cout<<"C="<<endl<<C<<endl<<endl;

  CtensorBpack Apack(A);
  CtensorBpack Bpack(B);
  CtensorBpack Cpack(C);

  for(int i=0; i<100; i++){
    Apack.push_back(A);
    Bpack.push_back(B);
    Cpack.push_back(C);
  }

  Cpack.add_Mprod<0>(Apack,Bpack);

  cout<<"C[22]="<<endl<<*Cpack.pack[22]<<endl<<endl;

  //delete GEnet_engine; 
}
