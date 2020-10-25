//#define DEBUG_ENGINE_FLAG 
//#define DEBUG_FLUSH_FLAG 
#define ENGINE_PRIORITY

#include "ArrayOf.hpp"
#include "Cengine_base.cpp"
#include "CtensorObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;
typedef CtensorObject Ctensor;


int main(int argc, char** argv){

  const int N=64;
  const int m=100;
  const int dev=1;

  CengineSession engine;
  cublasCreate(&Cengine_cublas);

  Cscalar c(2);
  cout<<"c="<<c<<endl;

  ArrayOf<Ctensor> A({N},Gdims({m,m}),fill::gaussian,dev);
  ArrayOf<Ctensor> B({N},Gdims({m,m}),fill::gaussian,dev);

  ArrayOf<Ctensor> C(Gdims({N}));

  for(int i=0; i<N; i++)
    C(i)=A(i)*B(i);

  Cengine_engine->flush();


  cout<<norm2(C(33)-(A(33).to(0)*B(33).to(0)))<<endl;

}
