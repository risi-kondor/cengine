#define DEBUG_ENGINE_FLAG 
//#define DEBUG_FLUSH_FLAG 
#define ENGINE_PRIORITY

#include "ArrayOf.hpp"
#include "Cengine_base.cpp"
#include "CtensorObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;
typedef CtensorObject Ctensor;


int main(int argc, char** argv){

  CengineSession engine;
  cublasCreate(&Cengine_cublas);

  Cscalar c(2);
  cout<<"c="<<c<<endl;

  const int N=10;

  ArrayOf<Ctensor> A({N},Gdims({4,4}),fill::gaussian,0);
  ArrayOf<Ctensor> B({N},Gdims({4,4}),fill::gaussian,0);

  ArrayOf<Ctensor> C(Gdims({N}));

  for(int i=0; i<N; i++)
    C(i)=A(i)*B(i);

  cout<<A(0)<<endl; 
  cout<<B(0)<<endl; 
  cout<<C(0)<<endl; 

}
