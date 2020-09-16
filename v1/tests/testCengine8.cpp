#include "Cengine_base.cpp"
#include "CscalarObject.hpp"
#include "CtensorObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;
typedef CtensorObject Ctensor;


int main(int argc, char** argv){

  Cscalar x(4,fill::gaussian);
  Cscalar y(4,fill::gaussian);

  cout<<endl<<"x= "<<x<<endl<<endl; 
  cout<<endl<<"y= "<<y<<endl<<endl; 

  x+=y; 
  cout<<endl<<"x+y="<<x<<endl<<endl; 

  cout<<endl<<"y*y="<<y*y<<endl<<endl; 

  Ctensor A({4,4},3,fill::identity);
  Ctensor B({4,4},3,fill::sequential);

  Ctensor N=B.column_norms(); 
  cout<<N<<endl<<endl; 
  B.divide_columns(N);
  cout<<B<<endl;
  cout<<B.column_norms()<<endl; 
  cout<<endl;

  Ctensor C=A+B;
  cout<<C<<endl; 

  Ctensor W({4,4},fill::gaussian);
  cout<<W.mix(x)<<endl; 

  

}
