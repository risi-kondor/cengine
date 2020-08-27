#include "Cengine_base.cpp"
#include "CscalarObject.hpp"

using namespace Cengine;

typedef CscalarObject Cscalar;


int main(int argc, char** argv){

  Cscalar x(2.0);
  Cscalar y(3.0);

  cout<<endl<<"x= "<<x<<endl<<endl; 
  cout<<endl<<"y= "<<y<<endl<<endl; 

  x+=y; 
  cout<<endl<<"x+y="<<x<<endl<<endl; 

  cout<<endl<<"y*y="<<y*y<<endl<<endl; 

}
