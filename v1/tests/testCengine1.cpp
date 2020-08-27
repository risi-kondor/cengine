#include "Cengine_base.cpp"

#include "CscalarInterface.hpp"
#include "CtensorInterface.hpp"

using namespace Cengine;


int main(int argc, char** argv){

  //  Cengine_engine=new Cengine();

  Gdims dims({4,4});
  Chandle* hdl0=engine::new_ctensor_gaussian(dims);
  Chandle* hdl1=engine::new_ctensor_gaussian(dims);


  cout<<endl<<"A="<<endl<<engine::ctensor_get(hdl0)<<endl<<endl;
  cout<<endl<<"B="<<endl<<engine::ctensor_get(hdl1)<<endl<<endl;

  Chandle* hdl2=engine::ctensor_add(hdl0,hdl1);
  cout<<endl<<"A+B="<<endl<<engine::ctensor_get(hdl2)<<endl<<endl;

  //delete Cengine_engine; 

}
