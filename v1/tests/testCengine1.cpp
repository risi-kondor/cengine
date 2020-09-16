#include "Cengine_base.cpp"

#include "InterfaceBase.hpp"

using namespace Cengine;


int main(int argc, char** argv){

  Gdims dims({4,4});
  Chandle* hdl0=Cengine_engine->push<new_ctensor_gaussian_op>(dims);
  Chandle* hdl1=Cengine_engine->push<new_ctensor_gaussian_op>(dims);


  cout<<endl<<"A="<<endl<<ctensor_get(hdl0)<<endl<<endl;
  cout<<endl<<"B="<<endl<<ctensor_get(hdl1)<<endl<<endl;

  Chandle* hdl2=Cengine_engine->push<ctensor_add_op>(hdl0,hdl1);
  cout<<endl<<"A+B="<<endl<<ctensor_get(hdl2)<<endl<<endl;

}


//Chandle* hdl0=engine::new_ctensor_gaussian(dims);
//Chandle* hdl1=engine::new_ctensor_gaussian(dims);
