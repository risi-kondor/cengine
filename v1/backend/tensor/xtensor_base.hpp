#ifndef _xtensor_base
#define _xtensor_base

#include "Coperator.hpp"

#include "Gdims.hpp"
#include "CscalarB.hpp"
#include "CtensorB.hpp"


namespace Cengine{


  class diamond_op: public Coperator{
  public:

    diamond_op(Cnode* x, Cnode* y):
      Coperator(x,y){}

    virtual void exec(){
      owner->obj=inputs[0]->obj;
      inputs[0]->is_view=true; 
      //owner->computed=true; 
    }

    string str() const{
      return "diamond"+inp_str();
	/*
      ostringstream oss;
      oss<<"diamond(";
      for(int i=0; i<inputs.size()-1; i++)
	oss<<inputs[i]->ident()<<",";
      if(inputs.size()>0) oss<<inputs[inputs.size()-1]->ident();
      oss<<")";
      return oss.str();
	*/
    }
    
  };




}
#endif

