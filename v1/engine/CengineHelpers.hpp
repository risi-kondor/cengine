#ifndef _CengineHelpers
#define _CengineHelpers

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
    }

    string str() const{
      return "diamond"+inp_str();
    }
    
  };




}
#endif

