#ifndef _InterfaceBase
#define _InterfaceBase

#include "Cengine.hpp"

extern Cengine::Cengine* Cengine_engine;


namespace Cengine{
  
  //  inline void replace(Chandle*& target, Chandle* hdl){
  //delete target;
  //    target=hdl;
  //}

  inline Chandle* new_handle(Cnode* node){
    return Cengine_engine->new_handle(node);
  }
  
}


#endif
