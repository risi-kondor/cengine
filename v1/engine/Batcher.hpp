#ifndef _Batcher
#define _Batcher

#include "Cengine_base.hpp"


namespace Cengine{


  class Batcher{
  public:

    virtual ~Batcher(){}

    virtual void push(Coperator* op)=0;
    virtual void release(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;

    virtual void exec()=0; 
    virtual int flush()=0; 

  };


}


#endif 
