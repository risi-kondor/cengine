#ifndef _Batcher
#define _Batcher

#include "Cengine_base.hpp"


namespace Cengine{


  class Batcher{
  public:

    //Cengine* engine;

    virtual ~Batcher(){}

    virtual void push(Coperator* op)=0;
    virtual void release(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;

    virtual void release()=0; 
    virtual int flush()=0; 
    virtual int npending() const=0; 

  };


}


#endif 
