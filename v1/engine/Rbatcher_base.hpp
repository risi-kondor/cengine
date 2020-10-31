#ifndef _Rbatcher_base
#define _Rbatcher_base

namespace Cengine{

  class Rbatcher_base{
  public:

    virtual ~Rbatcher_base(){}

    int id;
    virtual void push(Cnode* node)=0;
    //virtual void new_gang(Cnode* node){}
    virtual void release(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;

    virtual void release()=0; 
    virtual int flush()=0; 
    //virtual int npending() const=0; 

  };

}

#endif 
