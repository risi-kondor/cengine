#ifndef _MetaRbatcher
#define _MetaRbatcher

#include "Cnode.hpp"
#include "Rbatcher_base.hpp"

namespace Cengine{


  template<typename OP, typename SUBINDEX, typename RBATCHER>
  class MetaRbatcher: public Rbatcher_base{
  public:

    BasicCnodeEngine* engine;

    int batchercount=0;
    set<Rbatcher_base*> batchers;

    MetaRbatcher(BasicCnodeEngine* _engine): 
      engine(_engine){
      DEBUG_ENGINE2("    \e[1mNew MetaRbatcher for "<<OP::classname()<<"\e[0m") 
    }

    virtual ~MetaRbatcher(){
      for(auto& p: batchers) delete p; //.second;
    }


  public:

    void push(Cnode* node){
      OP* op=static_cast<OP*>(node->op);
      Rbatcher_base* sub=new RBATCHER(engine,op->rbatcher_name());
      sub->id=batchercount++;
      batchers.insert(sub);
      sub->push(node);
    }
    
    int flush(){
      int nwaiting=0; 
      for(auto p:batchers)
	nwaiting+=p->flush();
      return nwaiting; 
    }

    void release(Cnode* node){
    }

    void kill(Cnode* node){
    }

    void release(){}

  };

}


#endif

