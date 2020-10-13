#ifndef _MetaBatcher
#define _MetaBatcher

#include "Cnode.hpp"


namespace Cengine{


  template<typename OP, typename SUBINDEX, typename BATCHER>
  class MetaBatcher: public Batcher{
  public:

    BasicCnodeEngine* engine;

    unordered_map<SUBINDEX,BATCHER*> subengines;

    MetaBatcher(BasicCnodeEngine* _engine): 
      engine(_engine){
      DEBUG_ENGINE({CoutLock lk; cout<<"    New MetaBatcher for "<<OP::classname()<<endl;}) 
    }

    virtual ~MetaBatcher(){
      for(auto& p: subengines) delete p.second;
    }


  public:

    void push(Coperator* _op){
      
      BatchedOperator* op=dynamic_cast<BatchedOperator*>(_op);
      SUBINDEX ix=dynamic_cast<OP*>(op)->signature();

      auto it=subengines.find(ix);
      if(it!=subengines.end()){
	it->second->push(_op);
	return;
      }

      BATCHER* sub=new BATCHER(engine,op->batcher_name());
      subengines[ix]=sub;
      sub->push(_op);
    }


    int flush(){
      int nwaiting=0; 
      for(auto p:subengines)
	nwaiting+=p.second->flush();
      return nwaiting; 
    }

    int npending() const{
      int t=0;
      for(auto p:subengines)
	t+=p.second->npending();
      return t;
    }
    
    void release(Cnode* node){
    }

    void kill(Cnode* node){
    }

    void release(){}

  };

}


#endif
