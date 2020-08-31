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
      {CoutLock lk; cout<<"New MetaBatcher for "<<OP::classname()<<endl;}
    }

    virtual ~MetaBatcher(){
      for(auto& p: subengines) delete p.second;
    }


  public:

    void push(Coperator* op){
      
      SUBINDEX ix=dynamic_cast<OP*>(op)->signature();

      auto it=subengines.find(ix);
      if(it!=subengines.end()){
	it->second->push(op);
	return;
      }

      BATCHER* sub=new BATCHER(engine);
      subengines[ix]=sub;
      sub->push(op);
    }


    int flush(){
      int nwaiting=0; 
      for(auto p:subengines)
	nwaiting+=p.second->flush();
      return nwaiting; 
    }

    void release(Cnode* node){
    }

    void kill(Cnode* node){
    }

    void exec(){}

  };

}


#endif
