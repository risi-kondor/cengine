#ifndef _BatcherA
#define _BatcherA

#include "Batcher.hpp"
#include "Cnode.hpp"

//#include "SO3partObj.hpp"

namespace Cengine{


  template<typename OP> 
  class exec_batcher_op: public Coperator{
  public:

    vector<Cnode*> nodes;
    
    exec_batcher_op(vector<Cnode*> _nodes): 
      nodes(_nodes){};

    void exec(){
      assert(nodes.size()>0);
      dynamic_cast<OP*>(nodes[0]->op)->batched_exec(nodes); 
    }
    
    string str() const{
      return "exec_batcher_op";
    }

  };



  template<typename OP>
  class BatcherA: public Batcher{
  public:

    BasicCnodeEngine* engine;

    set<Cnode*> waiting;
    vector<Cnode*> ready;
    bool working=false; 

    BatcherA(BasicCnodeEngine* _engine):
      engine(_engine){
    }

    ~BatcherA(){}


  public:


    void push(Coperator* op){
      // make sure it is not executing 
      Cnode* node=op->owner;
      DEBUG_ENGINE({CoutLock lk; cout<<"    Batching "<<node->ident()<<" ["<<node->op->str()<<"] "<<endl;});

      if(node->nblockers==0) ready.push_back(node);
      else waiting.insert(node);
      node->batcher=this; 
      check_status();
    }


    void release(Cnode* node){
      // make sure it is not executing 
      DEBUG_ENGINE({CoutLock lk; cout<<"    Releasing "<<node->ident()<<" in batcher"<<endl;});
      ready.push_back(node);
      waiting.erase(node); 
      check_status(); 
    }


    void kill(Cnode* node){
      // make sure it is not executing 
      DEBUG_ENGINE({CoutLock lk; cout<<"    Killing "<<node->ident()<<" in batcher"<<endl;});
    }


    void check_status(){
      if(ready.size()>=3){
	exec();
      }
    }

    
    void exec(){
      working=true;
      Cnode* node=engine->new_node(new exec_batcher_op<OP>(ready));
      engine->release(node);
      ready.clear();
    }

    
    int flush(){
      DEBUG_ENGINE({CoutLock lk; cout<<"    \e[1mFlushing batcher.\e[0m"<<endl;});
      if(ready.size()>0) exec();
      return waiting.size(); 
    }


  };


}


#endif
