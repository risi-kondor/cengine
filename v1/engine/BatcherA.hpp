#ifndef _BatcherA
#define _BatcherA

#include "Batcher.hpp"
#include "Cnode.hpp"


namespace Cengine{


  template<typename OP> 
  class exec_batcher_op: public Coperator, public BatcherExecutor{
  public:

    vector<Cnode*> nodes;
    
    exec_batcher_op(const vector<Cnode*>& _nodes): 
      nodes(_nodes){};

    void exec(){
      if(nodes.size()==0){CoutLock lk; cout<<"\e[1mEmpty batcher\e[0m"<<endl; return;}
      assert(nodes.size()>0);
      dynamic_cast<OP*>(nodes[0]->op)->batched_exec(nodes); 
    }
    
    string str() const{
      if(nodes.size()==0)  return "exec_batcher_op";
      return "exec_batcher_op<"+nodes[0]->op->str()+">";
    }

  };


  template<typename OP>
  class BatcherA: public Batcher{
  public:

    BasicCnodeEngine* engine;
    string name; 

    set<Cnode*> waiting;
    vector<Cnode*> ready;
    bool working=false; 
    mutex mx; 

    BatcherA(BasicCnodeEngine* _engine):
      engine(_engine){
    }

    BatcherA(BasicCnodeEngine* _engine, const string _name):
      engine(_engine), name(_name){
    }

    ~BatcherA(){}


  public:


    void push(Coperator* op){ // protected by done_mx 
      Cnode* node=op->owner;
      DEBUG_ENGINE({CoutLock lk; cout<<"    Batching "<<node->ident()<<" ["<<node->op->str()<<"] "<<endl;});
      if(node->nblockers==0) ready.push_back(node);
      else waiting.insert(node);
      node->batcher=this; 
      check_status();
    }


    void release(Cnode* node){ // protected by done_mx 
      DEBUG_ENGINE({CoutLock lk; cout<<"    Releasing "<<node->ident()<<" in batcher"<<endl;});
      ready.push_back(node);
      waiting.erase(node); 
      check_status(); 
    }


    void kill(Cnode* node){
      DEBUG_ENGINE({CoutLock lk; cout<<"    Killing "<<node->ident()<<" in batcher"<<endl;});
      //CoutLock lk;
      //cout<<"\e[1mKill "<<node->ident()<<" \e[0m"<<endl; 
    }


    void check_status(){ // protected by done_mx 
      if(ready.size()>=10){
	release();
      }
    }

    
    void release(){ // protected by done_mx 
      DEBUG_ENGINE({CoutLock lk; cout<<"  Releasing batcher "<<name<<" ["<<ready.size()<<"]"<<endl;});
      working=true;
      Cnode* node=engine->new_node(new exec_batcher_op<OP>(ready));
      engine->release_batcher(node);
      ready.clear();
    }

    
    int flush(){ // protected_by done_mx 
      DEBUG_ENGINE({CoutLock lk; cout<<"    Flushing batcher. "<<waiting.size()<<" "<<ready.size()<<endl;});
      if(ready.size()>0) release();
      return waiting.size(); 
    }

    int npending() const{
      return waiting.size()+ready.size(); 
    }

  };


}


#endif
