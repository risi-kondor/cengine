#ifndef _Rbatcher
#define _Rbatcher

#include "Cnode.hpp"


namespace Cengine{


  class exec_rbatcher_op: public Coperator, public BatcherExecutor{
  public:

    vector<Cnode*> nodes;
    
    exec_rbatcher_op(const vector<Cnode*>& _nodes): 
      nodes(std::move(_nodes)){};

    void exec(){
      assert(nodes.size()>0);
      dynamic_cast<RbatchedOperator*>(nodes[0]->op)->rbatched_exec(nodes); 
    }
    
    string str() const{
      if(nodes.size()==0)  return "exec_batcher_op<>";
      return "exec_rbatcher_op<"+dynamic_cast<RbatchedOperator*>(nodes[0]->op)->rbatcher_name()+">";
    }

  };


  class Rbatcher: public Rbatcher_base{
  public:

    BasicCnodeEngine* engine;
    string name; 

    set<Cnode*> waiting;
    vector<Cnode*> ready;

    Rbatcher(BasicCnodeEngine* _engine):
      engine(_engine){
    }

    Rbatcher(BasicCnodeEngine* _engine, const string _name):
      engine(_engine), name(_name){
    }

    ~Rbatcher(){}

  public:


    void push(Cnode* node){ // protected by done_mx 
      //Cnode* node=op->owner;
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Batching "<<node->ident()<<" ["<<node->op->str()<<"] "<<endl;});
      DEBUG_ENGINE2("    Rbatching "<<node->ident()<<" ["<<node->op->str()<<"] in "<<name);
      if(node->nblockers==0) ready.push_back(node);
      else waiting.insert(node);
      node->rbatcher=this; 
      check_status();
    }


    void release(Cnode* node){ // protected by done_mx 
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Releasing "<<node->ident()<<" in batcher"<<endl;});
      ready.push_back(node);
      waiting.erase(node); 
      check_status(); 
    }


    void kill(Cnode* node){
      DEBUG_ENGINE({CoutLock lk; cout<<"    Killing "<<node->ident()<<" in batcher"<<endl;});
      //{CoutLock lk; cout<<"\e[1mKill "<<node->ident()<<" \e[0m"<<endl;} 
    }


    void check_status(){ // protected by done_mx 
      if(ready.size()>=64 && waiting.size()==0){
	release();
      }
    }

    
    void release(){ // protected by done_mx 
      DEBUG_ENGINE({CoutLock lk; cout<<"  Releasing batcher "<<name<<" ["<<ready.size()<<"]"<<endl;});
      //working=true;
      Cnode* node=engine->new_node(new exec_rbatcher_op(ready));
      engine->release_batcher(node);
      ready.clear();
    }

    
    int flush(){ // protected_by done_mx 
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Flushing batcher "<<name<<". "<<waiting.size()<<" "<<ready.size()<<endl;});
      if(ready.size()>0) release();
      return waiting.size(); 
    }

    //int npending() const{
    //return waiting.size()+ready.size(); 
    //}

  };


}


#endif
