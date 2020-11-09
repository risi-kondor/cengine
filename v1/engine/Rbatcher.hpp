#ifndef _Rbatcher
#define _Rbatcher

#include "Cnode.hpp"
#include "TinySet.hpp"
#include "MetaRbatcher.hpp"

namespace Cengine{


  class exec_rbatcher_op: public Coperator, public BatcherExecutor{
  public:

    BasicCnodeEngine* engine;
    vector<Cnode*> nodes;
    
    exec_rbatcher_op(BasicCnodeEngine* _engine, const vector<Cnode*>&& _nodes): 
      engine(_engine), nodes(std::move(_nodes)){};

    void exec(){
      const int N=nodes.size();
      if(N==0) return; 

#ifndef CENGINE_DRY_RUN
      int dev=nodes[0]->op->inputs[0]->obj->get_device();
      if(dev==0){
	//COUT("CPU rbatched "<<N);
	for(int i=0; i<N; i++)
	  nodes[i]->op->exec();
      }else{
	//COUT("GPU rbatched "<<N);
	if(dynamic_cast<CumulativeOperator*>(nodes[0]->op)){
	  Cobject* target=nodes[0]->op->inputs[0]->obj;
	  for(int i=0; i<N; i++)
	    nodes[i]->obj=target;
	}
	dynamic_cast<RbatchedOperator*>(nodes[0]->op)->rbatched_exec(nodes); 
      }
#endif

      for(int i=0; i<N; i++){
	engine->done(nodes[i]);
      }
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

#ifdef WITH_TINYSET
    TinySet<Cnode*> waiting;
#else
    set<Cnode*> waiting;
#endif

    vector<Cnode*> ready;

    Rbatcher(BasicCnodeEngine* _engine):
      engine(_engine){
    }

    Rbatcher(BasicCnodeEngine* _engine, const string _name):
      engine(_engine), name(_name){
      DEBUG_ENGINE2("    \e[1mNew Rbatcher for "<<name<<"\e[0m");
    }

    ~Rbatcher(){}

  public:


    void push(Cnode* node){ // protected by done_mx 
      DEBUG_ENGINE2("    Queuing "<<node->ident()<<" ["<<node->op->str()<<"] in Rbatcher "<<id);
      //if(node->nblockers==0){ready.push_back(node); COUT("ready.")}
      //else waiting.insert(node);
      waiting.insert(node);
      node->rbatcher=this; 
      //check_status();
    }


    void release(Cnode* node){ // protected by done_mx 
      DEBUG_ENGINE2("    Releasing "<<node->ident()<<" in Rbatcher");
      ready.push_back(node);
      waiting.erase(node); 
      WAITING_OPT(dynamic_cast<Cengine*>(engine)->waiting.erase(node););
      node->released=true;
      check_status(); 
    }


    void kill(Cnode* node){
      DEBUG_ENGINE2("    Killing "<<node->ident()<<" in Rbatcher");
    }


    void check_status(){ // protected by done_mx 
      if(ready.size()>=64 && waiting.size()==0){
	release();
      }
    }

    
    void release(){ // protected by done_mx 
      DEBUG_ENGINE2("    Releasing Rbatcher for "<<name<<" ["<<ready.size()<<"]");
      Cnode* node=engine->new_node(new exec_rbatcher_op(engine,std::move(ready)));
      engine->release_batcher(node);
      ready.clear();
    }

    
    int flush(){ // protected_by done_mx 
      DEBUG_ENGINE2("    Flushing Rbatcher "<<name<<". "<<waiting.size()<<" "<<ready.size());
      if(ready.size()>0) release();
      return waiting.size(); 
    }

    int npending() const{
      return waiting.size()+ready.size(); 
    }

    int nwaiting() const{
      return waiting.size(); 
    }

  };


}


#endif
