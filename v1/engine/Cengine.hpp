#ifndef _Cengine
#define _Cengine

#include <deque>
#include <chrono>

#include "priority_guard.hpp"

#include "Cobject.hpp"
#include "Coperator.hpp"
#include "Cnode.hpp"
#include "Chandle.hpp"
#include "Cworker.hpp"
#include "MetaBatcher.hpp"

#include "xtensor_base.hpp"


namespace Cengine{


  class Cengine: public BasicCnodeEngine{
  public:

    int nbatchers;

    set<Cnode*> nodes;
    deque<Cnode*> ready;
    set<Cnode*> waiting;
    vector<Cnode*> tokill;
    
    //set<Chandle*> handles;

    vector<Cworker*> workers;
    //vector<GenericMetaBatcher*> batchers;
    vector<Batcher*> batchers;
    bool batching=true; 
    bool shutdown=false; 

    int nnodes=0;
    int nhandles=0;

    mutex ready_mx;
#ifdef ENGINE_PRIORITY
    priority_mutex<3> done_pmx;
#else
    mutex done_mx;
#endif

    condition_variable get_task_cv;
    mutex get_task_mx;
    //mutex get_task_mx2;


    Cengine(){
      for(int i=0; i<3; i++)
	workers.push_back(new Cworker(this,i));
      //batchers.push_back(new MetaBatcher<ctensor_add_Mprod_op,ctensor_Mprod_signature,CtensorB_add_Mprod_batcher>());
    }


    ~Cengine(){
      DEBUG_ENGINE({CoutLock lk; cout<<"    Shutting down engine"<<endl;});
      shutdown=true; 
      //for(auto p:handles) delete p;
      get_task_cv.notify_all();
      for(auto p:workers) delete p;
      for(auto p:batchers) delete p;
      for(auto p:nodes) delete p;
    }


  public: // ---- Nodes --------------------------------------------------------------------------------------


    Cnode* new_node(Coperator* op){
      Cnode* node=new Cnode(op);
      op->owner=node;
      node->id=nnodes++;
      return node;
    }


  public: // ---- push templates 


    template<typename OP>
    Chandle* push(Chandle* h0){
      return new_handle(enqueue(new OP(nodeof(h0))));
    }

    template<typename OP>
    Chandle* push(Chandle* h0, Chandle* h1){
      return new_handle(enqueue(new OP(nodeof(h0),nodeof(h1))));
    }

    template<typename OP>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2){
      return new_handle(enqueue(new OP(nodeof(h0),nodeof(h1),nodeof(h2))));
    }


    template<typename OP, typename ARG0>
    Chandle* push(const ARG0 arg0){
      return new_handle(enqueue(new OP(arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, const ARG0 arg0){
      return new_handle(enqueue(new OP(nodeof(h0),arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0){
      return new_handle(enqueue(new OP(nodeof(h0),nodeof(h1),arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0){
      return new_handle(enqueue(new OP(nodeof(h0),nodeof(h1),nodeof(h2),arg0)));
    }


    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue(new OP(arg0,arg1)));
    }

    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(Chandle* h0, const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue(new OP(nodeof(h0), arg0,arg1)));
    }

    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue(new OP(nodeof(h0), nodeof(h1), arg0, arg1)));
    }

    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue(new OP(nodeof(h0), nodeof(h1), nodeof(h2), arg0, arg1)));
    }


    template<typename OP, typename ARG0, typename ARG1, typename ARG2>
    Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2){
      return new_handle(enqueue(new OP(arg0,arg1,arg2)));
    }



    Chandle* operator()(Coperator* op){
      return new_handle(enqueue(op));
    }


    Cnode* enqueue(Coperator* op){ // Protected by done_mx
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx,0);
#else
      lock_guard<mutex> lock(done_mx); 
#endif
      return enqueue_sub(op);
    }

    
    Cnode* enqueue_sub(Coperator* op){
      Cnode* node=new_node(op);
      node->engine=this;
      Cnode* rnode=node;

      // An in-place operator is dependent on on all dependents of its self-argument
      if(dynamic_cast<InPlaceOperator*>(op)){
	for(auto p: op->inputs[0]->dependents){
	  if((p->dependents.insert(node)).second){
	    if(!p->computed) node->nblockers++;
	  }
	}
      }

      // Delegate to batched operator, if it exists 
      if(dynamic_cast<BatchedOperator*>(op) && batching){
	BatchedOperator* bop=dynamic_cast<BatchedOperator*>(op);
	if(bop->batcher_id()==0){
	  bop->set_batcher_id(++nbatchers);
	  batchers.push_back(bop->spawn_batcher());
	}
	if(dynamic_cast<InPlaceOperator*>(op)){
	  op->inputs[0]->is_view=true; 
	}
	for(auto p: op->inputs){
	  if((p->dependents.insert(node)).second){
	    if(!p->computed) node->nblockers++;
	  }
	}
	batchers[bop->batcher_id()-1]->push(op);
	nodes.insert(node);
	return rnode;
      }

      // Make diamond to reflect commutativity of cumulative operators 
      Cnode* sibling=nullptr; 
      if(dynamic_cast<CumulativeOperator*>(op)){
	Cnode* father=op->inputs[0];
	//lock_guard<mutex> lock(done_mx);
	if(!father->computed && !father->working){
	  assert(father->op);
	  if(dynamic_cast<CumulativeOperator*>(father->op)){
	    DEBUG_ENGINE({CoutLock lk; cout<<"    Creating diamond"<<endl;});
	    Cnode* grandfather=father->father();
	    for(auto& p:op->inputs)
	      if(p==father) p=grandfather;
	    sibling=father;
	  }
	  if(dynamic_cast<diamond_op*>(father->op) && !father->released){ 
	    DEBUG_ENGINE({CoutLock lk; cout<<"    Extending diamond"<<endl;});
	    Cnode* greatgrandfather=father->father()->father();
	    for(auto& p:op->inputs)
	      if(p==father) p=greatgrandfather;
	    node->dependents.insert(father);
	    node->is_view=true;
	    father->op->inputs.push_back(node);
	    father->nblockers++;
	    rnode=father;
	  }
	}
      }

      if(dynamic_cast<InPlaceOperator*>(op)){ // Fix this!!!!
	//node->obj=op->inputs[0]->obj;
	op->inputs[0]->is_view=true; 
      }

      for(auto p: op->inputs){
	if((p->dependents.insert(node)).second){
	  //cout<<p->ident()<<":"<<p->computed<<endl; 
	  if(!p->computed) node->nblockers++;
	}
      }

      DEBUG_ENGINE({CoutLock lk; cout<<"    Enqueuing "<<node->ident()<<" ["<<node->op->str()<<"] "<<endl;});
      nodes.insert(node);

      // Complete diamond 
      if(sibling){
	Cnode* nnode=enqueue_sub(new diamond_op(sibling,node));
	nnode->obj=node->obj;
	node->is_view=true;
	sibling->is_view=true;
	rnode=nnode; 
      }
      
      if(node->nblockers==0){
	DEBUG_ENGINE({CoutLock lk; cout<<"    Early "<<node->ident()<<endl;});
	release(node);
      }
      else waiting.insert(node);

      return rnode;
    }


    void release(Cnode* node){ // visited by workers but protected by done_mx
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Releasing "<<node->ident()<<endl;});
      if(waiting.find(node)!=waiting.end()) waiting.erase(node);
      node->released=true; 
      {
	lock_guard<mutex> lock(ready_mx);
	auto it=find(ready.begin(), ready.end(),node);
	if(it!=ready.end()) ready.erase(it);
	ready.push_back(node);
      }
      //{CoutLock lk; for(auto p:ready) cout<<p->ident()<<" "; cout<<endl;}
      get_task_cv.notify_one();
    }


    void done(Cnode* node){ // visited by workers
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx,1); 
#else
      lock_guard<mutex> lock(done_mx);
#endif
      DEBUG_ENGINE({CoutLock lk; cout<<"    Done "<<node->ident()<<endl;});
      Coperator* op=node->op; 
      if(op){
	for(int i=0; i<op->inputs.size(); i++){
	  Cnode* p=op->inputs[i];
	  for(int j=0; j<i; j++) 
	    if(op->inputs[j]==p){p=nullptr;}
	  if(p!=nullptr) p->remove_dependent(node); // might kill *p
	}
      }
      for(auto p: node->dependents){
	//cout<<p->ident()<<endl;
	p->remove_blocker(node); // might release *p
      }
      node->computed=true;
      node->working=false; 
      //if(node->dependents.size()==0 && node->nhandles==0){ // may lead to orphan nodes 
      //{CoutLock lk; cout<<"Autokill "<<node->ident()<<endl;} 
      //kill(node);
      //}
    }

    /*
    void protected_kill(Cnode* node){
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx,2); 
#else
      lock_guard<mutex> lock(done_mx);
#endif
      kill(node);
    }
    */

    void kill(Cnode* node){
      // return; 
      DEBUG_ENGINE({CoutLock lk; cout<<"    Killing "<<node->ident()<<endl;}); 
      //if(nodes.find(node)==nodes.end()){CoutLock lk; cout<<"Cannot find node "<<node->ident()<<"!!!"<<endl;}
      if(node->dependents.size()>0) {CoutLock lk; cout<<"Caught dependent"<<endl; exit(-1); return;}
      if(node->nhandles>0) {CoutLock lk; cout<<"Caught handle"<<endl; exit(-1); return;}
      if(node->working){
	{CoutLock lk; cout<<"Caught working N"<<node->id<<endl;} 
	//exit(-1);
	tokill.push_back(node);
	return;
      }
      if(nodes.find(node)==nodes.end()){
	{CoutLock lk; cout<<"Cannot find node "<<endl; cout<<"N"<<node->id<<endl;}
	exit(-1);
      }
      nodes.erase(node);
      {
	lock_guard<mutex> lock(ready_mx);
	auto it=find(ready.begin(), ready.end(),node);
	if(it!=ready.end()) ready.erase(it);
      }
      delete node; 
     }


    void flush(Cnode* node){
      DEBUG_ENGINE({CoutLock lk; cout<<"flush"<<endl;})
      while(!node->computed){this_thread::sleep_for(chrono::milliseconds(13));} // TODO 
      return; 
     }


    void flush(){
      DEBUG_ENGINE({CoutLock lk; cout<<"flush"<<endl;})
      //for(auto p:batchers) p->flush(); 
      while(true){
	{
#ifdef ENGINE_PRIORITY
	  priority_guard<3> lock(done_pmx,2); 
#else
	  lock_guard<mutex> lock(done_mx);
#endif
	  lock_guard<mutex> lock2(ready_mx);
	  bool all_done=true;
	  for(auto p:batchers) 
	    if(p->flush()>0) all_done=false; 
	  if(ready.size()>0) all_done=false;
	  if(all_done) break;
	}
	this_thread::sleep_for(chrono::milliseconds(13));
      }
      {CoutLock lk; cout<<"flushed"<<endl;}
      return; 
    }


  public: // ---- Handles ------------------------------------------------------------------------------------

 
    Chandle* new_handle(Cnode* node){
     Chandle* hdl=new Chandle(node);
     hdl->id=nhandles++;
     //handles.insert(hdl);
     //DEBUG_ENGINE({CoutLock lk; cout<<"    New handle "<<hdl->ident()<<" ["<<hdl->node->ident()<<"]"<<endl;}); 
     return hdl;
    }


    void dec_handle(Cnode* node){
#ifdef ENGINE_PRIORITY
	priority_guard<3> lock(done_pmx,1); 
#else
	lock_guard<mutex> lock(done_mx);
#endif
	node->nhandles--;
	if(node->dependents.size()==0 && node->nhandles==0){
	  if(node->batcher) node->batcher->kill(node);
	  else kill(node); 
	}
    }


    //void kill(Chandle* hdl){
      // what about checking node??
      //handles.erase(hdl);
      //delete hdl;
    //}


  public: // ---- Backend ------------------------------------------------------------------------------------


    Coperator* get_task(){ // visited by workers
      Coperator* op; 
      unique_lock<mutex> lock(get_task_mx);
      get_task_cv.wait(lock,[this](){return ready.size()>0 || shutdown;});
      {
#ifdef ENGINE_PRIORITY
	priority_guard<3> lock(done_pmx,2); 
#else
	lock_guard<mutex> lock(done_mx);
#endif
	lock_guard<mutex> lock2(ready_mx);
	if(ready.size()==0) return nullptr;
	op=ready.front()->op;
	ready.pop_front();
	op->owner->working=true; 
	//{CoutLock lk; for(auto p:ready) cout<<p->ident()<<" "; cout<<endl;}
      }
      get_task_cv.notify_one();
      return op;      
    }
    
  };



  inline void Cworker::run(){
    while(!owner->shutdown){
      //this_thread::sleep_for(chrono::milliseconds(10));
      Coperator* op=owner->get_task();
      if(op){
	DEBUG_ENGINE({CoutLock lk; 
	    cout<<"    \e[1mWorker "<<id<<":\e[0m  "<<op->owner->ident()<<" <- "<<op->str()<<endl;});
	//uniform_int_distribution<int> distr(0,10);
	//this_thread::sleep_for(chrono::milliseconds(distr(rndGen)));
	op->exec();
	owner->done(op->owner);
      }
    }
  }


  // ---- Functions -----------------------------------------------------------------------------------------


  //inline Chandle* new_handle(Cnode* node){
  //return Cengine_engine->new_handle(node);
  //}
  


}

#endif
