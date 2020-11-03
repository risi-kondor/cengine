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
#include "MetaRbatcher.hpp"
#include "CengineHelpers.hpp"

namespace Cengine{


  class Cengine: public BasicCnodeEngine{
  public:

    int nbatchers;

    set<Cnode*> nodes;
    set<Cnode*> waiting;
    //set<Cnode*> tokill;
    
    vector<Cworker*> workers;
    vector<Batcher*> batchers;
    bool batching=true; 
    bool shutdown=false; 

    int nnodes=0;
    int nhandles=0;

#ifdef ENGINE_PRIORITY
    priority_mutex<3> done_pmx;
#else
    mutex done_mx;
#endif

    condition_variable get_task_cv;
    mutex get_task_mx;
    atomic<int> active_workers; 
    mutex active_workers_mx;
    condition_variable active_workers_cv;

    atomic<int> active_batchers; 
    mutex active_batchers_mx;
    condition_variable active_batchers_cv;

    deque<Cnode*> ready;
    mutex ready_mx;
    mutex ready_list_empty_mx;
    condition_variable ready_list_empty_cv;

    deque<Cnode*> ready_batchers;
    mutex ready_batchers_mx;
    mutex ready_batchers_empty_mx;
    condition_variable ready_batchers_empty_cv;

    int nmetarbatchers;
    vector<Rbatcher_base*> metarbatchers;

    thread* sentinel;

  public:

    Cengine(): Cengine(3){}

    Cengine(const int _nworkers){
      active_workers=_nworkers;
      active_batchers=0; 
      for(int i=0; i<_nworkers; i++)
	workers.push_back(new Cworker(this,i));
      if(false){
      sentinel=new thread([this](){
	  while(true){
	    cout<<":"<<active_workers<<endl;
	    this_thread::sleep_for(chrono::milliseconds(500)); 
	  }
	});
      }
    }


    ~Cengine(){
      DEBUG_ENGINE({CoutLock lk; cout<<"\e[1mShutting down engine.\e[0m"<<endl;});
      CENGINE_TRACE("\e[1mShutting down engine.\e[0m");
      shutdown=true; 
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


  public: // ---- Push templates -----------------------------------------------------------------------------


    template<typename OP>
    Chandle* push(Chandle* h0){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0))));
    }

    template<typename OP>
    Chandle* push(Chandle* h0, Chandle* h1){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1))));
    }

    template<typename OP>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),nodeof(h2))));
    }

    template<typename OP>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, Chandle* h3){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),nodeof(h2),nodeof(h3))));
    }

    template<typename OP>
    Chandle* push(vector<Chandle*> v){
      vector<Cnode*> n(v.size());
      for(int i=0; i<v.size(); i++) n[i]=v[i]->node;
      return enqueue_for_handle(new OP(n));
    }

    template<typename OP>
    Chandle* push(Chandle* h0, vector<Chandle*> v){
      vector<Cnode*> n(v.size());
      for(int i=0; i<v.size(); i++) n[i]=v[i]->node;
      return enqueue_for_handle(new OP(nodeof(h0),n));
    }


    // ---- 1 arg

    template<typename OP, typename ARG0>
    Chandle* push(const ARG0& arg0){ // changed!!
      return new_handle(enqueue_for_handle(new OP(arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, const ARG0 arg0){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),nodeof(h2),arg0)));
    }

    template<typename OP, typename ARG0>
    Chandle* push(Chandle* h0, vector<const Chandle*> _v1, const ARG0 arg0){
      vector<Cnode*> v1(_v1.size());
      for(int i=0; i<_v1.size(); i++) v1[i]=nodeof(_v1[i]);
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),v1,arg0)));
    }

    // ----  2 args
    
    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue_for_handle(new OP(arg0,arg1)));
    }

    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(Chandle* h0, const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0), arg0,arg1)));
    }

    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), arg0, arg1)));
    }

    template<typename OP, typename ARG0, typename ARG1>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0, const ARG1 arg1){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), nodeof(h2), arg0, arg1)));
    }

    // ---- 3 args 

    template<typename OP, typename ARG0, typename ARG1, typename ARG2>
    Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2){
      return new_handle(enqueue_for_handle(new OP(arg0,arg1,arg2)));
    }

    template<typename OP, typename ARG0, typename ARG1, typename ARG2>
    Chandle* push(Chandle* h0, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),arg0,arg1,arg2)));
    }

    template<typename OP, typename ARG0, typename ARG1, typename ARG2>
    Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),arg0,arg1,arg2)));
    }

    template<typename OP, typename ARG0, typename ARG1, typename ARG2>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),nodeof(h2),arg0,arg1,arg2)));
    }

    // ---- 4 args 

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3>
    Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3){
      return new_handle(enqueue_for_handle(new OP(arg0,arg1,arg2,arg3)));
    }

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3>
    Chandle* push(Chandle* h0, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),arg0,arg1,arg2,arg3)));
    }

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3>
    Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),arg0,arg1,arg2,arg3)));
    }

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3>
    Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3){
      return new_handle(enqueue_for_handle(new OP(nodeof(h0),nodeof(h1),nodeof(h2),arg0,arg1,arg2,arg3)));
    }


    // ---- 5 args 

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3, typename ARG4>
    Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3, const ARG4 arg4){
      return new_handle(enqueue_for_handle(new OP(arg0,arg1,arg2,arg3,arg4)));
    }


    // ---- 6 args 

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3, typename ARG4, typename ARG5>
    Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3, const ARG4 arg4, const ARG5 arg5){
      return new_handle(enqueue_for_handle(new OP(arg0,arg1,arg2,arg3,arg4,arg5)));
    }


    // ---- 7 args 

    template<typename OP, typename ARG0, typename ARG1, typename ARG2, typename ARG3, typename ARG4, typename ARG5, typename ARG6>
    Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2, const ARG3 arg3, const ARG4 arg4, const ARG5 arg5, const ARG6 arg6){
      return new_handle(enqueue_for_handle(new OP(arg0,arg1,arg2,arg3,arg4,arg5,arg6)));
    }


    // ---- Direct access ------------------------------------------------------------------------------------


    void direct(Chandle* h, std::function<void(Cobject& obj)> f){
      flush(h->node);
      f(*h->node->obj);
    }

    template<typename RET>
    RET direct(Chandle* h, std::function<RET(Cobject& obj)> f){
      flush(h->node);
      return f(*h->node->obj);
    }


    // ---- Enqueue ------------------------------------------------------------------------------------------


    Chandle* operator()(Coperator* op){
      return enqueue_for_handle(op);
    }


    Chandle* enqueue_for_handle(Coperator* op){ // Protected by done_mx
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx,0);
#else
      lock_guard<mutex> lock(done_mx); 
#endif
      Cnode* r=enqueue_sub(op);
      Chandle* hdl=new Chandle(r);
      nhandles++;
      hdl->id=nhandles-1;
      return hdl;
    }

    
    Cnode* enqueue_sub(Coperator* op){
      Cnode* node=new_node(op);
      node->engine=this;
      Cnode* rnode=node;
      Cnode* sibling=nullptr; 

      // An in-place operator is dependent on on 
      // all dependents of its self-argument
      if(dynamic_cast<InPlaceOperator*>(op)){
	for(auto p: op->inputs[0]->dependents){
	  if((p->dependents.insert(node)).second){
	    if(!p->computed) node->nblockers++;
	  }
	}
      }

      // Delegate to batched operator  
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
      if(dynamic_cast<CumulativeOperator*>(op)){
	Cnode* father=op->inputs[0];
	Coperator* father_op=father->op; 
	if(!father->computed && !father->working){

	  if(dynamic_cast<CumulativeOperator*>(father_op)){
	    DEBUG_ENGINE2("    Creating diamond");
	    CENGINE_TRACE("Creating diamond");
	    Cnode* grandfather=father->father();
	    for(auto& p:op->inputs)
	      if(p==father) p=grandfather;
	    sibling=father;

	    if(dynamic_cast<RbatchedOperator*>(op) && typeid(*father_op)==typeid(*op) && !father->released){
	      rbatch_with_sibling(father,node); // if father is released should really wait on it 
	    }
	  }

	  if(dynamic_cast<diamond_op*>(father_op) && !father->released){ 
	    DEBUG_ENGINE2("    (Extending diamond)");
	    CENGINE_TRACE("(Extending diamond)");
	    Cnode* greatgrandfather=father->father()->father();
	    for(auto& p:op->inputs)
	      if(p==father) p=greatgrandfather;
	    node->dependents.insert(father);
	    node->is_view=true;
	    father_op->inputs.push_back(node);
	    father->nblockers++;
	    rnode=father;

	    if(dynamic_cast<RbatchedOperator*>(op))
	      for(int i=0; i<father_op->inputs.size()-1; i++){
		Cnode* grandfather=father_op->inputs[i];
		if(typeid(*grandfather->op)==typeid(*op) && !grandfather->released){
		  rbatch_with_sibling(grandfather,node);
		  break;
		}
	      }
	  }
	}
      }

      if(dynamic_cast<InPlaceOperator*>(op)){
	op->inputs[0]->is_view=true; 
      }

      for(auto p: op->inputs){
	if((p->dependents.insert(node)).second){ //changed
	  if(!p->computed) {node->nblockers++;}// COUT(">>"<<p->ident());}
	}
      }

      // Complete diamond 
      if(sibling){
	Cnode* nnode=enqueue_sub(new diamond_op(sibling,node));
	//nnode->obj=node->obj;
	node->is_view=true;
	sibling->is_view=true;
	rnode=nnode; 
      }

      if(!node->rbatcher){
	if(node->nblockers==0){
	  release(node);
	  DEBUG_ENGINE2("    Early   "<<node->ident()<<" ["<<node->op->str()<<"]");
	  CENGINE_TRACE("Early      "+node->ident()+" ["+node->op->str()+"] ");
	}else{
	  waiting.insert(node);
	  DEBUG_ENGINE2("    Queuing "<<node->ident()<<" ["<<node->op->str()<<"]");
	  CENGINE_TRACE("Queuing    "+node->ident()+" ["+node->op->str()+"] ");
	}
      }else{
	if(node->nblockers==0){
	  node->rbatcher->release(node);
	}
      }  

      nodes.insert(node);
      return rnode;
    }

    // ---- Releasing nodes ----------------------------------------------------------------------------------


    void release(Cnode* node){ // visited by workers but protected by done_mx
      //DEBUG_ENGINE2("    Releasing "<<node->ident());

      if(waiting.find(node)!=waiting.end()) waiting.erase(node);
      node->released=true;

      //if(node->rbatcher){ // if rbatched do not put on ready_list 
      //node->rbatcher->release(node);
      //return; 
      //}

      {
	lock_guard<mutex> lock(ready_mx);
	auto it=find(ready.begin(), ready.end(),node);
	if(it!=ready.end()) ready.erase(it);
	ready.push_back(node);
      }
      //{CoutLock lk; for(auto p:ready) cout<<p->ident()<<" "; cout<<endl;}
      get_task_cv.notify_one();
    }


    void release_batcher(Cnode* node){ // protected by done_mx
      //DEBUG_ENGINE2("    Releasing "<<node->ident());
      node->released=true; 
      {
	lock_guard<mutex> lock(ready_batchers_mx);
	auto it=find(ready_batchers.begin(), ready_batchers.end(),node);
	if(it!=ready_batchers.end()){
	  CoutLock lk; cout<<"Batcher already released."<<endl; 
	  ready_batchers.erase(it);
	}
	ready_batchers.push_back(node);
      }
      get_task_cv.notify_one();
    }


    // ---- Finishing operators ------------------------------------------------------------------------------


    void done(Cnode* node){ // visited by workers
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx,1); 
#else
      lock_guard<mutex> lock(done_mx);
#endif

      //DEBUG_ENGINE2("    Done "<<node->ident());

      //waiting.erase(node);

      Coperator* op=node->op; 
      if(op){
	for(int i=0; i<op->inputs.size(); i++){
	  Cnode* p=op->inputs[i];
	  for(int j=0; j<i; j++) 
	    if(op->inputs[j]==p){p=nullptr;}
	  if(p!=nullptr) p->remove_dependent(node);
	}
      }

      for(auto p: node->dependents){
	p->remove_blocker(node);
      }

      node->computed=true;
      node->working=false;

      if(dynamic_cast<BatcherExecutor*>(op)){
	delete node; // changed!
	node=nullptr;
	{lock_guard<mutex> lock(active_batchers_mx); active_batchers--;}//why was this wrong?
	if(active_batchers==0) active_batchers_cv.notify_one();
      }
      
      if(node && node->dependents.size()==0 && node->nhandles==0){ // may lead to orphan nodes 
	kill(node);
      }

      if(ready.size()==0) 
	ready_list_empty_cv.notify_one();

      if(ready_batchers.size()==0) 
	ready_batchers_empty_cv.notify_one();

    }


    void kill(Cnode* node){
      //DEBUG_ENGINE2("    Killing "<<node->ident()); 
      CENGINE_TRACE("Killing "+node->ident()); 

      if(node->dependents.size()>0){
	CENGINE_DUMP_TRACE();
	CoutLock lk; cout<<"Caught dependent"<<endl; exit(-1);
      }

      if(node->nhandles>0){
	CENGINE_DUMP_TRACE();
	CoutLock lk; cout<<"Caught handle"<<endl; exit(-1); return;
      }

      {
	lock_guard<mutex> lock(ready_mx);
	auto it=find(ready.begin(), ready.end(),node);
	if(it!=ready.end()) ready.erase(it);
      }

      if(node->working){
	//{CoutLock lk; cout<<"Caught working N"<<node->id<<endl;} 
	//exit(-1);
	//tokill.insert(node);
	return;
      }

      if(nodes.find(node)==nodes.end()){
	CENGINE_DUMP_TRACE();
	CoutLock lk; cout<<"Cannot find node "<<endl; cout<<"N"<<node->id<<endl; exit(-1);
      }

      nodes.erase(node);
      delete node; 

    }


    // ---- Flushing -----------------------------------------------------------------------------------------

    
    void dump_batchers(){
      DEBUG_FLUSH({CoutLock lk; cout<<"Dumping batchers..."<<endl;});
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx,0);
#else
      lock_guard<mutex> lock(done_mx); 
#endif
      for(auto p:batchers)
	p->flush(); 
      for(auto p:metarbatchers)
	p->flush(); 
    }


    void flush(Cnode* node){
      flush(); 
     }


    void flush(){ // not protected by done_mx 
      DEBUG_ENGINE2(endl<<"    \e[1mFlushing engine...\e[0m");
      CENGINE_TRACE("\e[1mFlushing engine...\e[0m");
      int h=0;
      bool all_done=false;
      while(true){
	all_done=true; 

	dump_batchers(); 

	if(ready_batchers.size()>0){
	  DEBUG_FLUSH2("Flushing "<<ready_batchers.size()<<" batchers on ready list");
	  unique_lock<mutex> lock(ready_batchers_empty_mx);
	  ready_batchers_empty_cv.wait(lock,[this](){return ready_batchers.size()==0;});
	}

	while(ready.size()>0){
	  DEBUG_FLUSH2("Flushing "<<ready.size()<<" operations on ready list");
	  unique_lock<mutex> lock(ready_list_empty_mx);
	  ready_list_empty_cv.wait(lock,[this](){return ready.size()==0;});
	}

	if(true){
	  unique_lock<mutex> block(active_batchers_mx);
	  active_batchers_cv.wait(block,[this](){
	      if(active_batchers>0) DEBUG_FLUSH2("Waiting for "<<active_batchers<<" active batchers");
	      return active_batchers==0;
	    });
	}

	if(true){
	  unique_lock<mutex> wlock(active_workers_mx);
	  active_workers_cv.wait(wlock,[this](){
	      if(active_workers>0) DEBUG_FLUSH2("Waiting for "<<active_workers<<" workers");
	      return active_workers==0;
	    });
	}

	//DEBUG_FLUSH2(".");

	for(auto p:batchers)
	  if(p->npending()>0){
	    //cout<<"a:"<<p->npending()<<endl; 
	    all_done=false; break;
	  }
	if(ready.size()>0) all_done=false;
	if(ready_batchers.size()>0) all_done=false;
	if(waiting.size()>0) all_done=false;
	if(all_done) break;
	//COUT(ready.size()<<" "<<ready_batchers.size()<<" "<<waiting.size());

	if(h++>100){
	  CoutLock lk; cout<<"Timeout. "<<endl; 
	  for(auto p:waiting) cout<<p->str()<<endl;
	  cout<<"---"<<endl;
	  for(auto p:ready) cout<<p->str()<<endl;
	  exit(0);
	}

	//this_thread::sleep_for(chrono::milliseconds(13));

      }

      while(active_workers>0){
	{CoutLock lk; cout<<"/"<<endl;}
	/*
	bool all_done=true;
	for(auto p:workers){
	  if(p->working) all_done=false;
	}
	if(all_done) break;
	*/
	this_thread::sleep_for(chrono::milliseconds(13));	
      }

      DEBUG_FLUSH2("done.");
      DEBUG_ENGINE2("    \e[1mFlushed.\e[0m")
      CENGINE_TRACE("\e[1mFlushed.\e[0m")
      return; 
    }


  public: // ---- Handles ------------------------------------------------------------------------------------

 
    Chandle* new_handle(Cnode* node){
     Chandle* hdl=new Chandle(node);
     nhandles++;
     hdl->id=nhandles-1;
     return hdl;
    }

    Chandle* new_handle(Chandle* h){
      return h;
    }

    void dec_handle(Cnode* node){
#ifdef ENGINE_PRIORITY
	priority_guard<3> lock(done_pmx,1); 
#else
	lock_guard<mutex> lock(done_mx);
#endif
	node->nhandles--;
	//DEBUG_ENGINE2(node->ident()<<" nh="<<node->nhandles);
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

    

  public: // ---- Rbatching ----------------------------------------------------------------------------------


    void rbatch_with_sibling(Cnode* sibling, Cnode* node){

      if(sibling->rbatcher){
	sibling->rbatcher->push(node);
	return;
      }

      RbatchedOperator* bop=dynamic_cast<RbatchedOperator*>(node->op);
      Rbatcher_base* meta;
      if(bop->metarbatcher_id()==0){
	bop->set_metarbatcher_id(++nmetarbatchers);
	meta=bop->spawn_metarbatcher(this);
	metarbatchers.push_back(meta);
      }

      meta=metarbatchers[bop->metarbatcher_id()-1];
      meta->push(sibling);
      waiting.erase(sibling); 
      sibling->rbatcher->push(node);
  
    }     


  public: // ---- Backend ------------------------------------------------------------------------------------


    Coperator* get_task(Cworker* worker){ // visited by workers
      Coperator* op; 

      worker->working=false; 
      {lock_guard<mutex> lock(active_workers_mx); active_workers--;} // probably don't need lock
      if(active_workers==0) active_workers_cv.notify_one();
      
      unique_lock<mutex> lock(get_task_mx);
      get_task_cv.wait(lock,[this](){return ready.size()>0 || ready_batchers.size()>0 || shutdown;});

      {
#ifdef ENGINE_PRIORITY
	priority_guard<3> lock(done_pmx,2); 
#else
	lock_guard<mutex> lock(done_mx);
#endif
	lock_guard<mutex> lock2(ready_mx);

	{lock_guard<mutex> lock(active_workers_mx); active_workers++;}

	if(ready_batchers.size()>0){
	  worker->working=true;
	  op=ready_batchers.front()->op;
	  ready_batchers.pop_front();
	  op->owner->working=true; 
	  get_task_cv.notify_one();
	  active_batchers++;
	  return op;      
	}

	if(ready.size()>0){
	  worker->working=true;
	  op=ready.front()->op;
	  ready.pop_front();
	  op->owner->working=true; 
	  get_task_cv.notify_one();
	  return op;      
	}
	
	return nullptr;

      }

    }      
  };


  // ---- Functions -----------------------------------------------------------------------------------------


  inline void Cworker::run(){
    while(!owner->shutdown){
      Coperator* op=owner->get_task(this);
      if(op){
	DEBUG_ENGINE({CoutLock lk; 
	    cout<<"    \e[1mWorker "<<id<<":\e[0m  "<<op->owner->ident()<<" <- "<<op->str()<<endl;});
	CENGINE_TRACE("\e[1mWorker "+to_string(id)+":\e[0m  "+op->owner->ident()+" <- "+op->str());
	//if(dynamic_cast<BatcherExecutor*>(op)) op->exec();
	op->exec();
	owner->done(op->owner);
      }
    }
  }


}

#endif

