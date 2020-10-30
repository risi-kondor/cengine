#ifndef _CtensorB_rbatched_ops
#define _CtensorB_rbatched_ops

//#include "CtensorB.hpp"
#include "BatcherG.hpp"
#include "MetaRbatcher.hpp"
#include "Rbatcher.hpp"
#include "ctensor_signature.hpp"


namespace Cengine{


  class ctensor_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
			//public GatheringBatchedOperator, 
			public RbatchedOperator{
  public:

    Gdims dims; 
    Cengine* engine;

    ctensor_add_op(Cnode* r, Cnode* x, const Gdims& _dims):
      Coperator(r,x), dims(_dims){}

    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).add(CTENSORB(inputs[1]));
      this_thread::sleep_for(chrono::milliseconds(50)); 
    }

    void batched_exec(const vector<GatherGroup*>& ggroup, const vector<Cnode*>& nodes){
      DEBUG_ENGINE({CoutLock lk; cout<<"    Running batched ctensor_add..."<<endl;});

      for(auto p:ggroup){
	vector<Cnode*>& nodes=p->ready;
	const int N=nodes.size();
	for(int i=0; i<N; i++){
	  nodes[i]->obj=nodes[i]->op->inputs[0]->obj;
	  CTENSORB(nodes[i]).add(CTENSORB(nodes[i]->op->inputs[1]));
	}
	for(int i=0; i<N; i++){
	  engine->done(nodes[i]);
	}
      }

      const int N=nodes.size();
      for(int i=0; i<N; i++){
	nodes[i]->obj=nodes[i]->op->inputs[0]->obj;
	CTENSORB(nodes[i]).add(CTENSORB(nodes[i]->op->inputs[1]));
      }
      for(int i=0; i<N; i++){
	engine->done(nodes[i]);
      }

      DEBUG_ENGINE({CoutLock lk; cout<<"    \e[1mDone.\e[0m"<<endl;});

    }

    void rbatched_exec(const vector<Cnode*>& nodes){
    }

    ctensor_signature signature() const{
      //return ctensor_signature(asCtensorB(inputs[1],__PRETTY_FUNCTION__).dims);
      return ctensor_signature(dims);
    }

    ctensor_signature rsignature() const{
      //return ctensor_signature(asCtensorB(inputs[1],__PRETTY_FUNCTION__).dims);
      return ctensor_signature(dims);
    }

    Batcher* spawn_batcher() const{
      return new MetaBatcher<ctensor_add_op,ctensor_signature,BatcherG>(engine);
    }

    Rbatcher_base* spawn_rbatcher(BasicCnodeEngine* _engine) const{
      return new MetaRbatcher<ctensor_add_op,ctensor_signature,Rbatcher>(_engine);
    }


  public:

    string str() const{
      return "ctensor_add"+inp_str();
    }

    static string classname(){
      return "ctensor_add_op";}

    static int _batcher_id;
    void set_batcher_id(const int i){_batcher_id=i;}
    int batcher_id() const{return _batcher_id;}
    string batcher_name() const{return "ctensor_add<"+rsignature().str()+">";}
    
    static int _rbatcher_id;
    void set_rbatcher_id(const int i){_rbatcher_id=i;}
    int rbatcher_id() const{return _rbatcher_id;}
    string rbatcher_name() const{return "ctensor_add<"+rsignature().str()+">";}
    
    
  };
  

}

#endif 