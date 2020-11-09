#ifndef _CtensorB_rbatched_ops
#define _CtensorB_rbatched_ops

#include "MetaRbatcher.hpp"
#include "Rbatcher.hpp"
#include "CscalarBreducer.hpp"
#include "CtensorBreducer.hpp"
#include "ctensor_signature.hpp"


namespace Cengine{


  class ctensor_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
			public RbatchedOperator{
  public:

    Gdims dims; 

    ctensor_add_op(Cnode* r, Cnode* x, const Gdims& _dims):
      Coperator(r,x), dims(_dims){}

    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).add(CTENSORB(inputs[1]));
      //this_thread::sleep_for(chrono::milliseconds(5)); 
    }

    void rbatched_exec(const vector<Cnode*>& nodes){
      int dev=CTENSORB(nodes[0]->op->inputs[0]).device;
      assert(dev==1);

      CtensorBreducer R(nodes.size(),CTENSORB(nodes[0]->op->inputs[0]));
      CtensorBpack X(nodes,1);
      R.add(X);
    }



  public:

    string str() const{
      return "ctensor_add"+inp_str();
    }

    static string classname(){
      return "ctensor_add_op";
    }

    static int _batcher_id;
    void set_batcher_id(const int i){_batcher_id=i;}
    int batcher_id() const{return _batcher_id;}
    string batcher_name() const{return "ctensor_add<"+rsignature().str()+">";}

    static int _rbatcher_id;
    void set_rbatcher_id(const int i){_rbatcher_id=i;}
    int rbatcher_id() const{return _rbatcher_id;}
    string rbatcher_name() const{return "ctensor_add<"+rsignature().str()+">";}
    ctensor_signature rsignature() const{return ctensor_signature(dims);}
    Rbatcher_base* spawn_rbatcher(BasicCnodeEngine* engine) const{
      return new MetaRbatcher<ctensor_add_op,ctensor_signature,Rbatcher>(engine);
    }
    
    
  };
  


  class ctensor_add_prod_c_A_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
			public RbatchedOperator{
  public:

    Gdims dims; 

    ctensor_add_prod_c_A_op(Cnode* r, Cnode* c, Cnode* A, const Gdims& _dims):
      Coperator(r,c,A), dims(_dims){}

    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).add_prod(CSCALARB(inputs[1]),CTENSORB(inputs[2]));
      //this_thread::sleep_for(chrono::milliseconds(5)); 
    }

    void rbatched_exec(const vector<Cnode*>& nodes){
      CtensorBreducer R(nodes.size(),CTENSORB(nodes[0]->op->inputs[0]));
      CscalarBpack c(nodes,1);
      CtensorBpack A(nodes,2);
      R.add_prod(c,A);
    }



  public:

    string str() const{
      return "ctensor_add"+inp_str();
    }

    static string classname(){
      return "ctensor_add_prod_c_A_op";
    }

    /*
      static int _batcher_id;
      void set_batcher_id(const int i){_batcher_id=i;}
      int batcher_id() const{return _batcher_id;}
      string batcher_name() const{return "ctensor_add<"+rsignature().str()+">";}
    */

    static int _rbatcher_id;
    void set_rbatcher_id(const int i){_rbatcher_id=i;}
    int rbatcher_id() const{return _rbatcher_id;}
    string rbatcher_name() const{return "ctensor_add_prod_c_A<"+rsignature().str()+">";}
    ctensor_signature rsignature() const{return ctensor_signature(dims);}
    Rbatcher_base* spawn_rbatcher(BasicCnodeEngine* _engine) const{
      return new MetaRbatcher<ctensor_add_op,ctensor_signature,Rbatcher>(_engine);
    }
    
    
  };
  

}

#endif 

      /*
      if(dev==0){
	for(int i=0; i<N; i++){
	  nodes[i]->obj=nodes[i]->op->inputs[0]->obj;
	  CTENSORB(nodes[i]).add(CTENSORB(nodes[i]->op->inputs[1]));
	}
      }
      */

