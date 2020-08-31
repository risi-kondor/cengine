#ifndef _Cnode
#define _Cnode

#include "Coperator.hpp"
#include "Cobject.hpp"

namespace Cengine{


  class BasicCnodeEngine{
  public:

    virtual Cnode* new_node(Coperator* op)=0;
    virtual void release(Cnode* node)=0;
    virtual void done(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;
    //virtual void protected_kill(Cnode* node)=0;
    virtual void dec_handle(Cnode* node)=0;

  };


  class Batcher{
  public:

    virtual ~Batcher(){}

    virtual void push(Coperator* op)=0;
    virtual void release(Cnode* node)=0;
    virtual void kill(Cnode* node)=0;

    virtual void exec()=0; 
    virtual int flush()=0; 

  };


  class Cnode{
  public:

    BasicCnodeEngine* engine=nullptr;
    Batcher* batcher=nullptr; 

    Coperator* op=nullptr;
    Cobject* obj=nullptr;

    set<Cnode*> dependents; 
    int nblockers=0;
    int nhandles=0;
    bool released=false; 
    bool working=false; 
    bool computed=false;
    bool is_view=false; 

    int id;


  public:

    Cnode(Coperator* _op): op(_op){}

    ~Cnode(){
      if(nhandles>0){CoutLock lk; cout<<"Error: attempting to delete node with handle."<<endl;exit(0);}
      // DEBUG_ENGINE({CoutLock lk; cout<<"Deleting "<<ident()<<endl;});
      delete op;
      if(!is_view) delete obj; // !!!!
    }

    
  public:

    void remove_blocker(Cnode* blocker){ // protected_by done_mx 
      nblockers--;
      if(nblockers==0){
	if(batcher) batcher->release(this); 
	else engine->release(this);
      }
    }
    
    void remove_dependent(Cnode* dependent){ // protected by done_mx
      if(dependents.find(dependent)==dependents.end()) {CoutLock lk; cout<<"Dependent not found"<<endl;}
      dependents.erase(dependent);
      if(dependents.size()==0 && nhandles==0){
	if(batcher) batcher->kill(this);
	else engine->kill(this);
      }
    }

    //void assess(){ // may be called from master thread, not protected 
    //if(dependents.size()==0 && nhandles==0) engine->protected_kill(this); 
    //}

    Cnode* father(){ // protected by done_mx 
      assert(op);
      assert(op->inputs.size()>0);
      return op->inputs[0];
    }


  public:

    string ident() const{
      return "N"+to_string(id);
    }

    string str() const{
      return ident();
    }

  };


  inline string Coperator::inp_str() const{
    ostringstream oss;
    oss<<"(";
    for(int i=0; i<inputs.size()-1; i++)
      oss<<inputs[i]->ident()<<",";
    if(inputs.size()>0) oss<<inputs[inputs.size()-1]->ident();
    oss<<")";
    return oss.str();
  }


}


#endif


