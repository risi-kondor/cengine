/*
 * This file is part of Cengine, an asynchronous C++/CUDA compute engine. 
 *  
 * Copyright (c) 2020- Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _Coperator
#define _Coperator


namespace Cengine{

  class Cnode;
  class Batcher; 
  class Rbatcher_base; 
  class BasicCnodeEngine;

  class Coperator{
  public:

    Cnode* owner;
    vector<Cnode*> inputs;


  public:

    Coperator(){
      COPERATOR_CREATE();
    }

    Coperator(Cnode* x0){
      inputs.push_back(x0);
      COPERATOR_CREATE();
    }

    Coperator(Cnode* x0, Cnode* x1){
      inputs.push_back(x0);
      inputs.push_back(x1);
      COPERATOR_CREATE();
    }

    Coperator(Cnode* x0, Cnode* x1, Cnode* x2){
      inputs.push_back(x0);
      inputs.push_back(x1);
      inputs.push_back(x2);
      COPERATOR_CREATE();
    }

    Coperator(Cnode* x0, Cnode* x1, Cnode* x2, Cnode* x3){
      inputs.push_back(x0);
      inputs.push_back(x1);
      inputs.push_back(x2);
      inputs.push_back(x3);
      COPERATOR_CREATE();
    }

    Coperator(vector<Cnode*> v){
      inputs=v;
      COPERATOR_CREATE();
    }

    Coperator(Cnode* x0, vector<Cnode*> v1){
      inputs.push_back(x0);
      for(auto p:v1) inputs.push_back(p);
      COPERATOR_CREATE();
    }

    ~Coperator(){
      COPERATOR_DESTROY();
    }


  public:

    virtual void exec(){}


  public:

    virtual string str() const{
      return "";
    }

    string inp_str() const;	

    template<typename TYPE1>
    string inp_str(const TYPE1& x1) const;	
    
    template<typename TYPE1, typename TYPE2>
    string inp_str(const TYPE1& x1, const TYPE2& x2) const;	
    
    template<typename TYPE1, typename TYPE2, typename TYPE3>
    string inp_str(const TYPE1& x1, const TYPE2& x2, const TYPE3& x3) const;	
    
  };


  // ---- Subclasses ---------------------------------------------------------------------------------------


  class CumulativeOperator{};


  class InPlaceOperator{};


  class BatchedOperator{
  public: 
    virtual int batcher_id() const=0;
    virtual void set_batcher_id(const int i)=0;
    virtual Batcher* spawn_batcher() const=0;
    //virtual Batcher* spawn_batcher(BasicCnodeEngine* _engine) const=0;
    virtual string batcher_name() const {return "";}
  };

  class RbatchedOperator{
  public: 
    virtual int rbatcher_id() const=0;
    virtual void set_rbatcher_id(const int i)=0;
    virtual Rbatcher_base* spawn_rbatcher(BasicCnodeEngine* _engine) const=0;
    virtual string rbatcher_name() const {return "";}
    //virtual void rbatched_exec(BasicCnodeEngine* _engine, const vector<Cnode*>& nodes)=0; 
    virtual void rbatched_exec(const vector<Cnode*>& nodes)=0; 

  };

  class BatcherExecutor{};

}

#endif


    /*
      ostringstream oss;
      oss<<"[";
      for(int i=0; i<inputs.size(); i++){
	//oss<<inputs[i]->ident();
	if(i<inputs.size()-1) oss<<",";
      }
      //if(inputs.size()>0) oss<<inputs[inputs.size()-1]->ident();
      oss<<"]";
      return oss.str();
    }
    */
