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
#ifndef _MetaRbatcher
#define _MetaRbatcher

#include "Cnode.hpp"
#include "Rbatcher_base.hpp"

namespace Cengine{

  class Rbatcher;

  class RbatcherSet: public set<Rbatcher*>{
  public:
    //void erase(Rbatcher* x){COUT("erre");}

  };


  template<typename OP, typename SUBINDEX, typename RBATCHER>
  class MetaRbatcher: public Rbatcher_base{
  public:

    BasicCnodeEngine* engine;

    int batchercount=0;
    //set<Rbatcher_base*> batchers;
    //unordered_map<SUBINDEX,set<RBATCHER*> > batchersets;
    unordered_map<SUBINDEX,RbatcherSet* > batchersets;

    MetaRbatcher(BasicCnodeEngine* _engine): 
      engine(_engine){
      DEBUG_ENGINE2("    \e[1mNew MetaRbatcher for "<<OP::classname()<<"\e[0m") 
    }

    virtual ~MetaRbatcher(){
      //for(auto& p: batchers) delete p;

      for(auto p: batchersets){
	for(auto q: *p.second)
	  delete q; 
	delete p.second;
      }
    }


  public:

    void push(Cnode* node){
      OP* op=static_cast<OP*>(node->op);
      SUBINDEX ix=dynamic_cast<OP*>(op)->rsignature();


      auto it=batchersets.find(ix);
      if(it==batchersets.end()) 
	batchersets[ix]=new RbatcherSet;

      RbatcherSet* bset=batchersets[ix];
      //COUT("ab");
      RBATCHER* sub=new RBATCHER(engine,op->rbatcher_name());
      sub->id=batchercount++;
      sub->push(node);
      bset->insert(sub);
    }

    int flush(){
      //cout<<"k"<<batchers.size()<<endl;
      int nwaiting=0; 
      //for(auto p:batchers)
      //nwaiting+=p->flush();
      for(auto p:batchersets){
	vector<RBATCHER*> toerase;
	for(auto q:*p.second){
	  nwaiting+=q->flush();
	  if(q->nwaiting()==0)
	    toerase.push_back(q);
	}
	for(auto q:toerase){
	  p.second->erase(q);
	  delete q;
	}
      }
      return nwaiting; 
    }

    int npending() const{
      int t=0;
      //for(auto p:batchers)
      //t+=p->npending();
      for(auto p:batchersets)
	for(auto q:*p.second)
	  t+=q->npending();
      //t+=q->flush(); ???
      return t;
    }
    
    void release(Cnode* node){
    }

    void kill(Cnode* node){
    }

    void release(){}

  };

}


#endif

