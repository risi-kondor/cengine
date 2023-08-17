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
#ifndef _Chandle
#define _Chandle

#include "Cnode.hpp"

namespace Cengine{

  class Chandle{
  public:

    Cnode* node;

    int id;

    
  public:

    Chandle(Cnode* _node): node(_node){
      node->nhandles++; // Changed!
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Creating handle to "<<node->ident()<<endl;});
      CENGINE_TRACE("Creating handle to "+node->ident());
      CHANDLE_CREATE();
    }
    
    ~Chandle(){
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Deleting handle to "<<node->ident()<<endl;});
      CENGINE_TRACE("Deleting handle to "+node->ident());
      node->engine->dec_handle(node);
      CHANDLE_DESTROY();
      //cout<<"."<<endl;
    }

    Chandle(const Chandle& x){
      node=x.node;
      node->nhandles++;
      //node->engine->nhandles++;
      //id=dynamic_cast<Cengine*>(node->engine)->nhandles++;
      CHANDLE_CREATE();
    }

    Chandle(const Chandle&& x)=delete;

    Chandle& operator=(const Chandle& x)=delete;
    Chandle& operator=(Chandle&& x)=delete;

  public:

    string ident() const{
      return "H"+to_string(id);
    }

    string str() const{
      return ident()+" ["+node->ident()+"]";
    }

  };


  // ---- Functions -----------------------------------------------------------------------------------------


  inline Cnode* nodeof(Chandle* hdl){
    return hdl->node;
  }

  inline Cnode* nodeof(const Chandle* hdl){
    return hdl->node;
  }

  inline void replace(Chandle*& target, Chandle* hdl){
    delete target;
    target=hdl;
  }




}

#endif
