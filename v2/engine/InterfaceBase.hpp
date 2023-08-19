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
#ifndef _InterfaceBase
#define _InterfaceBase

#include "Cengine.hpp"
#include "RscalarB_ops.hpp"
#include "CscalarB_ops.hpp"

#include "ctensor_ops.hpp"
#include "ctensor_constructor_ops.hpp"
#include "ctensor_cumulative_ops.hpp"
#include "ctensor_mix_ops.hpp"
#include "ctensor_add_ops.hpp"
#include "ctensor_add_Mprod_ops.hpp"
#include "ctensor_add_inp_op.hpp"

extern ::Cengine::Cengine* Cengine_engine;


namespace Cengine{
  
  inline Chandle* new_handle(Cnode* node){
    return Cengine_engine->new_handle(node);
  }

  vector<float> rscalar_get(Chandle* hdl){
    Cengine_engine->flush(hdl->node);
#ifdef CENGINE_DRY_RUN
    return vector<float>(0); 
#else
    return RSCALARB(hdl->node->obj);
#endif 
  }
  
  vector<complex<float> > cscalar_get(Chandle* hdl){
    Cengine_engine->flush(hdl->node);
#ifdef CENGINE_DRY_RUN
    return vector<complex<float> >(0); 
#else
    return CSCALARB(hdl->node->obj);
#endif 
  }
  
  Gtensor<complex<float> > ctensor_get(Chandle* hdl){
    Cengine_engine->flush(hdl->node);
    //cout<<hdl->node->nhandles<<endl; 
    return asCtensorB(hdl->node->obj,__PRETTY_FUNCTION__);
  }

  //Gtensor<complex<float> > ctensor_get_element(Chandle* hdl){
  //Cengine_engine->flush(hdl->node);
    //cout<<hdl->node->nhandles<<endl; 
    //return asCtensorB(hdl->node->obj,__PRETTY_FUNCTION__);
  //}

  
  
}


#endif


  //  inline void replace(Chandle*& target, Chandle* hdl){
  //delete target;
  //    target=hdl;
  //}

