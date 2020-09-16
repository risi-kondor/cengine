#ifndef _InterfaceBase
#define _InterfaceBase

#include "Cengine.hpp"
#include "RscalarB_ops.hpp"
#include "CscalarB_ops.hpp"
#include "CtensorB_ops.hpp"
#include "CtensorB_constructor_ops.hpp"
#include "CtensorB_cumulative_ops.hpp"
#include "CtensorB_add_Mprod_ops.hpp"
#include "CtensorB_mix_ops.hpp"

extern ::Cengine::Cengine* Cengine_engine;


namespace Cengine{
  
  inline Chandle* new_handle(Cnode* node){
    return Cengine_engine->new_handle(node);
  }

  vector<float> rscalar_get(Chandle* hdl){
    Cengine_engine->flush(hdl->node);
    return asRscalarB(hdl->node->obj,__PRETTY_FUNCTION__);
  }
  
  vector<complex<float> > cscalar_get(Chandle* hdl){
    Cengine_engine->flush(hdl->node);
    return asCscalarB(hdl->node->obj,__PRETTY_FUNCTION__);
  }
  
  Gtensor<complex<float> > ctensor_get(Chandle* hdl){
    Cengine_engine->flush(hdl->node);
    cout<<hdl->node->nhandles<<endl; 
    return asCtensorB(hdl->node->obj,__PRETTY_FUNCTION__);
  }

  
  
}


#endif


  //  inline void replace(Chandle*& target, Chandle* hdl){
  //delete target;
  //    target=hdl;
  //}

