#ifndef _SO3partInterface
#define _SO3partInterface

#include "Cengine.hpp"
#include "SO3partB_ops.hpp"

extern ::Cengine::Cengine* Cengine_engine;


namespace Cengine{

  namespace engine{

    Chandle* new_SO3part(const Gtensor<complex<float> >& x, const int device=0){
      return new_handle(Cengine_engine->enqueue(new new_SO3part_from_Gtensor_op(x,device)));
    }

    Chandle* new_SO3part_spharm(const int l, const float x, const float y, const float z, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_spharm_op(l,x,y,z,nbd,device));
      return new_handle(node);
    }


    // ---- Cumulative operations ----------------------------------------------------------------------------

    
    Chandle* SO3part_add_CGproduct(Chandle* r, Chandle* x, Chandle* y, const int offs){
      return new_handle(Cengine_engine->enqueue(new SO3part_add_CGproduct_op(nodeof(r),nodeof(x),nodeof(y),offs)));
    }

    Chandle* SO3part_add_CGproduct_back0(Chandle* r, Chandle* g, Chandle* y, const int offs){
      return new_handle(Cengine_engine->enqueue(new SO3part_add_CGproduct_back0_op(nodeof(r),nodeof(g),nodeof(y),offs)));
    }

    Chandle* SO3part_add_CGproduct_back1(Chandle* r, Chandle* g, Chandle* x, const int offs){
      return new_handle(Cengine_engine->enqueue(new SO3part_add_CGproduct_back1_op(nodeof(r),nodeof(g),nodeof(x),offs)));
    }

  }
}

#endif 
