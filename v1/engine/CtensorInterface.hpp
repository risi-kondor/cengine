#ifndef _CtensorInterface
#define _CtensorInterface

#include "Cengine.hpp"
#include "InterfaceBase.hpp"
//#include "RscalarInterface.hpp"
#include "CscalarInterface.hpp"

#include "CtensorB_constructor_ops.hpp"
#include "CtensorB_cumulative_ops.hpp"
#include "CtensorB_ops.hpp"
#include "CtensorB_add_Mprod.hpp"


extern ::Cengine::Cengine* Cengine_engine;


namespace Cengine{

  namespace engine{

    Chandle* new_ctensor(const Gdims& dims, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_ctensor_op(dims,nbd,device));
      return new_handle(node);
    }

    Chandle* new_ctensor_zero(const Gdims& dims, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_ctensor_zero_op(dims,nbd,device));
      return new_handle(node);
    }

    Chandle* new_ctensor_ones(const Gdims& dims, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_ctensor_ones_op(dims,nbd,device));
      return new_handle(node);
    }

    Chandle* new_ctensor_identity(const Gdims& dims, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_ctensor_identity_op(dims,nbd,device));
      return new_handle(node);
    }

    Chandle* new_ctensor_sequential(const Gdims& dims, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_ctensor_sequential_op(dims,nbd,device));
      return new_handle(node);
    }

    Chandle* new_ctensor_gaussian(const Gdims& dims, const int nbd=-1, const int device=0){
      Cnode* node=Cengine_engine->enqueue(new new_ctensor_gaussian_op(dims,nbd,device));
      return new_handle(node);
    }

    Chandle* new_ctensor_from_gtensor(const Gtensor<complex<float> >& x, const int nbd=-1, const int device=0){
      return new_handle(Cengine_engine->enqueue(new new_ctensor_from_gtensor_op(x,nbd,device)));
    }

    Chandle* ctensor_copy(Chandle* x){
      Cnode* node=Cengine_engine->enqueue(new ctensor_copy_op(nodeof(x)));
      return new_handle(node);
    }


    // ---- Operations ---------------------------------------------------------------------------------------


    Chandle* ctensor_zero(Chandle* r){
      return new_handle(Cengine_engine->enqueue(new ctensor_zero_op(nodeof(r))));
    }

    Chandle* ctensor_conj(Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_conj_op(nodeof(x))));
    }

    Chandle* ctensor_transp(Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_transp_op(nodeof(x))));
    }

    Chandle* ctensor_herm(Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_herm_op(nodeof(x))));
    }


    Chandle* ctensor_normalize_cols(Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_normalize_cols_op(nodeof(x))));
    }

    Chandle* ctensor_add_normalize_cols_back(Chandle* r, Chandle*g, Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_normalize_cols_back_op(nodeof(r),nodeof(g),nodeof(x))));
    }


    // ---- Cumulative operations ----------------------------------------------------------------------------

    
    Chandle* ctensor_add(Chandle* r, Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_op(nodeof(r),nodeof(x))));
    }

    Chandle* ctensor_add_conj(Chandle* r, Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_conj_op(nodeof(r),nodeof(x))));
    }

    Chandle* ctensor_add_transp(Chandle* r, Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_transp_op(nodeof(r),nodeof(x))));
    }

    Chandle* ctensor_add_herm(Chandle* r, Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_herm_op(nodeof(r),nodeof(x))));
    }


    Chandle* ctensor_subtract(Chandle* r, Chandle* x){
      return new_handle(Cengine_engine->enqueue(new ctensor_subtract_op(nodeof(r),nodeof(x))));
    }


    Chandle* ctensor_add_times_real(Chandle* r, Chandle* x, float c){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_times_real_op(nodeof(r),nodeof(x),c)));
    }

    Chandle* ctensor_add_times_complex(Chandle* r, Chandle* x, complex<float> c){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_times_complex_op(nodeof(r),nodeof(x),c)));
    }


    Chandle* ctensor_add_prod_rA(Chandle* r, Chandle* c, Chandle* A){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_prod_rA_op(nodeof(r),nodeof(c),nodeof(A))));
    }

    Chandle* ctensor_add_prod_cA(Chandle* r, Chandle* c, Chandle* A){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_prod_cA_op(nodeof(r),nodeof(c),nodeof(A))));
    }

    Chandle* ctensor_add_prod_cc_A(Chandle* r, Chandle* c, Chandle* A){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_prod_cc_A_op(nodeof(r),nodeof(c),nodeof(A))));
    }

    Chandle* ctensor_add_prod_c_Ac(Chandle* r, Chandle* c, Chandle* A){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_prod_c_Ac_op(nodeof(r),nodeof(c),nodeof(A))));
    }


    /*
    Chandle* ctensor_add_Mprod(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<0,0>(nodeof(r),nodeof(A),nodeof(B))));
    }

    Chandle* ctensor_add_Mprod_AT(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<2,0>(nodeof(r),nodeof(A),nodeof(B))));
    }

    Chandle* ctensor_add_Mprod_TA(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<1,0>(nodeof(r),nodeof(A),nodeof(B))));
    }

    Chandle* ctensor_add_Mprod_AC(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<0,2>(nodeof(r),nodeof(A),nodeof(B))));
    }

    Chandle* ctensor_add_Mprod_TC(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<1,2>(nodeof(r),nodeof(A),nodeof(B))));
    }

    Chandle* ctensor_add_Mprod_AH(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<2,2>(nodeof(r),nodeof(A),nodeof(B))));
    }

    Chandle* ctensor_add_Mprod_HA(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_Mprod_op<1,1>(nodeof(r),nodeof(A),nodeof(B))));
    }
    */


    Chandle* ctensor_add_ReLU(Chandle* r, Chandle* x, const float c){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_ReLU_op(nodeof(r),nodeof(x),c)));
    }

    Chandle* ctensor_add_ReLU_back(Chandle* r, Chandle* g, Chandle* x, const float c){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_ReLU_back_op(nodeof(r),nodeof(g),nodeof(x),c)));
    }


    Chandle* ctensor_add_inp(Chandle* r, Chandle* A, Chandle* B){
      return new_handle(Cengine_engine->enqueue(new ctensor_add_inp_op(nodeof(r),nodeof(A),nodeof(B))));
    }



    // ---- Output operations --------------------------------------------------------------------------------


    Gtensor<complex<float> > ctensor_get(Chandle* hdl){
      Cengine_engine->flush(hdl->node);
      cout<<hdl->node->nhandles<<endl; 
      return asCtensorB(hdl->node->obj,__PRETTY_FUNCTION__);
    }

  }
}

#endif 
