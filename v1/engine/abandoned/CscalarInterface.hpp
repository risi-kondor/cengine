#ifndef _CscalarInterface
#define _CscalarInterface

#include "InterfaceBase.hpp"
//#include "RscalarB_ops.hpp"
//#include "CscalarB_ops.hpp"

extern ::Cengine::Cengine* Cengine_engine;


namespace Cengine{

  namespace engine{


    Chandle* new_cscalar(const int nbd=-1, const int device=0){
      return Cengine_engine->push<new_cscalar_op>(nbd,device);
    }

    Chandle* new_cscalar_zero(const int nbd=-1, const int device=0){
      return Cengine_engine->push<new_cscalar_zero_op>(nbd,device);
    }

    Chandle* new_cscalar_gaussian(const int nbd=-1, const int device=0){
      return Cengine_engine->push<new_cscalar_gaussian_op>(nbd,device);
    }

    Chandle* new_cscalar_set(const complex<float> x, const int nbd=-1, const int device=0){
      return Cengine_engine->push<new_cscalar_set_op>(nbd,x,device);
    }

    Chandle* cscalar_copy(Chandle* x){
      return Cengine_engine->push<cscalar_copy_op>(nodeof(x));
    }


    // ---- Access -------------------------------------------------------------------------------------


    Chandle* cscalar_get_real(Chandle* r){
      return Cengine_engine->push<cscalar_get_real_op>(nodeof(r));
    }

    Chandle* cscalar_get_imag(Chandle* r){
      return Cengine_engine->push<cscalar_get_imag_op>(nodeof(r));
    }

    Chandle* cscalar_set_real(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_set_real_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_set_imag(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_set_imag_op>(nodeof(r),nodeof(x));
    }


    // ---- Operations ---------------------------------------------------------------------------------


    Chandle* cscalar_zero(Chandle* r){
      return Cengine_engine->push<cscalar_set_zero_op>(nodeof(r));
    }


    //Chandle* cscalar_set(Chandle* r, complex<float> c){
    //return Cengine_engine->push<cscalar_set_op>(nodeof(r),c)));
    //}


    Chandle* cscalar_conj(Chandle* x){
      return Cengine_engine->push<cscalar_conj_op>(nodeof(x));
    }


    // ---- In-place operations ------------------------------------------------------------------------

    
    Chandle* cscalar_add(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_add_to_real(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_to_real_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_add_to_imag(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_to_imag_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_add_times_real(Chandle* r, Chandle* x, float c){
      return Cengine_engine->push<cscalar_add_times_real_op>(nodeof(r),nodeof(x),c);
    }

    Chandle* cscalar_add_times_complex(Chandle* r, Chandle* x, complex<float> c){
      return Cengine_engine->push<cscalar_add_times_complex_op>(nodeof(r),nodeof(x),c);
    }

    Chandle* cscalar_add_conj(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_conj_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_subtract(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_subtract_op>(nodeof(r),nodeof(x));
    }


    Chandle* cscalar_add_prod(Chandle* r, Chandle* x, Chandle* y){
      return Cengine_engine->push<cscalar_add_prod_op>(nodeof(r),nodeof(x),nodeof(y));
    }

    Chandle* cscalar_add_prodc(Chandle* r, Chandle* x, Chandle* y){
      return Cengine_engine->push<cscalar_add_prodc_op>(nodeof(r),nodeof(x),nodeof(y));
    }

    Chandle* cscalar_add_prod_r(Chandle* r, Chandle* x, Chandle* y){
      return Cengine_engine->push<cscalar_add_prod_r_op>(nodeof(r),nodeof(x),nodeof(y));
    }



    Chandle* cscalar_add_div(Chandle* r, Chandle* x, Chandle* y){
      return Cengine_engine->push<cscalar_add_div_op>(nodeof(r),nodeof(x),nodeof(y));
    }

    Chandle* cscalar_add_div_back0(Chandle* r, Chandle* g, Chandle* y){
      return Cengine_engine->push<cscalar_add_div_back0_op>(nodeof(r),nodeof(g),nodeof(y));
    }

    Chandle* cscalar_add_div_back1(Chandle* r, Chandle* g, Chandle* x, Chandle* y){
      return Cengine_engine->push<cscalar_add_div_back1_op>(nodeof(r),nodeof(g),nodeof(x),nodeof(y));
    }


    Chandle* cscalar_add_abs(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_abs_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_add_abs_back(Chandle* r, Chandle* g, Chandle* x){
      return Cengine_engine->push<cscalar_add_abs_back_op>(nodeof(r),nodeof(g),nodeof(x));
    }

    Chandle* cscalar_add_pow(Chandle* r, Chandle* x, float p, complex<float> c){
      return Cengine_engine->push<cscalar_add_pow_op>(nodeof(r),nodeof(x),p,c);
    }

    Chandle* cscalar_add_exp(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_exp_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_add_ReLU(Chandle* r, Chandle* x, const float c){
      return Cengine_engine->push<cscalar_add_ReLU_op>(nodeof(r),nodeof(x),c);
    }

    Chandle* cscalar_add_ReLU_back(Chandle* r, Chandle* g, Chandle* x, const float c){
      return Cengine_engine->push<cscalar_add_ReLU_back_op>(nodeof(r),nodeof(g),nodeof(x),c);
    }

    Chandle* cscalar_add_sigmoid(Chandle* r, Chandle* x){
      return Cengine_engine->push<cscalar_add_sigmoid_op>(nodeof(r),nodeof(x));
    }

    Chandle* cscalar_add_sigmoid_back(Chandle* r, Chandle* g, Chandle* x){
      return Cengine_engine->push<cscalar_add_sigmoid_back_op>(nodeof(r),nodeof(g),nodeof(x));
    }


  }
}


#endif
