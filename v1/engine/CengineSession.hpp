#ifndef _CengineSession
#define _CengineSession

#include "Cengine_base.hpp"

namespace Cengine{

  class CengineSession{
  public:

    CengineSession(){
#ifdef _WITH_CUBLAS
      //#include <cublas_v2.h>
      //cublasHandle_t Cengine_cublas;
      cublasCreate(&Cengine_cublas);
#endif 
    }

    ~CengineSession(){
      delete Cengine_engine;
#ifdef CENGINE_OBJ_COUNT
      cout<<"Cnode objects leaked: "<<Cnode_count<<endl; 
      cout<<"Chandle objects leaked: "<<Chandle_count<<endl; 
      cout<<"Coperator objects leaked: "<<Coperator_count<<endl; 
      cout<<"RscalarB objects leaked: "<<RscalarB_count<<endl; 
      cout<<"CscalarB objects leaked: "<<CscalarB_count<<endl; 
      cout<<"CtensorB objects leaked: "<<CtensorB_count<<endl; 
#endif
    }

  };

}

#endif 


