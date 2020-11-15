#ifndef _CengineSession
#define _CengineSession

#include "Cengine_base.hpp"

namespace Cengine{

  class CengineSession{
  public:

    CengineSession(const int nthreads=3, const engine_mode& mode=engine_mode::online){
      Cengine_engine=new Cengine(nthreads);
#ifdef _WITH_CUBLAS
      cublasCreate(&Cengine_cublas);
#endif 

      if(mode==engine_mode::biphasic){
	Cengine_engine->biphasic=true;
	Cengine_engine->hold=true;
      }

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


