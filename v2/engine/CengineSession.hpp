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
#ifndef _CengineSession
#define _CengineSession

#include "Cengine_base.hpp"

namespace Cengine{

  class CengineSession{
  public:

    CengineSession(const int nthreads=3, const engine_mode& mode=engine_mode::online){
      cengine=new Cengine(nthreads);
#ifdef _WITH_CUBLAS
      cublasCreate(&Cengine_cublas);
#endif 

      if(mode==engine_mode::biphasic){
	cengine->biphasic=true;
	cengine->hold=true;
      }

    }


    ~CengineSession(){
      delete cengine;
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


