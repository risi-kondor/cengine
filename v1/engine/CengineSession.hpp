#ifndef _CengineSession
#define _CengineSession

#include "Cengine_base.hpp"

namespace Cengine{

  class CengineSession{
  public:

    CengineSession(){}

    ~CengineSession(){
      delete Cengine_engine;
#ifdef CENGINE_OBJ_COUNT
      cout<<"CscalarB objects leaked: "<<CscalarB_count<<endl; 
#endif
    }

  };

}

#endif 


