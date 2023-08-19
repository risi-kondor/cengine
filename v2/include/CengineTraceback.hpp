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
#ifndef _CengineTraceback
#define _CengineTraceback

#include "Cengine_base.hpp"

namespace Cengine{

  class CengineTraceback{
  public:

    int nmsg=20;

    vector<string> msg;
    int head=0; 

  public:

    CengineTraceback(){
      msg.resize(nmsg);
    }

    void operator()(const string s){
      msg[head]=s;
      head=(head+1)%nmsg;
    }

    void dump(){
      CoutLock lk;
      cout<<endl;
      cout<<"\e[1mCengine traceback:\e[0m"<<endl<<endl;
      for(int i=0; i<nmsg; i++)
	if(msg[(head+i)%nmsg]!="") cout<<"  "<<msg[(head+i)%nmsg]<<endl;
      cout<<endl;
    }

  };

}

#endif
