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
#ifndef _Cworker
#define _Cworker

namespace Cengine{

  class Cengine;


  class Cworker{
  public:

    Cengine* owner;
    int id;
    bool killflag=false; 
    bool working=false; 

    thread th;

    Cworker(Cengine* _owner, const int _id): 
      owner(_owner), id(_id), 
      th([this](){this->run();}){}

    ~Cworker(){
      killflag=true; 
      th.join();
    }

	
  public:

    void run();

  };

}

#endif
