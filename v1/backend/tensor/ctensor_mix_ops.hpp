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
#ifndef _CtensorB_mix_ops
#define _CtensorB_mix_ops

#include "CtensorBpack.hpp"
#include "BatcherA.hpp"
#include "ctensor_Mprod_signature.hpp"


namespace Cengine{

  class cscalar_mix_op: public CumulativeOp3<CscalarB,CtensorB,CscalarB>{
  public:

    using CumulativeOp3::CumulativeOp3;

    void exec(CscalarB& r, const CtensorB& M, const CscalarB& x){
      M.mix_into(r,x);
    }



  public:

    string str() const{
      return "cscalar_mix_op"+inp_str();
    }


  };


  class ctensor_mix_op: public CumulativeOp3<CtensorB,CtensorB,CtensorB>{
  public:

    using CumulativeOp3::CumulativeOp3;

    void exec(CtensorB& r, const CtensorB& M, const CtensorB& x){
      M.mix_into(r,x);
    }



  public:

    string str() const{
      return "ctensor_mix_op"+inp_str();
    }


  };

}

#endif
