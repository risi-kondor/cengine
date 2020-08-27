#ifndef _Cobject
#define _Cobject

#include "Cengine_base.hpp"

namespace Cengine{

  class Cobject{
  public:

    virtual ~Cobject(){}


  public:

    virtual string classname() const=0;
      //{
      //return "CtensorB";
      //}


  };

}

#endif
