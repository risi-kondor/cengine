#ifndef _Chandle
#define _Chandle

#include "Cnode.hpp"

namespace Cengine{

  class Chandle{
  public:

    Cnode* node;

    int id;

    
  public:

    Chandle(Cnode* _node): node(_node){
      node->nhandles++;
    }
    
    ~Chandle(){
      //DEBUG_ENGINE({CoutLock lk; cout<<"    Deleting "<<ident()<<endl;})
      node->engine->dec_handle(node);
    }


  public:

    string ident() const{
      return "H"+to_string(id);
    }

    string str() const{
      return ident()+" ["+node->ident()+"]";
    }

  };


  // ---- Functions -----------------------------------------------------------------------------------------


  inline Cnode* nodeof(Chandle* hdl){
    return hdl->node;
  }

  inline Cnode* nodeof(const Chandle* hdl){
    return hdl->node;
  }

  inline void replace(Chandle*& target, Chandle* hdl){
    delete target;
    target=hdl;
  }




}

#endif
