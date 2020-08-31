#ifndef _CtensorB_add_Mprod_batcher
#define _CtensorB_add_Mprod_batcher

//#include "Cbatcher.hpp"
#include "ctensor_Mprod_signature.hpp"
#include "ctensor_add_Mprod_batcher.hpp"


namespace Cengine{



  class ctensor_add_Mprod_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
    public BatchedOperator{
  public:

    static int _batcher_id;

    ctensor_add_Mprod_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    static string classname(){
      return "ctensor_add_Mprod";
    }
    

  public:

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod<0>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
    }


  public:

    void set_batcher_id(const int i){
      _batcher_id=i;
    }
    
    int batcher_id() const{
      return _batcher_id;
    }

    ctensor_Mprod_signature signature() const{
      return ctensor_Mprod_signature(asCtensorB(inputs[1]).dims,asCtensorB(inputs[2]).dims); 
    }

    Batcher* spawn_batcher() const{
      return new MetaBatcher<ctensor_add_Mprod_op,ctensor_Mprod_signature,ctensor_add_Mprod_batcher>(inputs[0]->engine);
    }

    string str() const{
      return "ctensor_add_Mprod"+inp_str();
    }

  };



}


#endif
