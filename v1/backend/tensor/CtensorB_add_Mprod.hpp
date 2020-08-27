#ifndef _CtensorB_add_Mprod_batcher
#define _CtensorB_add_Mprod_batcher

#include "Cbatcher.hpp"


namespace Cengine{

  class ctensor_Mprod_signature{
  public:

    Gdims dims1;
    Gdims dims2;

    ctensor_Mprod_signature(const Gdims& _dims1, const Gdims& _dims2): 
      dims1(_dims1), dims2(_dims2){
    }

    bool operator==(const ctensor_Mprod_signature& x) const{
      return (dims1==x.dims1)&&(dims2==x.dims2);}

    string str() const{
      return "("+dims1.str()+dims2.str()+")";}

  };
  

}


namespace std{

  template<>
  struct hash<Cengine::ctensor_Mprod_signature>{
  public:
    size_t operator()(const Cengine::ctensor_Mprod_signature& ix) const{
      size_t t=((hash<Cengine::Gdims>()(ix.dims1)<<1)^hash<Cengine::Gdims>()(ix.dims2)<<1);
      return t;
    }
  };

}


namespace Cengine{


  class ctensor_add_Mprod_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
    public BatchedOperator{
  public:

    ctensor_add_Mprod_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod<0>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    int batcher_id() const {return 0;}

    ctensor_Mprod_signature signature() const{
      return ctensor_Mprod_signature(asCtensorB(inputs[1]).dims,asCtensorB(inputs[2]).dims); 
    }

    string str() const{
      return "ctensor_add_Mprod"+inp_str();
    }

  };



  class CtensorB_add_Mprod_batcher: public Cbatcher{
  public:

    CtensorB_add_Mprod_batcher(const ctensor_Mprod_signature& _signature){}

  public:


  };

}


#endif
