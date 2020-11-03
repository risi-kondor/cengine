#ifndef _CtensorB_add_Mprod_ops
#define _CtensorB_add_Mprod_ops

#include "CtensorBpack.hpp"
#include "BatcherA.hpp"
#include "ctensor_Mprod_signature.hpp"


namespace Cengine{


  template<int Tsel, int Csel>
  class ctensor_add_Mprod_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
    public BatchedOperator{
  public:

    Gdims dims1;
    Gdims dims2;

    static int _batcher_id;

    //ctensor_add_Mprod_op(Cnode* R, Cnode* A, Cnode* B):
    //Coperator(R,A,B){}

    ctensor_add_Mprod_op(Cnode* R, Cnode* A, Cnode* B, const Gdims& _dims1, const Gdims& _dims2):
      Coperator(R,A,B), dims1(_dims1), dims2(_dims2){}

    void set_batcher_id(const int i){_batcher_id=i;}

    int batcher_id() const{return _batcher_id;}

    static string classname(){
      if(Tsel==0) return "ctensor_add_Mprod<"+to_string(Csel)+">";
      if(Tsel==1) return "ctensor_add_Mprod_TA<"+to_string(Csel)+">";
      if(Tsel==2) return "ctensor_add_Mprod_AT<"+to_string(Csel)+">";
    }
    
    string batcher_name() const{
      if(Tsel==0) return "ctensor_add_Mprod<"+to_string(Csel)+">"+signature().str();
      if(Tsel==1) return "ctensor_add_Mprod_TA<"+to_string(Csel)+">"+signature().str();
      if(Tsel==2) return "ctensor_add_Mprod_AT<"+to_string(Csel)+">"+signature().str();
    }
    

  public:

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CtensorB& obj=asCtensorB(owner,__PRETTY_FUNCTION__); 
      if(Tsel==0) obj.add_Mprod<Csel>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
      if(Tsel==1) obj.add_Mprod_TA<Csel>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
      if(Tsel==2) obj.add_Mprod_AT<Csel>(asCtensorB(inputs[1],__PRETTY_FUNCTION__),asCtensorB(inputs[2],__PRETTY_FUNCTION__));
    }


    virtual void batched_exec(const vector<Cnode*>& nodes ){
      DEBUG_ENGINE({CoutLock lk; cout<<"    Running batched ctensor_add_Mprod..."<<endl;});
      assert(nodes.size()>0);
      BasicCnodeEngine* engine=nodes[0]->engine;
      const int N=nodes.size();

      CtensorBpack R(nodes,0);
      CtensorBpack X(nodes,1);
      CtensorBpack Y(nodes,2);

      //int dev=R.device;
      //R.to_device(0);
      //X.to_device(dev);
      //Y.to_device(dev);
      
      if(Tsel==0) R.add_Mprod<Csel>(X,Y);
      if(Tsel==1) R.add_Mprod_TA<Csel>(X,Y);
      if(Tsel==2) R.add_Mprod_AT<Csel>(X,Y);

      for(int i=0; i<N; i++)
      nodes[i]->op->owner->obj=R.pack[i];

      for(int i=0; i<N; i++){
	engine->done(nodes[i]);
      }

      DEBUG_ENGINE({CoutLock lk; cout<<"    \e[1mDone.\e[0m"<<endl;});
    }

    ctensor_Mprod_signature signature() const{
      return ctensor_Mprod_signature(dims1,dims2);
    }

    Batcher* spawn_batcher() const{
      return new MetaBatcher<ctensor_add_Mprod_op,ctensor_Mprod_signature,BatcherA<ctensor_add_Mprod_op<Tsel,Csel> > >(inputs[0]->engine);
    }


  public:

    string str() const{
      return "ctensor_add_Mprod"+inp_str();
    }

  };

}


#endif
