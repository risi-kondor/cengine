#ifndef _CtensorB_cumulative_ops
#define _CtensorB_cumulative_ops

#include "CtensorB.hpp"


namespace Cengine{


  class ctensor_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      // inputs[0]->is_view=true; taken care of in engine 
      asCtensorB(owner).add(asCtensorB(inputs[1]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add"+inp_str();
      //return "ctensor_add("+inputs[0]->ident()+","+inputs[1]->ident()+")";
    }
    
  };
  

  class ctensor_add_conj_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_conj_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_conj(asCtensorB(inputs[1]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_conj"+inp_str();
      //return "ctensor_add_conj("+inputs[0]->ident()+","+inputs[1]->ident()+")";
    }
    
  };
  

  class ctensor_add_transp_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_transp_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_transp(asCtensorB(inputs[1]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_transp"+inp_str();
      //return "ctensor_add_transp("+inputs[0]->ident()+","+inputs[1]->ident()+")";
    }
    
  };
  

  class ctensor_add_herm_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_herm_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_herm(asCtensorB(inputs[1]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_herm"+inp_str();
      //return "ctensor_add_herm("+inputs[0]->ident()+","+inputs[1]->ident()+")";
    }
    
  };
  

  // ---- Subtract -------------------------------------------------------------------------------------------


  class ctensor_subtract_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_subtract_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).subtract(asCtensorB(inputs[1]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_subtract"+inp_str();
    }

  };


  // ---- Products -------------------------------------------------------------------------------------------


  class ctensor_add_times_real_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c;

    ctensor_add_times_real_op(Cnode* r, Cnode* A, float _c):
      Coperator(r,A), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add(asCtensorB(inputs[1]),c);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_times_real"+inp_str();
    }

  };

  
  class ctensor_add_times_complex_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    complex<float> c;

    ctensor_add_times_complex_op(Cnode* r, Cnode* A, complex<float> _c):
      Coperator(r,A), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add(asCtensorB(inputs[1]),c);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_teims_complex"+inp_str();
    }

  };

  
  class ctensor_add_prod_rA_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_prod_rA_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_prod(asRscalarB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    int batcher_id() const {return 100;}

    string str() const{
      return "ctensor_add_prod_rA"+inp_str();
    }

  };

  
  class ctensor_add_prod_cA_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_prod_cA_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_prod(asCscalarB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_prod_cA"+inp_str();
    }

  };

  
  class ctensor_add_prod_cc_A_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_prod_cc_A_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_prod_cconj(asCscalarB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_prod_cc_A"+inp_str();
    }

  };

  
  class ctensor_add_prod_c_Ac_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_prod_c_Ac_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_prod_c_times_conj(asCscalarB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_prod_c_Ac"+inp_str();
    }

  };


  /*
  class ctensor_add_Mprod_AT_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_Mprod_AT_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod_AT<0>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_Mprod_AT"+inp_str();
    }

  };
  */

  /*
  class ctensor_add_Mprod_TA_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_Mprod_TA_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod_TA<0>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_Mprod_TA"+inp_str();
    }

  };
  */

  /*
  class ctensor_add_Mprod_AC_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:
  
    ctensor_add_Mprod_AC_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod<2>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_Mprod_AC"+inp_str();
    }

  };
  */

  /*
  class ctensor_add_Mprod_TC_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_Mprod_TC_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod_TA<2>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_Mprod_TC"+inp_str();
    }

  };
  */

  /*
  class ctensor_add_Mprod_AH_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_Mprod_AH_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod_AT<2>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
    }

    string str() const{
      return "ctensor_add_Mprod_AH"+inp_str();
    }

  };
  */

  /*
  class ctensor_add_Mprod_HA_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_Mprod_HA_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_Mprod_TA<1>(asCtensorB(inputs[1]),asCtensorB(inputs[2]));
    }

    string str() const{
      return "ctensor_add_Mprod_HA"+inp_str();
    }

  };
  */








  class ctensor_add_ReLU_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    ctensor_add_ReLU_op(Cnode* r, Cnode* x, float _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_LeakyReLU(asCtensorB(inputs[1]),c);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_ReLU"+inp_str();
    }

  };
  

  class ctensor_add_ReLU_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    ctensor_add_ReLU_back_op(Cnode* r, Cnode* g, Cnode* x, float _c):
      Coperator(r,g,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(owner).add_LeakyReLU_back(asCtensorB(inputs[1]),asCtensorB(inputs[2]),c);
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_ReLU_back"+inp_str();
    }
    
  };


  class ctensor_add_inp_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_inp_op(Cnode* R, Cnode* A, Cnode* B):
      Coperator(R,A,B){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asCtensorB(inputs[1]).add_inp_into(asCscalarB(owner),asCtensorB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensor_add_inp"+inp_str();
    }

  };



  

}

#endif 
