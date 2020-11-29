#ifndef _cmatrix_cumulative_ops
#define _cmatrix_cumulative_ops

#include "CmatrixB.hpp"


namespace GEnet{


  class cmatrix_add_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cmatrix_add_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).add(CMATRIXB(inputs[1]));
    }

    string str() const{
      return "cmatrix_add"+inp_str();
    }
    
  };
  

  class cmatrix_subtract_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cmatrix_subtract_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).subtract(CMATRIXB(inputs[1]));
    }

    string str() const{
      return "cmatrix_subtract"+inp_str();
    }
    
  };
  

  class cmatrix_add_conj_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cmatrix_add_conj_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).add_conj(CMATRIXB(inputs[1]));
    }

    string str() const{
      return "cmatrix_add_conj"+inp_str();
    }
    
  };
  

  class cmatrix_add_transp_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cmatrix_add_transp_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).add_transp(CMATRIXB(inputs[1]));
    }

    string str() const{
      return "cmatrix_add_transp"+inp_str();
    }
    
  };
  

  class cmatrix_add_herm_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    cmatrix_add_herm_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CMATRIXB(owner).add_herm(CMATRIXB(inputs[1]));
    }

    string str() const{
      return "cmatrix_add_herm"+inp_str();
    }
    
  };


  template<int Tsel, int Csel>
  class cmatrix_add_mprod_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
    //			      public BatchedOperator, public RbatchedOperator{
  public:

    int I;
    int J;
    int K;

    cmatrix_add_mprod_op(Cnode* R, Cnode* A, Cnode* B, const int _I, const int _J, const int _K): 
      Coperator(R,A,B), I(_I), J(_J), K(_K){} 

    static string classname(){
      if(Tsel==0) return "cmatrix_add_mprod<"+to_string(Csel)+">";
      if(Tsel==1) return "cmatrix_add_mprod_TA<"+to_string(Csel)+">";
      if(Tsel==2) return "cmatrix_add_mprod_AT<"+to_string(Csel)+">";
    }
    
  public:

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CmatrixB& obj=CMATRIXB(owner); 
      if(Tsel==0) obj.add_mprod<Csel>(CMATRIXB(inputs[1]),CMATRIXB(inputs[2]));
      if(Tsel==1) obj.add_mprod_TA<Csel>(CMATRIXB(inputs[1]),CMATRIXB(inputs[2]));
      if(Tsel==2) obj.add_mprod_AT<Csel>(CMATRIXB(inputs[1]),CMATRIXB(inputs[2]));
    }

  };

}


#endif
