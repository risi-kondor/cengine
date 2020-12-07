#ifndef _ctensorarray_cumulative_ops
#define _ctensorarray_cumulative_ops

#include "CtensorB.hpp"


namespace Cengine{

  /*
  class ctensorarray_add_conj_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_conj_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_conj(CTENSORARRAYB(inputs[1]));
    }

    string str() const{
      return "ctensorarray_add_conj"+inp_str();
    }
    
  };
  */
  

  /*
  class ctensorarray_add_transp_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_transp_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_transp(CTENSORARRAYB(inputs[1]));
    }

    string str() const{
    return "ctensorarray_add_transp"+inp_str();
      }
    
  };
  */
  
  /*
  class ctensorarray_add_herm_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_herm_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_herm(CTENSORARRAYB(inputs[1]));
    }

    string str() const{
      return "ctensorarray_add_herm"+inp_str();
    }
    
  };
  */
  
  /*
  class ctensorarray_add_sum_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_sum_op(Cnode* r, const vector<Cnode*> x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      vector<CFtensor*> v(inputs.size()-1);
      for(int i=0; i<inputs.size()-1; i++) 
	v[i]=&CTENSORARRAYB(inputs[i+1]);
      CTENSORARRAYB(owner).add_sum(v);
    }

    string str() const{
      return "add_ctensorarray_sum"+inp_str();
    }

  };
  */
  
  /*
  class ctensorarray_add_to_slice_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    int ix;
    int offs;

    ctensorarray_add_to_slice_op(Cnode* r, Cnode* x, const int _ix, const int _offs):
      Coperator(r,x), ix(_ix), offs(_offs){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_to_slice(CTENSORARRAYB(inputs[1]),ix,offs);
    }

    string str() const{
      return "ctensorarray_add_to_slice"+inp_str(ix,offs);
    }
    
  };
  */

  /*
  class ctensorarray_add_to_chunk_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    int ix;
    int offs;

    ctensorarray_add_to_chunk_op(Cnode* r, Cnode* x, const int _ix, const int _offs):
      Coperator(r,x), ix(_ix), offs(_offs){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_to_chunk(CTENSORARRAYB(inputs[1]),ix,offs);
    }

    string str() const{
      return "ctensorarray_add_to_chunk"+inp_str(ix,offs);
    }
    
  };
  */

  /*
  class ctensorarray_add_to_slices_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    int ix;

    ctensorarray_add_to_slices_op(Cnode* r, vector<Cnode*> v, const int _ix):
      Coperator(r,v), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      vector<const CFtensor*> v(inputs.size()-1);
      for(int i=0; i<inputs.size()-1; i++) v[i]=&CTENSORARRAYB(inputs[i+1]);
      CTENSORARRAYB(owner).add_to_slices(v,ix);
    }

    string str() const{
      return "ctensorarray_add_to_slices";
    }
    
  };
  */

  /*
  class ctensorarray_add_slice_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    int ix;
    int offs;

    ctensorarray_add_slice_op(Cnode* r, Cnode* x, const int _ix, const int _offs):
      Coperator(r,x), ix(_ix), offs(_offs){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_slice(CTENSORARRAYB(inputs[1]),ix,offs);
    }

    string str() const{
      return "ctensorarray_add_slice"+inp_str(ix,offs);
    }
    
  };
  */

  /*
  class ctensorarray_add_chunk_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    int ix;
    int offs;
    int n; 

    ctensorarray_add_chunk_op(Cnode* r, Cnode* x, const int _ix, const int _offs, const int _n):
      Coperator(r,x), ix(_ix), offs(_offs), n(_n){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_chunk(CTENSORARRAYB(inputs[1]),ix,offs,n);
    }

    string str() const{
      return "ctensorarray_add_chunk"+inp_str(ix,offs,n);
    }
    
  };
  */
  

  // ---- Subtract -------------------------------------------------------------------------------------------


  class ctensorarray_subtract_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_subtract_op(Cnode* r, Cnode* x):
      Coperator(r,x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).subtract(CTENSORARRAYB(inputs[1]));
    }

    string str() const{
      return "ctensorarray_subtract"+inp_str();
    }

  };


  // ---- Products -------------------------------------------------------------------------------------------


  class ctensorarray_add_times_realarray_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c;

    ctensorarray_add_times_realarray_op(Cnode* r, Cnode* A, float _c):
      Coperator(r,A), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      //CTENSORARRAYB(owner).add(CTENSORARRAYB(inputs[1]),c);
    }

    string str() const{
      return "ctensorarray_add_times_realarray"+inp_str();
    }

  };

  
  class ctensorarray_add_times_complexarray_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    complex<float> c;

    ctensorarray_add_times_complexarray_op(Cnode* r, Cnode* A, complex<float> _c):
      Coperator(r,A), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      //CTENSORARRAYB(owner).add(CTENSORARRAYB(inputs[1]),c);
    }

    string str() const{
      return "ctensorarray_add_teims_complex"+inp_str();
    }

  };

  
  /*
  class ctensorarray_add_prod_r_A_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_prod_r_A_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_prod(asRscalarB(inputs[1]),CTENSORARRAYB(inputs[2]));
    }

    int batcher_id() const {return 100;}

    string str() const{
      return "ctensorarray_add_prod_r_A"+inp_str();
    }

  };
  */


  /*
  class ctensor_add_prod_c_A_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensor_add_prod_c_A_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_prod(asCscalarB(inputs[1]),CTENSORARRAYB(inputs[2]));
    }

    string str() const{
      return "ctensor_add_prod_cA"+inp_str();
    }

  };
  */

  /*
  class ctensorarray_add_prod_cc_A_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_prod_cc_A_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).add_prod_cconj(asCscalarB(inputs[1]),CTENSORB(inputs[2]));
    }

    string str() const{
      return "ctensorarray_add_prod_cc_A"+inp_str();
    }

  };
  */

  /*
  class ctensorarray_add_prod_c_Ac_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    ctensorarray_add_prod_c_Ac_op(Cnode* r, Cnode* c, Cnode* A):
      Coperator(r,c,A){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).add_prod_c_times_conj(asCscalarB(inputs[1]),CTENSORB(inputs[2]));
      //owner->computed=true; 
    }

    string str() const{
      return "ctensorarray_add_prod_c_Ac"+inp_str();
    }

  };
  */

  class ctensorarray_add_ReLU_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    ctensorarray_add_ReLU_op(Cnode* r, Cnode* x, float _c):
      Coperator(r,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      //CTENSORARRAYB(owner).add_LeakyReLU(CTENSORARRAYB(inputs[1]),c);
    }

    string str() const{
      return "ctensorarray_add_ReLU"+inp_str();
    }

  };
  

  class ctensorarray_add_ReLU_back_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    float c=0;

    ctensorarray_add_ReLU_back_op(Cnode* r, Cnode* g, Cnode* x, float _c):
      Coperator(r,g,x), c(_c){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      //CTENSORARRAYB(owner).add_LeakyReLU_back(CTENSORARRAYB(inputs[1]),CTENSORARRAYB(inputs[2]),c);
    }

    string str() const{
      return "ctensorarray_add_ReLU_back"+inp_str();
    }
    
  };


  /*
  class ctensorarray_add_element_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    Gindex ix;

    ctensorarray_add_element_op(Cnode* r, Cnode* A, const Gindex& _ix):
      Coperator(r,A), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(inputs[1]).add_element_into(asCscalarB(owner),ix);
    }

    string str() const{
      return "ctensorarray_add_element"+inp_str(ix);
    }

  };


  class ctensorarray_add_to_element_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    Gindex ix;

    ctensorarray_add_to_element_op(Cnode* R, Cnode* a, const Gindex& _ix):
      Coperator(R,a), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORARRAYB(owner).add_to_element(ix,asCscalarB(inputs[1]));
    }

    string str() const{
      return "ctensorarray_add_element"+inp_str(ix);
    }

  };
  */


  

}

#endif 
