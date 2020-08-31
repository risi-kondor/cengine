#ifndef _Coperators
#define _Coperators


namespace Cengine{


  template<typename RET, typename OBJ0>
  class Coperator1: public Coperator{
  public:

    Coperator1(Cnode* x0){
      inputs.push_back(x0);
    }

    RET& reuse(const RET& r){
      owner->obj=const_cast<RET*>(&r);
      return dynamic_cast<RET&>(*owner->obj);
    }

    RET& output(){
      return dynamic_cast<RET&>(*owner->obj);
    }

    void exec(){
      exec(dynamic_cast<OBJ0&>(*inputs[0]->obj));
    }

    virtual void exec(const OBJ0& x0)=0;
      
  };



  template<typename RET, typename OBJ0, typename OBJ1>
  class Coperator2: public Coperator{
  public:

    Coperator2(Cnode* x0, Cnode* x1){
      inputs.push_back(x0);
      inputs.push_back(x1);
    }

    RET& reuse(const RET& r){
      owner->obj=const_cast<RET*>(&r);
      return dynamic_cast<RET&>(*owner->obj);
    }

    RET& output(){
      return dynamic_cast<RET&>(*owner->obj);
    }

    void exec(){
      exec(dynamic_cast<OBJ0&>(*inputs[0]->obj), 
	dynamic_cast<OBJ1&>(*inputs[1]->obj)); 
    }

    virtual void exec(const OBJ0& x0, const OBJ1& x1)=0;
      
  };


 
  template<typename RET, typename OBJ0, typename OBJ1, typename OBJ2>
  class Coperator3: public Coperator{
  public:

    Coperator3(Cnode* x0, Cnode* x1, Cnode* x2){
      inputs.push_back(x0);
      inputs.push_back(x1);
      inputs.push_back(x2);
    }

    RET& reuse(const RET& r){
      owner->obj=const_cast<RET*>(&r);
      return dynamic_cast<RET&>(*owner->obj);
    }

    RET& output(){
      return dynamic_cast<RET&>(*owner->obj);
    }

    void exec(){
      exec(dynamic_cast<OBJ0&>(*inputs[0]->obj), 
	dynamic_cast<OBJ1&>(*inputs[1]->obj), 
	dynamic_cast<OBJ2&>(*inputs[2]->obj)); 
    }

    virtual void exec(const OBJ0& x0, const OBJ1& x1, const OBJ2& x2)=0;
      
  };


  // ---- Cumulative operators -------------------------------------------------------------------------------

 
  template<typename OBJ0, typename OBJ1>
  class CumulativeOp2: public Coperator, public InPlaceOperator, public CumulativeOperator{
  public:

    CumulativeOp2(Cnode* x0, Cnode* x1){
      inputs.push_back(x0);
      inputs.push_back(x1);
    }

    OBJ0& output(){
      return dynamic_cast<OBJ0&>(*owner->obj);
    }

    void exec(){
      owner->obj=inputs[0]->obj;
      exec(dynamic_cast<OBJ0&>(*inputs[0]->obj), 
	dynamic_cast<OBJ1&>(*inputs[1]->obj));
    }

    virtual void exec(OBJ0& x0, const OBJ1& x1)=0;
      
  };

 
  template<typename OBJ0, typename OBJ1, typename OBJ2>
  class CumulativeOp3: public Coperator, public InPlaceOperator, public CumulativeOperator{
  public:

    CumulativeOp3(Cnode* x0, Cnode* x1, Cnode* x2){
      inputs.push_back(x0);
      inputs.push_back(x1);
      inputs.push_back(x2);
    }

    OBJ0& output(){
      return dynamic_cast<OBJ0&>(*owner->obj);
    }

    void exec(){
      owner->obj=inputs[0]->obj;
      exec(dynamic_cast<OBJ0&>(*inputs[0]->obj), 
	dynamic_cast<OBJ1&>(*inputs[1]->obj), 
	dynamic_cast<OBJ2&>(*inputs[2]->obj)); 
    }

    virtual void exec(OBJ0& x0, const OBJ1& x1, const OBJ2& x2)=0;
      
  };

 


}

#endif
