#ifndef _GenericOperators
#define _GenericOperators

namespace Cengine{


  template<typename RTYPE, typename XTYPE, typename CTYPE>
  class add_x_times_const_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    CTYPE c;

    add_x_times_const_op(Cnode* r, Cnode* x, CTYPE _c):
      Coperator(r,x), c(_c){}
    
    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      downcast<RTYPE>(owner,__PRETTY_FUNCTION__).
	add(downcast<XTYPE>(inputs[1],__PRETTY_FUNCTION__),c);
    }

    string str() const{
      return "add_x_times_const"+inp_str();
    }

  };


  // ---- add_x_times_y --------------------------------------------------------------------------------------


  template<typename RTYPE, typename XTYPE, typename YTYPE>
  class add_x_times_y_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    add_x_times_y_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}
    
    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      downcast<RTYPE>(owner,__PRETTY_FUNCTION__).
	add(downcast<XTYPE>(inputs[1],__PRETTY_FUNCTION__),downcast<YTYPE>(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "add_x_times_y"+inp_str();
    }

  };


  template<typename RTYPE, typename XTYPE, typename YTYPE>
  class add_x_times_yc_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    add_x_times_yc_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}
    
    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      downcast<RTYPE>(owner,__PRETTY_FUNCTION__).
	add_x_times_yc(downcast<XTYPE>(inputs[1],__PRETTY_FUNCTION__),downcast<YTYPE>(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "add_x_times_yc"+inp_str();
    }

  };


  template<typename RTYPE, typename XTYPE, typename YTYPE>
  class add_xc_times_y_op: public Coperator, public CumulativeOperator, public InPlaceOperator{
  public:

    add_xc_times_y_op(Cnode* r, Cnode* x, Cnode* y):
      Coperator(r,x,y){}
    
    void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      downcast<RTYPE>(owner,__PRETTY_FUNCTION__).
	add_xc_times_y(downcast<XTYPE>(inputs[1],__PRETTY_FUNCTION__),downcast<YTYPE>(inputs[2],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "add_xc_times_y"+inp_str();
    }

  };


}


#endif 