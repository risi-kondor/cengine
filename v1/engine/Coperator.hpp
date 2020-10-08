#ifndef _Coperator
#define _Coperator


namespace Cengine{

  class Cnode;
  class Batcher; 


  class Coperator{
  public:

    Cnode* owner;
    vector<Cnode*> inputs;


  public:

    Coperator(){}

    Coperator(Cnode* x0){
      inputs.push_back(x0);
    }

    Coperator(Cnode* x0, Cnode* x1){
      inputs.push_back(x0);
      inputs.push_back(x1);
    }

    Coperator(Cnode* x0, Cnode* x1, Cnode* x2){
      inputs.push_back(x0);
      inputs.push_back(x1);
      inputs.push_back(x2);
    }

    Coperator(Cnode* x0, Cnode* x1, Cnode* x2, Cnode* x3){
      inputs.push_back(x0);
      inputs.push_back(x1);
      inputs.push_back(x2);
      inputs.push_back(x3);
    }


  public:

    virtual void exec(){}


  public:

    virtual string str() const{
      return "";
    }

    string inp_str() const;	

    template<typename TYPE1>
    string inp_str(const TYPE1& x1) const;	
    
    template<typename TYPE1, typename TYPE2>
    string inp_str(const TYPE1& x1, const TYPE2& x2) const;	
    
    template<typename TYPE1, typename TYPE2, typename TYPE3>
    string inp_str(const TYPE1& x1, const TYPE2& x2, const TYPE3& x3) const;	
    
  };


  // ---- Subclasses ---------------------------------------------------------------------------------------


  class CumulativeOperator{};


  class InPlaceOperator{};


  class BatchedOperator{
  public: 
    virtual int batcher_id() const=0;
    virtual void set_batcher_id(const int i)=0;
    virtual Batcher* spawn_batcher() const=0;
    virtual string batcher_name() const {return "";}
  };

  class BatcherExecutor{};

}

#endif


    /*
      ostringstream oss;
      oss<<"[";
      for(int i=0; i<inputs.size(); i++){
	//oss<<inputs[i]->ident();
	if(i<inputs.size()-1) oss<<",";
      }
      //if(inputs.size()>0) oss<<inputs[inputs.size()-1]->ident();
      oss<<"]";
      return oss.str();
    }
    */
