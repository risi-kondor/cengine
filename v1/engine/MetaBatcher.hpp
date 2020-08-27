#ifndef _MetaBatcher
#define _MetaBatcher

#include "Cnode.hpp"


namespace Cengine{

  class GenericMetaBatcher{
  public:
    
    virtual ~GenericMetaBatcher(){}

    virtual void push(Coperator* op)=0; 

  };


  template<typename OP, typename SUBINDEX, typename BATCHER>
  class MetaBatcher: public GenericMetaBatcher{
  public:

    unordered_map<SUBINDEX,BATCHER*> subengines;

    MetaBatcher(){
      //cout<<"New MetaBatcher for "<<OP::classname()<<endl;
    }

    virtual ~MetaBatcher(){
      for(auto& p: subengines) delete p.second;
    }


  public:

    void push(Coperator* op){
      
      SUBINDEX ix=dynamic_cast<OP*>(op)->signature();

      auto it=subengines.find(ix);
      if(it!=subengines.end()){
	//it->second->push(op);
	return;
      }

      BATCHER* sub=new BATCHER(ix);
      subengines[ix]=sub;
      //sub->push(op);
    }


  };

}


#endif
