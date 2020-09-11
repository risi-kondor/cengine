#ifndef _RscalarB
#define _RscalarB

#include "Cengine_base.hpp"
#include "Cnode.hpp"

extern default_random_engine rndGen;


namespace Cengine{

  class RscalarB: public Cobject{
  public:

    int nbu=-1;

    float val;
    float* arr=nullptr;

    mutable int device=0;


    RscalarB(){}

    ~RscalarB(){
      delete[] arr;
    }

    string classname() const{
      return "RscalarB";
    }

    string describe() const{
      if(nbu>=0) return "RscalarB["+to_string(nbu)+"]";
      return "RscalarB";
    }



  public: // ---- Constructors ------------------------------------------------------------------------------


    //RscalarB(const Gscalar<complex<float>>& x):
    //CGscalar(std::move(x)){}


  public: // ---- Filled constructors -----------------------------------------------------------------------


    RscalarB(const float c, const device_id& dev=0): val(c){}

    RscalarB(const fill_raw& fill, const device_id& dev=0){}

    RscalarB(const fill_zero& fill, const device_id& dev=0): val(0){}
 
    RscalarB(const fill_gaussian& fill, const device_id& dev=0){
      normal_distribution<float> distr;
      val=distr(rndGen);
    }


    RscalarB(const int _nbu, const fill_raw& fill, const device_id& dev=0): nbu(_nbu){
      reallocate();
    }

    RscalarB(const int _nbu, const fill_zero& fill, const device_id& dev=0): 
      RscalarB(_nbu,fill::raw){
      if(nbu==-1) val=0; 
      else std::fill(arr,arr+nbu,0);
    }
 
    RscalarB(const int _nbu, const fill_gaussian& fill, const device_id& dev=0):
      RscalarB(_nbu,fill::raw){
      normal_distribution<float> distr;
      if(nbu==-1) val=distr(rndGen);
      else for(int i=0; i<nbu; i++) arr[i]=distr(rndGen);
    }

    RscalarB(const int _nbu, const float c, const device_id& dev=0): 
      RscalarB(_nbu,fill::raw){
      if(nbu==-1) val=c; 
      else std::fill(arr,arr+nbu,c);
    }
 

    void reallocate(){
      delete[] arr;
      if(nbu==-1) return;
      arr=new float[nbu];
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    RscalarB(const RscalarB& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	reallocate();
	std::copy(x.arr,x.arr+nbu,arr);
      }
    }

    RscalarB(RscalarB&& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	arr=x.arr;
	x.arr=nullptr;
      }
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

        
    operator vector<float>(){
      if(nbu==-1){return vector<float>(1,val);}
      vector<float> R(nbu);
      to_device(0);
      for(int i=0; i<nbu; i++)
	R[i]=arr[i];
      return R;
    }


    void to_device(const device_id& _dev) const{
      //CFscalar::to_device(_dev);
    }



  public: // ---- Access -------------------------------------------------------------------------------------


    int getnbu() const{
      return nbu;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void zero(){
      if(nbu==-1) val=0;
      else std::fill(arr, arr+nbu,0);
    }

    RscalarB* conj() const{
      RscalarB* R=new RscalarB(nbu,fill::raw);
      if(nbu==-1) R->val=val;
      else for(int i=0; i<nbu; i++) R->arr[i]=arr[i];
      return R;
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------

    
    void add(const RscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i];
    }

    void add(const RscalarB& x, const float c){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=c*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=c*x.arr[i];
    }

    void subtract(const RscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val-=x.val;
      else for(int i=0; i<nbu; i++) arr[i]-=x.arr[i];
    }

    void add_prod(const RscalarB& x, const RscalarB& y){
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_div(const RscalarB& x, const RscalarB& y){
      if(nbu==-1) val+=x.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/y.arr[i];
    }

    void add_div_back0(const RscalarB& x, const RscalarB& y){
      if(nbu==-1) val+=x.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/y.arr[i];
    }

    void add_div_back1(const RscalarB& g, const RscalarB& x, const RscalarB& y){
      if(nbu==-1) val-=g.val*x.val*pow(y.val,-2.0);
      else for(int i=0; i<nbu; i++) arr[i]-=g.arr[i]*x.arr[i]*pow(y.arr[i],-2.0);
    }


    void add_abs(const RscalarB& x){
      if(nbu==-1) val+=std::abs(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::abs(x.arr[i]);
    }

    void add_abs_back(const RscalarB& g, const RscalarB& x){
      if(nbu==-1) val+=ifthen(x.val>0,g.val,-g.val);
      else for(int i=0; i<nbu; i++) arr[i]+=ifthen(x.arr[i]>0,g.arr[i],-g.arr[i]);
    }

    void add_pow(const RscalarB& x, const float p, const float c=1.0){
      if(nbu==-1) val+=c*std::pow(x.val,p);
      else for(int i=0; i<nbu; i++) arr[i]+=c*std::pow(x.arr[i],p);
    }

    void add_exp(const RscalarB& x){
      if(nbu==-1) val+=std::exp(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::exp(x.arr[i]);
    }

    void add_ReLU(const RscalarB& x, const float c){
      if(nbu==-1){
	val+=ifthen(x.val>0,x.val,c*x.val);
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=ifthen(x.arr[i]>0,x.arr[i],c*x.arr[i]);
	}
      }
    }

    void add_ReLU_back(const RscalarB& g, const RscalarB& x, const float c){
      if(nbu==-1){
	val+=ifthen(x.val>0,g.val,c*g.val);
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=ifthen(x.arr[i]>0,g.arr[i],c*g.arr[i]);
	}
      }
    }

    void add_sigmoid(const RscalarB& x){
      if(nbu==-1){
	val+=1.0/(1.0+std::exp(-x.val));
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=1.0/(1.0+std::exp(-x.arr[i]));
	}
      }
    }

    void add_sigmoid_back(const RscalarB& g, const RscalarB& x){
      if(nbu==-1){
	val+=x.val*(1.0-x.val)*g.val;
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=x.arr[i]*(1.0-x.arr[i])*g.arr[i];
	}
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    RscalarB(const string filename, const device_id& dev=0):
      CFscalar(filename,dev){}

    int save(const string filename) const{
      CFscalar::save(filename);
      return 0;
    }

    RscalarB(Bifstream& ifs): 
      CFscalar(ifs){
    }

    void serialize(Bofstream& ofs){
      CFscalar::serialize(ofs);
    }
    */

    string str(const string indent="") const{
      stringstream oss;
      return oss.str();
    }
   
  };


  //inline RscalarB& asRscalarB(Cobject* x){
  //return downcast<RscalarB>(x,"");
  //}

  inline RscalarB& asRscalarB(Cobject* x, const char* s){
    return downcast<RscalarB>(x,s);
  }

  //inline RscalarB& asRscalarB(Cnode* x){
  //return downcast<RscalarB>(x,"");
  //}

  inline RscalarB& asRscalarB(Cnode* x, const char* s){
    return downcast<RscalarB>(x,s);
  }
  
  //inline RscalarB& asRscalarB(Cnode& x){
  //return downcast<RscalarB>(x,"");
  //}

  inline RscalarB& asRscalarB(Cnode& x, const char* s){
    return downcast<RscalarB>(x,s);
  }
  

}

#endif



  /*
  inline RscalarB& asRscalarB(Cobject* x){
    assert(x); 
    if(!dynamic_cast<RscalarB*>(x))
      cerr<<"Cengine error: Cobject is of type "<<x->classname()<<" instead of RscalarB."<<endl;
    assert(dynamic_cast<RscalarB*>(x));
    return static_cast<RscalarB&>(*x);
  }

  inline RscalarB& asRscalarB(Cnode* x){
    assert(x->obj);
    if(!dynamic_cast<RscalarB*>(x->obj))
      cerr<<"Cengine error: Cobject is of type "<<x->obj->classname()<<" instead of RscalarB."<<endl;
    assert(dynamic_cast<RscalarB*>(x->obj));
    return static_cast<RscalarB&>(*x->obj);
  }

  inline RscalarB& asRscalarB(Cnode& x){
    assert(x.obj);
    if(!dynamic_cast<RscalarB*>(x.obj))
      cerr<<"Cengine error: Cobject is of type "<<x.obj->classname()<<" instead of RscalarB."<<endl;
    assert(dynamic_cast<RscalarB*>(x.obj));
    return static_cast<RscalarB&>(*x.obj);
  }
  */
