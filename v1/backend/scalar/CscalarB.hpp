#ifndef _CscalarB
#define _CscalarB

#include "Cengine_base.hpp"
#include "RscalarB.hpp"


extern default_random_engine rndGen;


namespace Cengine{

  class CscalarB: public Cobject{
  public:

    int nbu=-1;

    complex<float> val;
    complex<float>* arr=nullptr;

    mutable int device=0;


    CscalarB(){
      CSCALARB_CREATE();
    }

    ~CscalarB(){
      delete[] arr;
      CSCALARB_DESTROY();
    }

    string classname() const{
      return "CscalarB";
    }

    string describe() const{
      if(nbu>=0) return "CscalarB["+to_string(nbu)+"]";
      return "CscalarB";
    }



  public: // ---- Constructors ------------------------------------------------------------------------------



  public: // ---- Filled constructors -----------------------------------------------------------------------


    CscalarB(const complex<float> c, const device_id& dev=0): 
      val(c){
      CSCALARB_CREATE();
    }

    CscalarB(const fill_raw& fill, const device_id& dev=0){
      CSCALARB_CREATE();
    }

    CscalarB(const fill_zero& fill, const device_id& dev=0): 
      val(0){
      CSCALARB_CREATE();
    }
 
    CscalarB(const fill_gaussian& fill, const float c, const device_id& dev=0){
      normal_distribution<float> distr;
      val=complex<float>(distr(rndGen)*c,distr(rndGen)*c);
      CSCALARB_CREATE();
    }


    CscalarB(const int _nbu, const fill_raw& fill, const device_id& dev=0): 
      nbu(_nbu){
      reallocate();
      CSCALARB_CREATE();
   }

    CscalarB(const int _nbu, const fill_zero& fill, const device_id& dev=0): 
      CscalarB(_nbu,fill::raw){
      if(nbu==-1) val=0; 
      else std::fill(arr,arr+nbu,0);
    }
 
    CscalarB(const int _nbu, const fill_gaussian& fill, const float c, const device_id& dev=0):
      CscalarB(_nbu,fill::raw){
      normal_distribution<float> distr;
      if(nbu==-1) val=complex<float>(distr(rndGen)*c,distr(rndGen)*c);
      else for(int i=0; i<nbu; i++) arr[i]=complex<float>(distr(rndGen)*c,distr(rndGen))*c;
    }

    CscalarB(const int _nbu, const complex<float> c, const device_id& dev=0): 
      CscalarB(_nbu,fill::raw){
      if(nbu==-1) val=c; 
      else std::fill(arr,arr+nbu,c);
    }
 

    void reallocate(){
      delete[] arr;
      if(nbu==-1) return;
      arr=new complex<float>[nbu];
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CscalarB(const CscalarB& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	reallocate();
	std::copy(x.arr,x.arr+nbu,arr);
      }
      CSCALARB_CREATE();
    }

    CscalarB(CscalarB&& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	arr=x.arr;
	x.arr=nullptr;
      }
      CSCALARB_CREATE();
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    operator vector<complex<float> >(){
      if(nbu==-1){return vector<complex<float> >(1,val);}
      vector<complex<float> > R(nbu);
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

    RscalarB* real() const{
      RscalarB* R=new RscalarB(nbu,fill::raw);
      if(nbu==-1) R->val=val.real();
      else for(int i=0; i<nbu; i++) R->arr[i]=arr[i].real();  
      return R; 
    }

    RscalarB* imag() const{
      RscalarB* R=new RscalarB(nbu,fill::raw);
      if(nbu==-1) R->val=val.imag();
      else for(int i=0; i<nbu; i++) R->arr[i]=arr[i].imag();  
      return R; 
    }

    void set_real(const RscalarB& x){
      if(nbu==-1) val.real(x.val);
      else for(int i=0; i<nbu; i++) arr[i].real(x.arr[i]);
    }

    void set_imag(const RscalarB& x){
      if(nbu==-1) val.imag(x.val);
      else for(int i=0; i<nbu; i++) arr[i].imag(x.arr[i]);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void zero(){
      if(nbu==-1) val=0;
      else std::fill(arr, arr+nbu,0);
    }

    CscalarB* conj() const{
      CscalarB* R=new CscalarB(nbu,fill::raw);
      if(nbu==-1) R->val=std::conj(val);
      else for(int i=0; i<nbu; i++) R->arr[i]=std::conj(arr[i]);
      return R;
    }

    static CscalarB* sum(const vector<CscalarB*> v){
      const int N=v.size();
      if(N==0) return new CscalarB(0);
      const int nbu=v[0]->nbu;
      if(nbu==-1){
	complex<float> s=0;
	for(int i=0; i<N; i++)
	  s+=v[i]->val;
	return new CscalarB(s);
      }else{
	FCG_UNIMPL();
	return new CscalarB(0);
      }
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------

    
    void add(const CscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i];
    }

    void add(const CscalarB& x, const float c){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=c*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=c*x.arr[i];
    }

    void add(const CscalarB& x, const complex<float> c){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=c*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=c*x.arr[i];
    }

    void add_to_real(const RscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=complex<float>(x.val,0);
      else for(int i=0; i<nbu; i++) arr[i]+=complex<float>(x.arr[i],0);
    }

    void add_to_imag(const RscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=complex<float>(0,x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=complex<float>(0,x.arr[i]);
    }

    void add_conj(const CscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=std::conj(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::conj(x.arr[i]);
    }

    void add_sum(const vector<CscalarB*> v){
      const int N=v.size();
      if(N==0) return; 
      const int nbu=v[0]->nbu;
      if(nbu==-1){
	for(int i=0; i<N; i++)
	  val+=v[i]->val;
      }else{
	FCG_UNIMPL();
      }
    }


    void subtract(const CscalarB& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val-=x.val;
      else for(int i=0; i<nbu; i++) arr[i]-=x.arr[i];
    }

    void add_prod(const CscalarB& x, const CscalarB& y){
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_prodc(const CscalarB& x, const CscalarB& y){
      if(nbu==-1) val+=x.val*std::conj(y.val);
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*std::conj(y.arr[i]);
    }

    void add_prodcc(const CscalarB& x, const CscalarB& y){
      if(nbu==-1) val+=std::conj(x.val*y.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::conj(x.arr[i]*y.arr[i]);
    }

    void add_prod(const CscalarB& x, const RscalarB& y){
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_div(const CscalarB& x, const CscalarB& y){
      if(nbu==-1) val+=x.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/y.arr[i];
    }

    void add_div_back0(const CscalarB& x, const CscalarB& y){
      if(nbu==-1) val+=x.val/std::conj(y.val); 
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/std::conj(y.arr[i]);
    }
 
    void add_div_back1(const CscalarB& g, const CscalarB& x, const CscalarB& y){
      if(nbu==-1) val-=g.val*std::conj(x.val*complex<float>(pow(y.val,-2.0)));
      else for(int i=0; i<nbu; i++) arr[i]-=g.arr[i]*std::conj(x.arr[i]*complex<float>(pow(y.arr[i],-2.0)));
    }


    void add_abs(const CscalarB& x){
      if(nbu==-1) val+=std::abs(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::abs(x.arr[i]);
    }

    void add_abs_back(const CscalarB& g, const CscalarB& x){
      if(nbu==-1) val+=complex<float>(ifthen(x.val.real()>0,g.val.real(),-g.val.real()),
	ifthen(x.val.imag()>0,g.val.imag(),-g.val.imag()));
      else for(int i=0; i<nbu; i++){
	float re=g.arr[i].real();
	float im=g.arr[i].imag();
	arr[i]+=complex<float>(ifthen(x.arr[i].real()>0,re,-re),ifthen(x.arr[i].imag()>0,im,-im));
      }
    }

    void add_pow(const CscalarB& x, const float p, const complex<float> c=1.0){
      if(nbu==-1) val+=c*std::pow(x.val,p);
      else for(int i=0; i<nbu; i++) arr[i]+=c*std::pow(x.arr[i],p);
    }

    void add_pow_back(const CscalarB& g, const CscalarB& x, const float p, const complex<float> c=1.0){
      if(nbu==-1) val+=c*g.val*std::conj(std::pow(x.val,p));
      else for(int i=0; i<nbu; i++) arr[i]+=c*g.arr[i]*std::conj(std::pow(x.arr[i],p));
    }

    void add_exp(const CscalarB& x){
      if(nbu==-1) val+=std::exp(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::exp(x.arr[i]);
    }

    void add_ReLU(const CscalarB& x, const float c){
      if(nbu==-1){
	float re=x.val.real();
	float im=x.val.imag();
	val+=complex<float>(ifthen(re>0,re,c*re),ifthen(im>0,im,c*im));
      }
      else{
	for(int i=0; i<nbu; i++){
	  float re=x.arr[i].real();
	  float im=x.arr[i].imag();
	  arr[i]+=complex<float>(ifthen(re>0,re,c*re),ifthen(im>0,im,c*im));
	}
      }
    }

    void add_ReLU_back(const CscalarB& g, const CscalarB& x, const float c){
      if(nbu==-1){
	float re=g.val.real();
	float im=g.val.imag();
	val+=complex<float>(ifthen(x.val.real()>0,re,c*re),ifthen(x.val.imag()>0,im,c*im));
      }
      else{
	for(int i=0; i<nbu; i++){
	  float re=g.arr[i].real();
	  float im=g.arr[i].imag();
	  arr[i]+=complex<float>(ifthen(x.arr[i].real()>0,re,c*re),ifthen(x.arr[i].imag()>0,im,c*im));
	}
      }
    }

    void add_sigmoid(const CscalarB& x){
      if(nbu==-1){
	val+=complex<float>(1.0/(1.0+std::exp(-x.val.real())),1.0/(1.0+std::exp(-x.val.imag())));
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=complex<float>(1.0/(1.0+std::exp(-x.arr[i].real())),1.0/(1.0+std::exp(-x.arr[i].imag())));
	}
      }
    }

    void add_sigmoid_back(const CscalarB& g, const CscalarB& x){
      if(nbu==-1){
	val+=complex<float>(x.val.real()*(1.0-x.val.real())*g.val.real(),x.val.imag()*(1.0-x.val.imag())*g.val.imag());
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=complex<float>(x.arr[i].real()*(1.0-x.arr[i].real())*g.arr[i].real(),
	    x.arr[i].imag()*(1.0-x.arr[i].imag())*g.arr[i].imag());
	}
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    CscalarB(const string filename, const device_id& dev=0):
      CFscalar(filename,dev){}

    int save(const string filename) const{
      CFscalar::save(filename);
      return 0;
    }

    CscalarB(Bifstream& ifs): 
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


  inline CscalarB& asCscalarB(Cobject* x, const char* s){
    return downcast<CscalarB>(x,s);
  }

  inline CscalarB& asCscalarB(Cnode* x, const char* s){
    return downcast<CscalarB>(x,s);
  }
  
  inline CscalarB& asCscalarB(Cnode& x, const char* s){
    return downcast<CscalarB>(x,s);
  }

}


#define CSCALARB(x) asCscalarB(x,__PRETTY_FUNCTION__) 


#endif


   //assert(x->obj);
    //if(!dynamic_cast<CscalarB*>(x->obj))
    //cerr<<"Cengine error: Cobject is of type "<<x->obj->classname()<<" instead of CscalarB."<<endl;
    //assert(dynamic_cast<CscalarB*>(x->obj));
    //return static_cast<CscalarB&>(*x->obj);

    //if(!dynamic_cast<CscalarB*>(x))
    //cerr<<"Cengine error: Cobject is of type "<<x->classname()<<" instead of CscalarB."<<endl;
    //assert(dynamic_cast<CscalarB*>(x));
    //return static_cast<CscalarB&>(*x);
    //assert(x.obj);
    //if(!dynamic_cast<CscalarB*>(x.obj))
    //cerr<<"Cengine error: Cobject is of type "<<x.obj->classname()<<" instead of CscalarB."<<endl;
    //assert(dynamic_cast<CscalarB*>(x.obj));
    //return static_cast<CscalarB&>(*x.obj);
  //inline CscalarB& asCscalarB(Cobject* x){
  //return downcast<CscalarB>(x,"");
  //}

  //inline CscalarB& asCscalarB(Cnode* x){
  //return downcast<CscalarB>(x,"");
  //}

  //inline CscalarB& asCscalarB(Cnode& x){
  //return downcast<CscalarB>(x,"");
  //}

