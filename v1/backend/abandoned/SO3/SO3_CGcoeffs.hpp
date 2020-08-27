#ifndef _SO3_CGcoeffs
#define _SO3_CGcoeffs

#include "SO3_CGindex.hpp" 
#include "SO3_CGcoeffs.hpp" 
#include "SO3_CGbank.hpp" 
#include "Gtensor.hpp"

extern default_random_engine rndGen;


namespace GEnet{

  template<class TYPE>
  class SO3_CGcoeffs: public Gtensor<TYPE>{ 
  public:

    using Gtensor<TYPE>::arr; 
    using Gtensor<TYPE>::arrg; 

    int l,l1,l2;

    SO3_CGcoeffs(const CGindex& ix):
      Gtensor<TYPE>({2*ix.l1+1,2*ix.l2+1},fill::zero,0), 
      l(ix.l), l1(ix.l1), l2(ix.l2){
      for(int m1=-l1; m1<=l1; m1++)
	for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
	  this->element(m1,m2)=slowCG(m1,m2);
    }

    SO3_CGcoeffs(Gtensor<TYPE>&& T, const int _l, const int _l1, const int _l2): 
      Gtensor<TYPE>(std::move(T)), l(_l), l1(_l1), l2(_l2){}
    
    ~SO3_CGcoeffs(){
    }
    

  public:
    
    SO3_CGcoeffs(const SO3_CGcoeffs<TYPE>&  x): 
      Gtensor<TYPE>(x,nowarn), l(x.l), l1(x.l1), l2(x.l2){};

    SO3_CGcoeffs& operator=(const SO3_CGcoeffs<TYPE>& x)=delete;
    
  public:

    TYPE& element(const int m1, const int m2){
      return (*this)(m1+l1,m2+l2);
    }


  private:

    TYPE logfact(int n){
      return lgamma(n+1);
    }
    
    TYPE plusminus(int k){ if(k%2==1) return -1; else return +1; }

    TYPE slowCG(const int m1, const int m2){
      
      int m=m1+m2;
      int m3=-m;
      int t1=l2-m1-l;
      int t2=l1+m2-l;
      int t3=l1+l2-l;
      int t4=l1-m1;
      int t5=l2+m2;
  
      int tmin=std::max(0,std::max(t1,t2));
      int tmax=std::min(t3,std::min(t4,t5));

      TYPE logA=(logfact(l1+l2-l)+logfact(l1-l2+l)+logfact(-l1+l2+l)-logfact(l1+l2+l+1))/2;
      logA+=(logfact(l1+m1)+logfact(l1-m1)+logfact(l2+m2)+logfact(l2-m2)+logfact(l+m3)+logfact(l-m3))/2;

      TYPE wigner=0;
      for(int t=tmin; t<=tmax; t++){
	TYPE logB=logfact(t)+logfact(t-t1)+logfact(t-t2)+logfact(t3-t)+logfact(t4-t)+logfact(t5-t);
	wigner += plusminus(t)*exp(logA-logB);
      }
      
      return plusminus(l1-l2-m3)*plusminus(l1-l2+m)*sqrt((TYPE)(2*l+1))*wigner; 
    }

  };

  
} 


#endif
