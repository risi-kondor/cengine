#ifndef _CFtensorHelpers
#define _CFtensorHelpers

#include "Cengine_base.hpp"
#include "Gindex.hpp"
#include "Gdims.hpp"


namespace Cengine{

  class CFtensor;

  class CFtensorProductType{
  public:
    int k;
    vector<pair<int,int> > xout;
    vector<pair<int,int> > yout;
    vector<pair<int,int> > contract;
    vector<tuple<int,int,int> > direct;

  public:

    CFtensorProductType(int _k, vector<pair<int,int> > _xout, vector<pair<int,int>> _yout, 
      vector<pair<int,int> > _contract, vector<tuple<int,int,int> > _direct):
      k(_k),xout(_xout), yout(_yout), contract(_contract), direct(_direct){}


  public:
    
    tuple<int,int,int,int> type() const{
      return {xout.size(),yout.size(),contract.size(),direct.size()};
    }

    bool is_xout(const int a, const int b) const{
      if(xout.size()!=1) return false;
      auto& p=xout[0];
      if(p.first!=a) return false;
      if(p.second!=b) return false;
      return true;
    }

    bool is_yout(const int a, const int b) const{
      if(yout.size()!=1) return false;
      auto& p=yout[0];
      if(p.first!=a) return false;
      if(p.second!=b) return false;
      return true;
    }

    bool is_contr(const int a, const int b) const{
      if(contract.size()!=1) return false;
      auto& p=contract[0];
      if(p.first!=a) return false;
      if(p.second!=b) return false;
      return true;
    }

    inline Gdims dims(const CFtensor& x, const CFtensor& y) const;

    
  public:

    string str() const{
      ostringstream oss;
      oss<<"out: "; 
      for(auto p: xout) oss<<"x"<<p.first<<"->"<<p.second<<" ";
      //oss<<"/ ";
      for(auto p: yout) oss<<"y"<<p.first<<"->"<<p.second<<" ";
      oss<<endl; 
      oss<<"contract: ";
      for(auto p:contract)
	oss<<"(x"<<p.first<<",y"<<p.second<<") ";
      oss<<endl;
      oss<<"direct: ";
      for(auto p: direct)
	oss<<"("<<std::get<0>(p)<<","<<std::get<0>(p)<<")->"<<std::get<2>(p)<<" ";
      oss<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CFtensorProductType& x){
      stream<<x.str(); return stream;}

  };


}

#endif
