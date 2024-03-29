/*
 * This file is part of Cengine, an asynchronous C++/CUDA compute engine. 
 *  
 * Copyright (c) 2020- Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */
#ifndef _CscalarObject
#define _CscalarObject

#include "Cengine_base.hpp"
#include "ExprTemplates.hpp"
#include "RscalarObject.hpp"
#include "CscalarB_ops.hpp"
#include "InterfaceBase.hpp"

//#include "CscalarInterface.hpp"

extern Cengine::Cengine* Cengine_engine;


namespace Cengine{

  class CscalarObject{
  public:

    int nbu=-1; 

    Chandle* hdl;

    ~CscalarObject(){
      delete hdl; 
    }


  public: // ---- Filled constructors ------------------------------------------------------------------------


    CscalarObject(){
      hdl=Cengine_engine->push<new_cscalar_op>(-1,0);
    }

    CscalarObject(Chandle* _hdl): hdl(_hdl){}

    CscalarObject(const fill_raw& fill, const int device=0){
      hdl=Cengine_engine->push<new_cscalar_op>(-1,device);
    }

    CscalarObject(const fill_zero& fill, const int device=0){
      hdl=Cengine_engine->push<new_cscalar_zero_op>(-1,device);
    }

    CscalarObject(const fill_gaussian& fill, const int device=0){
      hdl=Cengine_engine->push<new_cscalar_gaussian_op>(-1,fill.c,device);
    }

    CscalarObject(const complex<float> x, const int device=0){
      hdl=Cengine_engine->push<new_cscalar_set_op>(-1,x,device);
    }

    CscalarObject(const int x){
      hdl=Cengine_engine->push<new_cscalar_set_op>(-1,x,0);
    }

    CscalarObject(const float x){
      hdl=Cengine_engine->push<new_cscalar_set_op>(-1,x);
    }

    CscalarObject(const double x){
      hdl=Cengine_engine->push<new_cscalar_set_op>(-1,x,0);
    }

    CscalarObject(const float x, const int device){
      hdl=Cengine_engine->push<new_cscalar_set_op>(-1,x,device);
    }

    CscalarObject(const int nbd, const fill_raw& fill, const int device=0): nbu(nbd){
      hdl=Cengine_engine->push<new_cscalar_op>(nbd,device);
    }

    CscalarObject(const int nbd, const fill_zero& fill, const int device=0): nbu(nbd){
      hdl=Cengine_engine->push<new_cscalar_zero_op>(nbd,device);
    }

    CscalarObject(const int nbd, const fill_gaussian& fill, const int device=0): nbu(nbd){
      hdl=Cengine_engine->push<new_cscalar_gaussian_op>(nbd,fill.c,device);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    CscalarObject(const CscalarObject& x):
      nbu(x.nbu),
      hdl(Cengine_engine->push<cscalar_copy_op>(x.hdl)){
    }
      
    CscalarObject(CscalarObject&& x){
      nbu=x.nbu; 
      hdl=x.hdl;
      x.hdl=nullptr;
    }

    CscalarObject& operator=(const CscalarObject& x){
      delete hdl;
      nbu=x.nbu; 
      hdl=Cengine_engine->push<cscalar_copy_op>(x.hdl);
      return *this;
    }

    CscalarObject& operator=(CscalarObject&& x){
      delete hdl;
      nbu=x.nbu; 
      hdl=x.hdl;
      x.hdl=nullptr;
      return *this;
    }
    


  public: // ---- Conversions --------------------------------------------------------------------------------


    CscalarObject(const Conjugate<CscalarObject>& x):
      CscalarObject(x.obj.conj()){}


  public: // ---- Access -------------------------------------------------------------------------------------


    int getnbu() const{ 
      return nbu; 
    }

    complex<float> val() const{
      Cengine_engine->flush(hdl->node);
      vector<complex<float> > v=asCscalarB(hdl->node->obj,__PRETTY_FUNCTION__);
      return v[0];
    }

    RscalarObject real() const{
      return Cengine_engine->push<cscalar_get_real_op>(hdl);
    }

    RscalarObject imag() const{
      return Cengine_engine->push<cscalar_get_imag_op>(hdl);
    }

    void add_real_to(RscalarObject& x){
      Chandle* h=Cengine_engine->push<cscalar_get_real_op>(hdl);
      replace(x.hdl,Cengine_engine->push<rscalar_add_op>(x.hdl,h));
      delete h;
    }

    void add_imag_to(RscalarObject& x){
      Chandle* h=Cengine_engine->push<cscalar_get_imag_op>(hdl);
      replace(x.hdl,Cengine_engine->push<rscalar_add_op>(x.hdl,h));
      delete h;
    }

    void set_real(const RscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_set_real_op>(hdl,x.hdl));
    }

    void set_imag(const RscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_set_imag_op>(hdl,x.hdl));
    }

    void flush() const{
      Cengine_engine->flush(hdl->node);
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      replace(hdl,Cengine_engine->push<cscalar_set_zero_op>(hdl));
    }

    void zero(){
      replace(hdl,Cengine_engine->push<cscalar_set_zero_op>(hdl));
    }


  public: // ---- Non-inplace operations ---------------------------------------------------------------------


    CscalarObject conj() const{
      return Cengine_engine->push<cscalar_conj_op>(hdl);
    }

    CscalarObject plus(const CscalarObject& x){
      return CscalarObject(Cengine_engine->push<cscalar_add_op>(hdl,x.hdl));
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_op>(hdl,x.hdl));
    }

    void add_to_real(const RscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_to_real_op>(hdl,x.hdl));
    }

    void add_to_imag(const RscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_to_imag_op>(hdl,x.hdl));
    }

    void add(const CscalarObject& x, const float c){
      replace(hdl,Cengine_engine->push<cscalar_add_times_real_op>(hdl,x.hdl,c));
    }

    void add(const CscalarObject& x, const complex<float> c){
      replace(hdl,Cengine_engine->push<cscalar_add_times_complex_op>(hdl,x.hdl,c));
    }

    void add_conj(const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_conj_op>(hdl,x.hdl));
    }

    //void add_conj(const CscalarObject& x, const CscalarObject& c){
    //replace(hdl,engine::cscalar_add_prodc(hdl,c.hdl,x.hdl));
    //}

    void subtract(const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_subtract_op>(hdl,x.hdl));
    }

    void add_minus(const CscalarObject& x, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_op>(hdl,x.hdl));
      replace(hdl,Cengine_engine->push<cscalar_subtract_op>(hdl,y.hdl));
    }


    void add_prod(const CscalarObject& x, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_prod2_op>(hdl,x.hdl,y.hdl));
    }

    void add_prodc1(const CscalarObject& x, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_prodc_op>(hdl,x.hdl,y.hdl));
    }

    void add_prod(const CscalarObject& x, const RscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_prod_op>(hdl,x.hdl,y.hdl));
    }


    void add_inp(const CscalarObject& x, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_prodc_op>(hdl,x.hdl,y.hdl));
    }

    void add_div(const CscalarObject& x, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_div_op>(hdl,x.hdl,y.hdl));
    }

    void add_div_back0(const CscalarObject& g, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_div_back0_op>(hdl,g.hdl,y.hdl));
    }

    void add_div_back1(const CscalarObject& g, const CscalarObject& x, const CscalarObject& y){
      replace(hdl,Cengine_engine->push<cscalar_add_div_back1_op>(hdl,g.hdl,x.hdl,y.hdl));
    }

    void add_abs(const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_abs_op>(hdl,x.hdl));
    }

    void add_abs_back(const CscalarObject& g, const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_abs_back_op>(hdl,g.hdl,x.hdl));
    }

    void add_norm2_back(const CscalarObject& g, const CscalarObject& x){
      add_prod(g,x);
      add_prodc1(g,x);
    }

    void add_pow(const CscalarObject& x, const float p, const complex<float> c=1.0){
      replace(hdl,Cengine_engine->push<cscalar_add_pow_op>(hdl,x.hdl,p,c));
    }

    void add_exp(const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_exp_op>(hdl,x.hdl));
    }

    void add_ReLU(const CscalarObject& x, const float c=0){
      replace(hdl,Cengine_engine->push<cscalar_add_ReLU_op>(hdl,x.hdl,c));
    }

    void add_ReLU_back(const CscalarObject& g, const CscalarObject& x, const float c=0){
      replace(hdl,Cengine_engine->push<cscalar_add_ReLU_back_op>(hdl,g.hdl,x.hdl,c));
    }

    void add_sigmoid(const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_sigmoid_op>(hdl,x.hdl));
    }

    void add_sigmoid_back(const CscalarObject& g, const CscalarObject& x){
      replace(hdl,Cengine_engine->push<cscalar_add_sigmoid_back_op>(hdl,g.hdl,x.hdl));
    }


  public: // ---- Into operations ----------------------------------------------------------------------------


    void inp_into(const CscalarObject& y, CscalarObject& R) const{
      R.add_inp(*this,y);
    }

    void norm2_into(CscalarObject& R) const{
      R.add_inp(*this,*this);
    }


  public: // ---- Operators ---------------------------------------------------------------------------------


    CscalarObject& operator+=(const CscalarObject& y){
      add(y);
      return *this;
    }

    CscalarObject& operator-=(const CscalarObject& y){
      subtract(y);
      return *this;
    }


    CscalarObject operator+(const CscalarObject& y) const{
      CscalarObject R(*this);
      R.add(y);
      return R;
    }

    CscalarObject operator-(const CscalarObject& y) const{
      CscalarObject R(*this);
      R.subtract(y);
      return R;
    }

    CscalarObject operator*(const CscalarObject& y) const{
      CscalarObject R(nbu,fill::zero);
      R.add_prod(*this,y);
      return R;
    }

    CscalarObject operator/(const CscalarObject& y) const{
      CscalarObject R(nbu,fill::zero);
      R.add_div(*this,y);
      return R;
    }

    CscalarObject operator*(const float c) const{
      CscalarObject R(nbu,fill::zero);
      R.add(*this,c);
      return R;
    }

    CscalarObject operator*(const complex<float> c) const{
      CscalarObject R(nbu,fill::zero);
      R.add(*this,c);
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string classname() const{
      return "Cengine::CscalarObject";
    }

    string str(const string indent="") const{
      vector<complex<float> > R=cscalar_get(hdl);
      ostringstream oss;
      oss<<"[ ";
      for(int i=0; i<R.size(); i++)
	oss<<R[i]<<" ";
      oss<<"]";
      return oss.str();
    }
    

    friend ostream& operator<<(ostream& stream, const CscalarObject& x){
      stream<<x.str(); return stream;}

  };

  
}

#endif


