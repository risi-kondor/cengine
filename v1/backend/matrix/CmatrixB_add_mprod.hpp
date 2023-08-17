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

// Simple matrix/matrix multiplication 
// Selector: x is conjugated if selector is 1 or 3
// Selector: y is conjugated if selector is 2 or 3

template<int selector> 
void add_mprod(const CmatrixB& x, const CmatrixB& y, const int nx=1, const int ny=1){

  const int K=x.n1;
  assert(y.n0==K);
  const int I=x.n0;
  const int J=y.n1;
  assert(asize==I*J);

  if(dev==0){
    x.to_device(0);
    y.to_device(0);

    const int istridex=K;
    const int istrider=J;
    const int pstridey=J;
    
    for(int i=0; i<I; i++)
      for(int j=0; j<J; j++){
	float tr=0; 
	float ti=0;
	for(int p=0; p<K; p++){
	  //cout<<i<<" "<<j<<" "<<p<<endl;
	  int qx=i*istridex+p;
	  int qy=p*pstridey+j;
	  float xr=x.arr[qx];
	  float xi=x.arrc[qx];
	  float yr=y.arr[qy];
	  float yi=y.arrc[qy];
	  if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
	  if (selector==1) {tr+=xr*yr+xi*yi; ti+=xr*yi-xi*yr;}
	  if (selector==2) {tr+=xr*yr+xi*yi; ti+=(-xr*yi)+xi*yr;}
	  if (selector==3) {tr+=xr*yr-xi*yi; ti-=xr*yi+xi*yr;}
	}
	int qr=i*istrider+j;
	arr[qr]+=tr;
	arrc[qr]+=ti;
      }
  }

  if(dev>0){

    float alpha0=1.0;
    float alpha1=1.0;
    float alpha2=1.0;
    float alpha3=1.0;
    float beta=1.0;
	
    if (selector==0||selector==3) alpha1=-1.0;
    if (selector==2||selector==3) alpha2=-1.0;
    if (selector==1||selector==3) alpha3=-1.0;
    
    //cout<<"Mprod"<<endl;

    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha0,
	y.arrg,J,x.arrg,K,&beta,arrg,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha1,
	y.arrgc,J,x.arrgc,K,&beta,arrg,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha2,
	y.arrgc,J,x.arrg,K,&beta,arrgc,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha3,
	y.arrg,J,x.arrgc,K,&beta,arrgc,J)); 
  }

}

template<int selector> 
void add_mprod_AT(const CmatrixB& x, const CmatrixB& y, const int nx=1, const int ny=1){

  const int K=x.n1;
  assert(y.n1==K);
  const int I=x.n0;
  const int J=y.n0;
  assert(asize==I*J);

  if(dev==0){

    const int istridex=K;
    const int istrider=J;
    const int jstridey=K;

    for(int i=0; i<I; i++)
      for(int j=0; j<J; j++){
	float tr=0; 
	float ti=0;
	for(int p=0; p<K; p++){
	  int qx=i*istridex+p;
	  int qy=p+j*jstridey;
	  float xr=x.arr[qx];
	  float xi=x.arrc[qx];
	  float yr=y.arr[qy];
	  float yi=y.arrc[qy];
	  if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
	  if (selector==1) {tr+=xr*yr+xi*yi; ti+=xr*yi-xi*yr;}
	  if (selector==2) {tr+=xr*yr+xi*yi; ti+=(-xr*yi)+xi*yr;}
	  if (selector==3) {tr+=xr*yr-xi*yi; ti-=xr*yi+xi*yr;}
	}
	int qr=i*istrider+j;
	arr[qr]+=tr;
	arrc[qr]+=ti;
      }

  }

  if(dev>0){
	
    float alpha0=1.0;
    float alpha1=1.0;
    float alpha2=1.0;
    float alpha3=1.0;
    float beta=1.0;
	
    if (selector==0||selector==3) alpha1=-1.0;
    if (selector==2||selector==3) alpha2=-1.0;
    if (selector==1||selector==3) alpha3=-1.0;

    //cout<<"Mprod_AT"<<endl;

    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha0,
	y.arrg,K,x.arrg,K,&beta,arrg,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha1,
	y.arrgc,K,x.arrgc,K,&beta,arrg,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha2,
	y.arrgc,K,x.arrg,K,&beta,arrgc,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha3,
	y.arrg,K,x.arrgc,K,&beta,arrgc,J)); 
  }

}


// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The first nx indices of x are contracted with the first ny indices of y

template<int selector>
void add_mprod_TA(const CmatrixB& x, const CmatrixB& y, const int nx=1, const int ny=1){
  
  const int K=x.n0;
  assert(y.n0==K);
  const int I=x.n1;
  const int J=y.n1;
  assert(asize==I*J);

  if(dev==0){

    const int istrider=J;
    const int pstridex=I;
    const int pstridey=J;

    for(int i=0; i<I; i++)
      for(int j=0; j<J; j++){
	float tr=0; 
	float ti=0;
	for(int p=0; p<K; p++){
	  int qx=i+p*pstridex;
	  int qy=p*pstridey+j;
	  float xr=x.arr[qx];
	  float xi=x.arrc[qx];
	  float yr=y.arr[qy];
	  float yi=y.arrc[qy];
	  if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
	  if (selector==1) {tr+=xr*yr+xi*yi; ti+=xr*yi-xi*yr;}
	  if (selector==2) {tr+=xr*yr+xi*yi; ti+=(-xr*yi)+xi*yr;}
	  if (selector==3) {tr+=xr*yr-xi*yi; ti-=xr*yi+xi*yr;}
	}
	int qr=i*istrider+j;
	arr[qr]+=tr;
	arrc[qr]+=ti;
      }

  }

  if(dev>0){
	
    float alpha0=1.0;
    float alpha1=1.0;
    float alpha2=1.0;
    float alpha3=1.0;
    float beta=1.0;
	
    if (selector==0||selector==3) alpha1=-1.0;
    if (selector==2||selector==3) alpha2=-1.0;
    if (selector==1||selector==3) alpha3=-1.0;

    //cout<<"Mprod_TA"<<endl;

    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha0,
	y.arrg,J,x.arrg,I,&beta,arrg,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha1,
	y.arrgc,J,x.arrgc,I,&beta,arrg,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha2,
	y.arrgc,J,x.arrg,I,&beta,arrgc,J)); 
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha3,
	y.arrg,J,x.arrgc,I,&beta,arrgc,J)); 
  }
      
}

