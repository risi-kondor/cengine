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

// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The last nx indices of x are contracted with the first ny indices of y
// Selector: x is conjugated if selector is 1 or 3
// Selector: y is conjugated if selector is 2 or 3

template<int selector> 
void add_Mprod(const CFtensor& x, const CFtensor& y, const int nx=1, const int ny=1){

  if(x.asize==0 || y.asize==0) return;

  const int K=x.combined_size(x.k-nx,x.k);
  assert(y.combined_size(0,ny)==K);

  const int I=x.combined_size(0,x.k-nx);
  const int J=y.combined_size(ny,y.k);
  assert(asize==I*J);
  if(asize==0) return;

  //cout<<device<<endl;

  if(device==0){
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

  if(device>0){

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
    cudaDeviceSynchronize(); 
  }

}


// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The last nx indices of x are contracted with the last ny indices of y

template<int selector> 
void add_Mprod_AT(const CFtensor& x, const CFtensor& y, const int nx=1, const int ny=1){

  if(x.asize==0 || y.asize==0) return;

  const int K=x.combined_size(x.k-nx,x.k);
  assert(y.combined_size(y.k-ny,y.k)==K);

  //cout<<dims<<endl; 
  const int I=x.combined_size(0,x.k-nx);
  const int J=y.combined_size(0,y.k-ny);
  assert(asize==I*J);
  if(asize==0) return;

  if(device==0){

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

  if(device>0){

    //cout<<"here"<<endl;
    //cout<<x.device<<endl;
    //cout<<y.device<<endl;

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
    cudaDeviceSynchronize(); 
  }

}


// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The first nx indices of x are contracted with the first ny indices of y

template<int selector>
void add_Mprod_TA(const CFtensor& x, const CFtensor& y, const int nx=1, const int ny=1){
  
  if(x.asize==0 || y.asize==0) return;

  const int K=x.combined_size(0,nx);
  
  //if(y.combined_size(0,ny)!=K){CoutLock lk; cout<<K<<" "<<y.combined_size(0,ny)<<endl;};
  assert(y.combined_size(0,ny)==K);

  const int I=x.combined_size(nx,x.k);
  const int J=y.combined_size(ny,y.k);
  assert(asize==I*J);
  if(asize==0) return;

  if(device==0){

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

  if(device>0){
	
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
    cudaDeviceSynchronize(); 
  }
      
}

