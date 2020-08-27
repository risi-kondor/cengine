

template<int selector> 
void add_Mprod(const CtensorBpack& x, const CtensorBpack& y, const int nx=1, const int ny=1){

  const int N=pack.size();
  assert(x.pack.size()==N);
  assert(y.pack.size()==N);

  if(device==0){
    x.to_device(0);
    y.to_device(0);
    for(int i=0; i<N; i++)
      pack[i]->add_Mprod<selector>(*x.pack[i],*y.pack[i]);
    return; 
  }

  x.to_device(1);
  y.to_device(1);

  float alpha0=1.0;
  float alpha1=1.0;
  float alpha2=1.0;
  float alpha3=1.0;
  float beta=1.0;
	
  if (selector==0||selector==3) alpha1=-1.0;
  if (selector==2||selector==3) alpha2=-1.0;
  if (selector==1||selector==3) alpha3=-1.0;

  CUBLAS_SAFE(cublasSgemmbatched(GEnet_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha0,
      y.parr,J,x.parr,K,&beta,parr,J,N);); 
  CUBLAS_SAFE(cublasSgemmBatched(GEnet_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha1,
      y.parrc,J,x.parrc,K,&beta,parr,J,N);); 
  CUBLAS_SAFE(cublasSgemmBatched(GEnet_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha2,
      y.parrc,J,x.parr,K,&beta,parrc,J,N);); 
  CUBLAS_SAFE(cublasSgemmBatched(GEnet_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha3,
      y.parr,J,x.parrc,K,&beta,parrc,J,N);); 

}
