
void add(const CFtensor& x){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i];
    return; 
  }
  const float alpha = 1.0;
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
}


void add_conj(const CFtensor& x){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
    return; 
  }
  const float alpha = 1.0;
  const float malpha = -1.0;
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &malpha, x.arrgc, 1, arrgc, 1));
}


void add_transp(const CFtensor& x, const int n=1) const{
  assert(asize==x.asize);
  const int J=x.combined_size(0,n);
  const int I=x.asize/J;
  if(device==0){
    for(int i=0; i<I; i++)
      for(int j=0; j<J; j++){
	arr[i*J+j]+=x.arr[j*I+i];
	arrc[i*J+j]+=x.arrc[j*I+i];
      }
    return;
  }
  const float alpha = 1.0;
  const float beta = 1.0;
  CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
      &alpha,x.arrg,I,&beta,arrg,J,arrg,J));
  CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
      &alpha,x.arrgc,I,&beta,arrgc,J,arrgc,J));
}


void add_herm(const CFtensor& x, const int n=1) const{
  assert(asize==x.asize);
  const int J=x.combined_size(0,n);
  const int I=x.asize/J;
  if(device==0){
    for(int i=0; i<I; i++)
      for(int j=0; j<J; j++){
	arr[i*J+j]+=x.arr[j*I+i];
	arrc[i*J+j]-=x.arrc[j*I+i];
      }
    return;
  }
  const float alpha = 1.0;
  const float malpha = -1.0;
  const float beta = 1.0;
  CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
      &alpha,x.arrg,I,&beta,arrg,J,arrg,J));
  CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
      &malpha,x.arrgc,I,&beta,arrgc,J,arrgc,J));
}


// ---- With constants -----------------------------------------------------------------------------------


void add(const CFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*c;
    return;
  }
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
}

void add(const CFtensor& x, const complex<float> c){
  assert(asize==x.asize);
  float cr=std::real(c);
  float ci=std::imag(c);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
    return;
  }
  const float mci=-ci; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
}

void add_conj(const CFtensor& x, const complex<float> c){
  assert(asize==x.asize);
  float cr=std::real(c);
  float ci=-std::imag(c);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
    return;
  }
  const float mci=-ci; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrgc, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrg, 1, arrgc, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
}


// ---- Subtract ---------------------------------------------------------------------------------------------


void subtract(const CFtensor& x){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
    return;
  }
  const float c=-1.0; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
}

void subtract(const CFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*c;
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i]*c;
    return; 
  }
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
}

void subtract(const CFtensor& x, const complex<float> c){
  assert(asize==x.asize);
  float cr=std::real(c);
  float ci=std::imag(c);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*cr-x.arrc[i]*ci;
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i]*cr+x.arr[i]*ci;
  }
  const float mcr=-cr; 
  const float mci=-ci; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mcr, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrgc, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrg, 1, arrgc, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mcr, x.arrgc, 1, arrgc, 1));
}

void subtractc(const CFtensor& x, const complex<float> c){
  assert(asize==x.asize);
  float cr=std::real(c);
  float ci=std::imag(c);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*cr+x.arrc[i]*ci;
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i]*cr+x.arr[i]*ci;
    return;
  }
  const float mcr=-cr; 
  const float mci=-ci; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mcr, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mcr, x.arrgc, 1, arrgc, 1));
}


