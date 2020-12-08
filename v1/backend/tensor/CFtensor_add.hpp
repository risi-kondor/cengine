
void add(const CFtensor& x){
  assert(asize==x.asize);
  //cout<<device<<x.device<<endl;
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i];
    return; 
  }
  if(device==1){
    const float alpha = 1.0;
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    cudaDeviceSynchronize();
  }
}


void add_conj(const CFtensor& x){
  assert(asize==x.asize);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
    return; 
  }
  if(device==1){
    const float alpha = 1.0;
    const float malpha = -1.0;
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &malpha, x.arrgc, 1, arrgc, 1));
  }
}


void add_transp(const CFtensor& x, const int n=1) const{
  assert(asize==x.asize);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
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
  if(device==1){
    const float alpha = 1.0;
    const float beta = 1.0;
    CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	&alpha,x.arrg,I,&beta,arrg,J,arrg,J));
    CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	&alpha,x.arrgc,I,&beta,arrgc,J,arrgc,J));
  }
}


void add_herm(const CFtensor& x, const int n=1) const{
  assert(asize==x.asize);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
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
  if(device==1){
    const float alpha = 1.0;
    const float malpha = -1.0;
    const float beta = 1.0;
    CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	&alpha,x.arrg,I,&beta,arrg,J,arrg,J));
    CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	&malpha,x.arrgc,I,&beta,arrgc,J,arrgc,J));
  }
}


void add_sum(const vector<CFtensor*> v){
  const int N=v.size();
  if(N==0) return; 
  if(device==0){
    for(int i=0; i<N; i++){
      const CFtensor& o=*v[i];
      assert(o.asize==asize);
      for(int j=0; j<asize; j++){
	arr[j]+=o.arr[j];
	arrc[j]+=o.arrc[j];
      }
    }
    return;
  }
  FCG_UNIMPL();
}


void add_to_slice(const CFtensor& x, const int ix, const int offs){
  assert(k==x.k+1);
  for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
  for(int i=ix; i<x.k; i++) assert(dims[i+1]==x.dims[i]);
  int subsize=x.asize;
  if(ix>0) subsize=x.strides[ix-1];
  int supsize=x.asize/subsize;
  int jstride=asize; 
  if(ix>0) jstride=strides[ix-1];

  if(device==0){
    for(int j=0; j<supsize; j++){
      int toffs=j*jstride+offs*strides[ix];
      for(int i=0; i<subsize; i++){
	arr[toffs+i]+=x.arr[j*subsize+i];
	arrc[toffs+i]+=x.arrc[j*subsize+i];
      }
    }
    return; 
  }
  FCG_UNIMPL();
}


void add_to_slices(const vector<const CFtensor*> v, const int ix){
  assert(v.size()==dims[ix]);
  const CFtensor& x=*v[0];
  assert(k==x.k+1);
  for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
  for(int i=ix; i<x.k; i++) assert(dims[i+1]==x.dims[i]);
  int subsize=x.asize;
  if(ix>0) subsize=x.strides[ix-1];
  int supsize=x.asize/subsize;
  int jstride=asize; 
  if(ix>0) jstride=strides[ix-1];

  if(device==0){
    for(int m=0; m<dims[ix]; m++){
      for(int j=0; j<supsize; j++){
	int toffs=j*jstride+m*strides[ix];
	const CFtensor& x=*v[m];
	for(int i=0; i<subsize; i++){
	  arr[toffs+i]+=x.arr[j*subsize+i];
	  arrc[toffs+i]+=x.arrc[j*subsize+i];
	}
      }
    }
    return; 
  }
  FCG_UNIMPL();
}


void add_to_chunk(const CFtensor& x, const int ix, const int offs){
  assert(k==x.k);
  for(int i=0; i<k; i++) 
    if(i!=ix) assert(dims[i]==x.dims[i]);
    else assert(dims[i]>=x.dims[i]);
  int subsize=x.asize;
  if(ix>0) subsize=x.strides[ix-1];
  int supsize=x.asize/subsize;
  int jstride=asize; 
  if(ix>0) jstride=strides[ix-1];

  if(device==0){
    for(int j=0; j<supsize; j++){
      int toffs=j*jstride+offs*strides[ix];
      //for(int m=0; m<x.dims[ix];; m++){
	for(int i=0; i<subsize; i++){
	  arr[toffs+i]+=x.arr[j*subsize+i];
	  arrc[toffs+i]+=x.arrc[j*subsize+i];
	}
	//toffs+=strides[ix];
	//}
    }
    return; 
  }
  FCG_UNIMPL();
  //const float alpha = 1.0;
  //CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
  //CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
}


void set_chunk(const CFtensor& x, const int ix, const int offs){
  assert(k==x.k);
  for(int i=0; i<k; i++) 
    if(i!=ix) assert(dims[i]==x.dims[i]);
    else assert(dims[i]>=x.dims[i]);
  int subsize=x.asize;
  if(ix>0) subsize=x.strides[ix-1];
  int supsize=x.asize/subsize;
  int jstride=asize; 
  if(ix>0) jstride=strides[ix-1];

  if(device==0){
    for(int j=0; j<supsize; j++){
      int toffs=j*jstride+offs*strides[ix];
      //for(int m=0; m<x.dims[ix];; m++){
	for(int i=0; i<subsize; i++){
	  arr[toffs+i]=x.arr[j*subsize+i];
	  arrc[toffs+i]=x.arrc[j*subsize+i];
	}
	//toffs+=strides[ix];
	//}
    }
    return; 
  }
  FCG_UNIMPL();
  //const float alpha = 1.0;
  //CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
  //CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
}


void add_slice(const CFtensor& x, const int ix, const int offs){
  assert(x.k==k+1);
  for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
  for(int i=ix; i<k; i++) assert(x.dims[i+1]==dims[i]);
  int subsize=asize;
  if(ix>0) subsize=strides[ix-1];
  int supsize=asize/subsize;
  int jstride=x.asize; 
  if(ix>0) jstride=x.strides[ix-1];

  if(device==0){
    for(int j=0; j<supsize; j++){
      int toffs=j*jstride+offs*x.strides[ix];
      for(int i=0; i<subsize; i++){
	arr[j*subsize+i]+=x.arr[toffs+i];
	arrc[j*subsize+i]+=x.arrc[toffs+i];
      }
    }
    return; 
  }
  FCG_UNIMPL();
}


void add_chunk(const CFtensor& x, const int ix, const int offs, const int n){
  assert(k==x.k);
  for(int i=0; i<k; i++) 
    if(i!=ix) assert(dims[i]==x.dims[i]);
    else assert(x.dims[i]>=dims[i]);
  int subsize=strides[ix];
  int supsize=x.asize/(strides[ix]*dims[ix]);
  int jstride=asize; 
  if(ix>0) jstride=strides[ix-1];
  int jxstride=x.asize;
  if(ix>0) jxstride=x.strides[ix-1];

  if(device==0){
    for(int j=0; j<supsize; j++){
      for(int m=0; m<n; m++){
	for(int i=0; i<subsize; i++){
	  arr[j*jstride+m*strides[ix]+i]+=x.arr[j*jxstride+(m+offs)*x.strides[ix]+i];
	  arrc[j*jstride+m*strides[ix]+i]+=x.arrc[j*jxstride+(m+offs)*x.strides[ix]+i];
	}
      }
    }
    return; 
  }
  FCG_UNIMPL();
}



// ---- With constants -----------------------------------------------------------------------------------


void add(const CFtensor& x, const float c){
  assert(asize==x.asize);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*c;
    return;
  }
  if(device==1){
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
    cudaDeviceSynchronize();
  }
}

void add(const CFtensor& x, const complex<float> c){
  assert(asize==x.asize);
  float cr=std::real(c);
  float ci=std::imag(c);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
    return;
  }
  if(device==1){
    const float mci=-ci; 
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
    cudaDeviceSynchronize();
  }
}

void add_conj(const CFtensor& x, const complex<float> c){
  assert(asize==x.asize);
  float cr=std::real(c);
  float ci=-std::imag(c);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
    for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
    return;
  }
  if(device==1){
    const float mci=-ci; 
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &ci, x.arrgc, 1, arrg, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mci, x.arrg, 1, arrgc, 1));
    CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
    cudaDeviceSynchronize();
  }
}


// ---- Subtract ---------------------------------------------------------------------------------------------


void subtract(const CFtensor& x){
  assert(asize==x.asize);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
    for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
    return;
  }
  const float c=-1.0; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
  cudaDeviceSynchronize();
}

void subtract(const CFtensor& x, const float c){
  assert(asize==x.asize);
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
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
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
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
  if(device!=1 || x.device!=1){
    to_device(0);
    x.to_device(0);
  }
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


// ---- Normalization ----------------------------------------------------------------------------------------


void add_col_norms(const CFtensor& x){

  int xk=x.dims.size();
  assert(xk>=2);
  const int J=x.dims[xk-1];
  const int I=x.dims[xk-2];
  const int A=x.asize/(I*J);
  assert(asize==A*J);

  if(device==0){
    for(int a=0; a<A; a++){
      int offs=a*I*J;
      for(int j=0; j<J; j++){
	float t=0;
	for(int i=0; i<I; i++){
	  t+=x.arr[offs+i*J+j]*x.arr[offs+i*J+j]+x.arrc[offs+i*J+j]*x.arrc[offs+i*J+j];
	}
	arr[a*J+j]+=sqrt(t);
      }
    }
    return;
  }

  FCG_UNIMPL();

}


void add_col_norms_back(const CFtensor& G, const CFtensor& X, const CFtensor& N){
  assert(k>=2);
  assert(X.asize==asize);
  const int J=dims[k-1];
  const int I=dims[k-2];
  const int A=asize/(I*J);
  assert(N.asize==asize/I);
  assert(G.asize==N.asize);
  if(device==0){
    for(int a=0; a<A; a++){
      int offs=a*I*J;
      for(int j=0; j<J; j++){
	float z=G.arr[a*J+j]/N.arr[a*J+j];
	for(int i=0; i<I; i++){
	  arr[offs+i*J+j]+=X.arr[offs+i*J+j]*z;
	  arrc[offs+i*J+j]+=X.arrc[offs+i*J+j]*z;
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}


void add_divide_cols(const CFtensor& X, const CFtensor& N){
  assert(k>=2);
  assert(X.asize==asize);
  const int J=dims[k-1];
  const int I=dims[k-2];
  const int A=asize/(I*J);
  assert(N.asize==asize/I);
  if(device==0){
    for(int a=0; a<A; a++){
      int offs=a*I*J;
      for(int j=0; j<J; j++){
	//complex<float> z=complex<float>(N.arr[a*J+j],N.arrc[a*J+j]);
	float z=N.arr[a*J+j];
	for(int i=0; i<I; i++){
	  //complex<float> u=complex<float>()
	  arr[offs+i*J+j]+=X.arr[offs+i*J+j]/z;
	  arrc[offs+i*J+j]+=X.arrc[offs+i*J+j]/z;
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}


void add_divide_cols_back0(const CFtensor& G, const CFtensor& N){
  assert(k>=2);
  const int J=dims[k-1];
  const int I=dims[k-2];
  const int A=asize/(I*J);
  assert(N.asize==asize/I);
  assert(G.asize==asize);
  if(device==0){
    for(int a=0; a<A; a++){
      int offs=a*I*J;
      for(int j=0; j<J; j++){
	complex<float> n(N.arr[a*J+j],0); //-N.arrc[a*J+j]);
	//complex<float> z=complex<float>(1,0)/n/n; //complex<float>(G.arr[a*J+j],G.arrc[a*J+j])/n/n;
	for(int i=0; i<I; i++){
	  complex<float> u=complex<float>(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])/n;
	  //complex<float> u=z*complex<float>(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])*
	  //complex<float>(X.arr[offs+i*J+j],-X.arrc[offs+i*J+j]);
	  arr[offs+i*J+j]+=std::real(u);
	  arrc[offs+i*J+j]+=std::imag(u);
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}


void add_divide_cols_back1(const CFtensor& G, const CFtensor& X, const CFtensor& N){
  const int _k=G.k;
  assert(_k>=2);
  assert(G.dims==X.dims);
  assert(dims==N.dims);
  const int J=G.dims[_k-1];
  const int I=G.dims[_k-2];
  const int A=G.asize/(I*J);
  assert(N.asize==G.asize/I);
  assert(asize==N.asize);
  if(device==0){
    for(int a=0; a<A; a++){
      int offs=a*I*J;
      for(int j=0; j<J; j++){
	float z=-pow(N.arr[a*J+j],-2);
	for(int i=0; i<I; i++){ // improve
 	  complex<float> t=complex<float>(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])*
	    complex<float>(X.arr[offs+i*J+j],-X.arrc[offs+i*J+j])*z;
	  arr[a*J+j]+=std::real(t);
	  arrc[a*J+j]+=std::imag(t);
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}



