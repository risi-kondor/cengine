
void add(const RFtensor& x){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    return; 
  }
  const float alpha = 1.0;
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
}


void add_transp(const RFtensor& x, const int n=1) const{
  assert(asize==x.asize);
  const int J=x.combined_size(0,n);
  const int I=x.asize/J;
  if(device==0){
    for(int i=0; i<I; i++)
      for(int j=0; j<J; j++){
	arr[i*J+j]+=x.arr[j*I+i];
      }
    return;
  }
  const float alpha = 1.0;
  const float beta = 1.0;
  CUBLAS_SAFE(cublasSgeam(Cengine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
      &alpha,x.arrg,I,&beta,arrg,J,arrg,J));
}


void add_to_slice(const RFtensor& x, const int ix, const int offs){
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
      }
    }
    return; 
  }
  FCG_UNIMPL();
}


void add_to_slices(const vector<const RFtensor*> v, const int ix){
  assert(v.size()==dims[ix]);
  const RFtensor& x=*v[0];
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
	const RFtensor& x=*v[m];
	for(int i=0; i<subsize; i++){
	  arr[toffs+i]+=x.arr[j*subsize+i];
	}
      }
    }
    return; 
  }
  FCG_UNIMPL();
}


void add_to_chunk(const RFtensor& x, const int ix, const int offs){
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


void add_slice(const RFtensor& x, const int ix, const int offs){
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
      }
    }
    return; 
  }
  FCG_UNIMPL();
}


void add_chunk(const RFtensor& x, const int ix, const int offs, const int n){
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
	}
      }
    }
    return; 
  }
  FCG_UNIMPL();
}



// ---- With constants -----------------------------------------------------------------------------------


void add(const RFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    return;
  }
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
}

void add(const RFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    return;
  }
  const float mci=-ci; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
}

void add_conj(const RFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    return;
  }
  const float mci=-ci; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
}


// ---- Subtract ---------------------------------------------------------------------------------------------


void subtract(const RFtensor& x){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
    return;
  }
  const float c=-1.0; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
}

void subtract(const RFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*c;
    return; 
  }
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &c, x.arrg, 1, arrg, 1));
}

void subtract(const RFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*c;
  }
  const float mc=-c; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mc, x.arrg, 1, arrg, 1));
}

void subtractc(const RFtensor& x, const float c){
  assert(asize==x.asize);
  if(device==0){
    for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*c;
    return;
  }
  const float mc=-c; 
  CUBLAS_SAFE(cublasSaxpy(Cengine_cublas, asize, &mc, x.arrg, 1, arrg, 1));
}


// ---- Normalization ----------------------------------------------------------------------------------------


void add_col_norms(const RFtensor& x){

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
	  t+=x.arr[offs+i*J+j]*x.arr[offs+i*J+j];
	}
	arr[a*J+j]+=sqrt(t);
      }
    }
    return;
  }

  FCG_UNIMPL();

}


void add_col_norms_back(const RFtensor& G, const RFtensor& X, const RFtensor& N){
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
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}


void add_divide_cols(const RFtensor& X, const RFtensor& N){
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
	//float z=float(N.arr[a*J+j],N.arrc[a*J+j]);
	float z=N.arr[a*J+j];
	for(int i=0; i<I; i++){
	  //float u=float()
	  arr[offs+i*J+j]+=X.arr[offs+i*J+j]/z;
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}


void add_divide_cols_back0(const RFtensor& G, const RFtensor& N){
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
	float n(N.arr[a*J+j],0); //-N.arrc[a*J+j]);
	//float z=float(1,0)/n/n; //float(G.arr[a*J+j],G.arrc[a*J+j])/n/n;
	for(int i=0; i<I; i++){
	  float u=G.arr[offs+i*J+j]/n;
	  //float u=z*float(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])*
	  //float(X.arr[offs+i*J+j],-X.arrc[offs+i*J+j]);
	  arr[offs+i*J+j]+=u;
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}


void add_divide_cols_back1(const RFtensor& G, const RFtensor& X, const RFtensor& N){
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
 	  float t=G.arr[offs+i*J+j]*X.arr[offs+i*J+j]*z;
	  arr[a*J+j]+=t;
	}
      }    
    }
  }else{
    FCG_UNIMPL(); 
  }
}



