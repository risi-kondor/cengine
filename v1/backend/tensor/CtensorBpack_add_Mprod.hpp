

template <int selector>
void add_Mprod(const CtensorBpack& x,
               const CtensorBpack& y,
               const int nx = 1,
               const int ny = 1) {
  const int N = pack.size();
  // assert(x.pack.size()==N);
  // assert(y.pack.size()==N);

  //{CoutLock lk; cout<<"Mprod Dev="<<device<<endl;}

  if (device == 0) {
    x.to_device(0);
    y.to_device(0);
    for (int i = 0; i < N; i++) {
      pack[i]->add_Mprod<selector>(*x.pack[i], *y.pack[i], nx, ny);
    }
    return;
  }

  // x.to_device(1);
  // y.to_device(1);

  const int xk = x.pack[0]->k;
  const int yk = y.pack[0]->k;
  const int k = pack[0]->k;

  const int K = x.pack[0]->combined_size(xk - nx, xk);
  assert(y.pack[0]->combined_size(0, ny) == K);

  const int I = x.pack[0]->combined_size(0, xk - nx);
  const int J = y.pack[0]->combined_size(ny, yk);

  float alpha0 = 1.0;
  float alpha1 = 1.0;
  float alpha2 = 1.0;
  float alpha3 = 1.0;
  float beta = 1.0;

  if (selector == 0 || selector == 3) {
    alpha1 = -1.0;
  }
  if (selector == 2 || selector == 3) {
    alpha2 = -1.0;
  }
  if (selector == 1 || selector == 3) {
    alpha3 = -1.0;
  }

  get_parr();
  x.get_parr();
  y.get_parr();

  // cout<<"Batched Mprod"<<endl;

  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J, I,
                                 K, &alpha0, const_cast<const float**>(y.parr),
                                 J, const_cast<const float**>(x.parr), K, &beta,
                                 parr, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J, I,
                                 K, &alpha1, const_cast<const float**>(y.parrc),
                                 J, const_cast<const float**>(x.parrc), K,
                                 &beta, parr, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J, I,
                                 K, &alpha2, const_cast<const float**>(y.parrc),
                                 J, const_cast<const float**>(x.parr), K, &beta,
                                 parrc, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J, I,
                                 K, &alpha3, const_cast<const float**>(y.parr),
                                 J, const_cast<const float**>(x.parrc), K,
                                 &beta, parrc, J, N));
}

template <int selector>
void add_Mprod_AT(const CtensorBpack& x,
                  const CtensorBpack& y,
                  const int nx = 1,
                  const int ny = 1) {
  const int N = pack.size();
  assert(x.pack.size() == N);
  assert(y.pack.size() == N);

  //{CoutLock lk; cout<<"Mprod_AT Dev="<<device<<endl;}

  if (device == 0) {
    x.to_device(0);
    y.to_device(0);
    for (int i = 0; i < N; i++) {
      pack[i]->add_Mprod_AT<selector>(*x.pack[i], *y.pack[i]);
    }
    return;
  }

  x.to_device(1);
  y.to_device(1);

  const int xk = x.pack[0]->k;
  const int yk = y.pack[0]->k;
  const int k = pack[0]->k;

  const int K = x.pack[0]->combined_size(xk - nx, xk);
  assert(y.pack[0]->combined_size(yk - ny, yk) == K);

  const int I = x.pack[0]->combined_size(0, xk - nx);
  const int J = y.pack[0]->combined_size(0, yk - ny);
  // assert(asize==I*J);

  float alpha0 = 1.0;
  float alpha1 = 1.0;
  float alpha2 = 1.0;
  float alpha3 = 1.0;
  float beta = 1.0;

  if (selector == 0 || selector == 3) {
    alpha1 = -1.0;
  }
  if (selector == 2 || selector == 3) {
    alpha2 = -1.0;
  }
  if (selector == 1 || selector == 3) {
    alpha3 = -1.0;
  }

  get_parr();
  x.get_parr();
  y.get_parr();

  // cout<<"Batched Mprod_AT"<<endl;

  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J, I,
                                 K, &alpha0, const_cast<const float**>(y.parr),
                                 K, const_cast<const float**>(x.parr), K, &beta,
                                 parr, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J, I,
                                 K, &alpha1, const_cast<const float**>(y.parrc),
                                 K, const_cast<const float**>(x.parrc), K,
                                 &beta, parr, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J, I,
                                 K, &alpha2, const_cast<const float**>(y.parrc),
                                 K, const_cast<const float**>(x.parr), K, &beta,
                                 parrc, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J, I,
                                 K, &alpha3, const_cast<const float**>(y.parr),
                                 K, const_cast<const float**>(x.parrc), K,
                                 &beta, parrc, J, N));
}

template <int selector>
void add_Mprod_TA(const CtensorBpack& x,
                  const CtensorBpack& y,
                  const int nx = 1,
                  const int ny = 1) {
  const int N = pack.size();
  assert(x.pack.size() == N);
  assert(y.pack.size() == N);

  //{CoutLock lk; cout<<"Mprod_TA Dev="<<device<<endl;}

  if (device == 0) {
    x.to_device(0);
    y.to_device(0);
    for (int i = 0; i < N; i++) {
      pack[i]->add_Mprod_TA<selector>(*x.pack[i], *y.pack[i]);
    }
    return;
  }

  x.to_device(1);
  y.to_device(1);

  const int xk = x.pack[0]->k;
  const int yk = y.pack[0]->k;
  const int k = pack[0]->k;

  const int K = x.pack[0]->combined_size(0, nx);
  assert(y.pack[0]->combined_size(0, ny) == K);

  const int I = x.pack[0]->combined_size(nx, xk);
  const int J = y.pack[0]->combined_size(ny, yk);
  // assert(asize==I*J);

  float alpha0 = 1.0;
  float alpha1 = 1.0;
  float alpha2 = 1.0;
  float alpha3 = 1.0;
  float beta = 1.0;

  if (selector == 0 || selector == 3) {
    alpha1 = -1.0;
  }
  if (selector == 2 || selector == 3) {
    alpha2 = -1.0;
  }
  if (selector == 1 || selector == 3) {
    alpha3 = -1.0;
  }

  get_parr();
  x.get_parr();
  y.get_parr();

  // cout<<"Batched Mprod_TA"<<endl;

  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_T, J, I,
                                 K, &alpha0, const_cast<const float**>(y.parr),
                                 J, const_cast<const float**>(x.parr), I, &beta,
                                 parr, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_T, J, I,
                                 K, &alpha1, const_cast<const float**>(y.parrc),
                                 J, const_cast<const float**>(x.parrc), I,
                                 &beta, parr, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_T, J, I,
                                 K, &alpha2, const_cast<const float**>(y.parrc),
                                 J, const_cast<const float**>(x.parr), I, &beta,
                                 parrc, J, N));
  CUBLAS_SAFE(cublasSgemmBatched(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_T, J, I,
                                 K, &alpha3, const_cast<const float**>(y.parr),
                                 J, const_cast<const float**>(x.parrc), I,
                                 &beta, parrc, J, N));
}
