// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The last nx indices of x are contracted with the first ny indices of y
// Selector: x is conjugated if selector is 1 or 3
// Selector: y is conjugated if selector is 2 or 3

void add_Mprod(const RFtensor& x,
               const RFtensor& y,
               const int nx = 1,
               const int ny = 1) {
  const int K = x.combined_size(x.k - nx, x.k);
  assert(y.combined_size(0, ny) == K);

  // cout<<dims<<endl;
  const int I = x.combined_size(0, x.k - nx);
  const int J = y.combined_size(ny, y.k);
  assert(asize == I * J);

  if (device == 0) {
    const int istridex = K;
    const int istrider = J;
    const int pstridey = J;

    for (int i = 0; i < I; i++) {
      for (int j = 0; j < J; j++) {
        float tr = 0;
        for (int p = 0; p < K; p++) {
          int qx = i * istridex + p;
          int qy = p * pstridey + j;
          float xr = x.arr[qx];
          float yr = y.arr[qy];
          tr += xr * yr;
        }
        int qr = i * istrider + j;
        arr[qr] += tr;
      }
    }
  }

  if (device > 0) {
    float alpha0 = 1.0;
    float beta = 1.0;
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J, I, K,
                            &alpha0, y.arrg, J, x.arrg, K, &beta, arrg, J));
  }
}

// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The last nx indices of x are contracted with the last ny indices of y

void add_Mprod_AT(const RFtensor& x,
                  const RFtensor& y,
                  const int nx = 1,
                  const int ny = 1) {
  const int K = x.combined_size(x.k - nx, x.k);
  assert(y.combined_size(y.k - ny, y.k) == K);

  // cout<<dims<<endl;
  const int I = x.combined_size(0, x.k - nx);
  const int J = y.combined_size(0, y.k - ny);
  assert(asize == I * J);

  if (device == 0) {
    const int istridex = K;
    const int istrider = J;
    const int jstridey = K;

    for (int i = 0; i < I; i++) {
      for (int j = 0; j < J; j++) {
        float tr = 0;
        for (int p = 0; p < K; p++) {
          int qx = i * istridex + p;
          int qy = p + j * jstridey;
          float xr = x.arr[qx];
          float yr = y.arr[qy];
          tr += xr * yr;
        }
        int qr = i * istrider + j;
        arr[qr] += tr;
      }
    }
  }

  if (device > 0) {
    float alpha0 = 1.0;
    float beta = 1.0;
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J, I, K,
                            &alpha0, y.arrg, K, x.arrg, K, &beta, arrg, J));
  }
}

// Simple vector/matrix matrix/vector matrix/matrix multiply routine
// The first nx indices of x are contracted with the first ny indices of y

void add_Mprod_TA(const RFtensor& x,
                  const RFtensor& y,
                  const int nx = 1,
                  const int ny = 1) {
  const int K = x.combined_size(0, nx);

  if (y.combined_size(0, ny) != K) {
    CoutLock lk;
    cout << K << " " << y.combined_size(0, ny) << endl;
  };
  assert(y.combined_size(0, ny) == K);

  const int I = x.combined_size(nx, x.k);
  const int J = y.combined_size(ny, y.k);
  assert(asize == I * J);

  if (device == 0) {
    const int istrider = J;
    const int pstridex = I;
    const int pstridey = J;

    for (int i = 0; i < I; i++) {
      for (int j = 0; j < J; j++) {
        float tr = 0;
        for (int p = 0; p < K; p++) {
          int qx = i + p * pstridex;
          int qy = p * pstridey + j;
          float xr = x.arr[qx];
          float yr = y.arr[qy];
          tr += xr * yr;
        }
        int qr = i * istrider + j;
        arr[qr] += tr;
      }
    }
  }

  if (device > 0) {
    float alpha0 = 1.0;
    float beta = 1.0;
    CUBLAS_SAFE(cublasSgemm(Cengine_cublas, CUBLAS_OP_N, CUBLAS_OP_T, J, I, K,
                            &alpha0, y.arrg, J, x.arrg, I, &beta, arrg, J));
  }
}
