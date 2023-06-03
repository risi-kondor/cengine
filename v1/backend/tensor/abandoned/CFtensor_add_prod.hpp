template <int selector>
void add_prod(const CFtensor& x,
              const CFtensor& y,
              const CFtensorProductType& T) {
  if (device > 0) {
    add_prodG<selector>(x, y, T);
    return;
  }

  tuple<int, int, int, int> Ttype = T.type();
  Gdims D = T.dims(x, y);
  assert(dims == D);

  int I1 = dims[T.xout[0].second];
  int J1 = dims[T.yout[0].second];
  int P1 = x.dims[T.contract[0].first];

  int i1stridex = x.strides[T.xout[0].first];
  int i1strider = strides[T.xout[0].second];

  int j1stridey = y.strides[T.yout[0].first];
  int j1strider = strides[T.yout[0].second];

  int p1stridex = x.strides[T.contract[0].first];
  int p1stridey = y.strides[T.contract[0].second];

  if (Ttype == tuple<int, int, int, int>({1, 1, 1, 0})) {
    for (int i1 = 0; i1 < I1; i1++) {
      for (int j1 = 0; j1 < J1; j1++) {
        float tr = 0;
        float ti = 0;
        for (int p1 = 0; p1 < P1; p1++) {
          int qx = i1 * i1stridex + p1 * p1stridex;
          int qy = j1 * j1stridey + p1 * p1stridey;
          float xr = x.arr[qx];
          float xi = x.arrc[qx];
          float yr = y.arr[qy];
          float yi = y.arrc[qy];
          if constexpr (selector == 0) {
            tr += xr * yr - xi * yi;
            ti += xr * yi + xi * yr;
          }
          if constexpr (selector == 1) {
            tr += xr * yr + xi * yi;
            ti += xr * yi - xi * yr;
          }
          if constexpr (selector == 2) {
            tr += xr * yr + xi * yi;
            ti += (-xr * yi) + xi * yr;
          }
          if constexpr (selector == 3) {
            tr += xr * yr - xi * yi;
            ti -= xr * yi + xi * yr;
          }
        }
        arr[i1 * i1strider + j1 * j1strider] += tr;
        arrc[i1 * i1strider + j1 * j1strider] += ti;
      }
    }

    return;
  }

  if (Ttype == tuple<int, int, int, int>({1, 1, 1, 1})) {
    int C1 = dims[std::get<2>(T.direct[0])];
    int c1stridex = x.strides[std::get<0>(T.direct[0])];
    int c1stridey = y.strides[std::get<1>(T.direct[0])];
    int c1strider = strides[std::get<2>(T.direct[0])];

    for (int c1 = 0; c1 < C1; c1++) {
      for (int i1 = 0; i1 < I1; i1++) {
        for (int j1 = 0; j1 < J1; j1++) {
          float tr = 0;
          float ti = 0;
          for (int p1 = 0; p1 < P1; p1++) {
            int qx = i1 * i1stridex + p1 * p1stridex + c1 * c1stridex;
            int qy = j1 * j1stridey + p1 * p1stridey + c1 * c1stridex;
            float xr = x.arr[qx];
            float xi = x.arrc[qx];
            float yr = y.arr[qy];
            float yi = y.arrc[qy];
            if constexpr (selector == 0) {
              tr += xr * yr - xi * yi;
              ti += xr * yi + xi * yr;
            }
            if constexpr (selector == 1) {
              tr += xr * yr + xi * yi;
              ti += xr * yi - xi * yr;
            }
            if constexpr (selector == 2) {
              tr += xr * yr + xi * yi;
              ti += (-xr * yi) + xi * yr;
            }
            if constexpr (selector == 3) {
              tr += xr * yr - xi * yi;
              ti -= xr * yi + xi * yr;
            }
          }
          int qr = i1 * i1strider + j1 * j1strider + c1 * c1strider;
          arr[qr] += tr;
          arrc[qr] += ti;
        }
      }
    }

    return;
  }

  return;
}

template <int selector>
void add_prodG(const CFtensor& x,
               const CFtensor& y,
               const CFtensorProductType& T) {
  tuple<int, int, int, int> Ttype = T.type();
  x.to_device(device);
  y.to_device(device);

  if (Ttype == tuple<int, int, int, int>({1, 1, 1, 0})) {
    float alpha0 = 1.0;
    float alpha1 = 1.0;
    float alpha2 = 1.0;
    float alpha3 = 1.0;
    float beta = 1.0;

    int I1 = dims[T.xout[0].second];
    int J1 = dims[T.yout[0].second];
    int P1 = x.dims[T.contract[0].first];

    if (T.is_xout(0, 0) && T.is_yout(1, 1) && T.is_contr(1, 0)) {
      if (selector == 0 || selector == 3) {
        alpha1 = -1.0;
      }
      if (selector == 2 || selector == 3) {
        alpha2 = -1.0;
      }
      if (selector == 1 || selector == 3) {
        alpha3 = -1.0;
      }

#ifdef _WITH_CUBLAS
      cublasSgemm(GEnet_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J1, I1, P1, &alpha0,
                  y.arrg, y.strides[0], x.arrg, x.strides[0], &beta, arrg,
                  strides[0]);
      cublasSgemm(GEnet_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J1, I1, P1, &alpha1,
                  y.arrgc, y.strides[0], x.arrgc, x.strides[0], &beta, arrg,
                  strides[0]);
      cublasSgemm(GEnet_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J1, I1, P1, &alpha2,
                  y.arrgc, y.strides[0], x.arrg, x.strides[0], &beta, arrgc,
                  strides[0]);
      cublasSgemm(GEnet_cublas, CUBLAS_OP_N, CUBLAS_OP_N, J1, I1, P1, &alpha3,
                  y.arrg, y.strides[0], x.arrgc, x.strides[0], &beta, arrgc,
                  strides[0]);
#endif
      return;
    }

    if (T.is_xout(0, 1) && T.is_yout(0, 0) && T.is_contr(1, 1)) {
      if (selector == 0 || selector == 3) {
        alpha1 = -1.0;
      }
      if (selector == 2 || selector == 3) {
        alpha2 = -1.0;
      }
      if (selector == 1 || selector == 3) {
        alpha3 = -1.0;
      }

#ifdef _WITH_CUBLAS
      cublasSgemm(GEnet_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J1, I1, P1, &alpha0,
                  x.arrg, x.strides[0], y.arrg, y.strides[0], &beta, arrg,
                  strides[0]);
      cublasSgemm(GEnet_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J1, I1, P1, &alpha1,
                  x.arrgc, x.strides[0], y.arrgc, y.strides[0], &beta, arrg,
                  strides[0]);
      cublasSgemm(GEnet_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J1, I1, P1, &alpha2,
                  x.arrgc, x.strides[0], y.arrg, y.strides[0], &beta, arrgc,
                  strides[0]);
      cublasSgemm(GEnet_cublas, CUBLAS_OP_T, CUBLAS_OP_N, J1, I1, P1, &alpha3,
                  x.arrg, x.strides[0], y.arrgc, y.strides[0], &beta, arrgc,
                  strides[0]);
#endif
      return;
    }
  }
}
