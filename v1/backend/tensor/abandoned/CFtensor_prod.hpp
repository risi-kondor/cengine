public
    :  // ---- Contractive products
       // -----------------------------------------------------------------------
template <int selector>
CFtensor prod(const CFtensor& y, const CFtensorProductType& T) const {
  const CFtensor& x = *this;
  Gdims D = T.dims(x, y);
  tuple<int, int, int, int> Ttype = T.type();
  CFtensor r(D, fill::raw);

  int I1 = r.dims[T.xout[0].second];
  int J1 = r.dims[T.yout[0].second];
  int P1 = x.dims[T.contract[0].first];

  int i1stridex = x.strides[T.xout[0].first];
  int i1strider = r.strides[T.xout[0].second];

  int j1stridey = y.strides[T.yout[0].first];
  int j1strider = r.strides[T.yout[0].second];

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
          if (constexpr(selector == 0)) {
            tr += xr * yr - xi * yi;
            ti += xr * yi + xi * yr;
          }
          // if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
        }
        r.arr[i1 * i1strider + j1 * j1strider] += tr;
        r.arrc[i1 * i1strider + j1 * j1strider] += ti;
      }
    }

    return r;
  }

  if (Ttype == tuple<int, int, int, int>({1, 1, 1, 1})) {
    int C1 = r.dims[std::get<2>(T.direct[0])];
    int c1stridex = x.strides[std::get<0>(T.direct[0])];
    int c1stridey = y.strides[std::get<1>(T.direct[0])];
    int c1strider = r.strides[std::get<2>(T.direct[0])];

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
            tr += xr * yr - xi * yi;
            ti += xr * yi + xi * yr;
          }
          int qr = i1 * i1strider + j1 * j1strider + c1 * c1strider;
          r.arr[qr] += tr;
          r.arrc[qr] += ti;
        }
      }
    }

    return r;
  }

  return r;
}
