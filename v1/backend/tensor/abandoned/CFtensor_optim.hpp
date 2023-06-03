public
    :  // ---- Optimization kernels
       // ------------------------------------------------------------------------
void adam_update(CFtensor& g,
                 CFtensor& mt,
                 CFtensor& vt,
                 const float beta1,
                 const float beta2,
                 const float alpha,
                 const float epsilon) {
  assert(g.asize == asize);
  assert(mt.asize == asize);
  assert(vt.asize == asize);

  if (device == 0) {
    assert(g.device == 0);
    assert(mt.device == 0);
    assert(vt.device == 0);
    for (int i = 0; i < asize; i++) {
      mt.arr[i] =
          beta1 * mt.arr[i] + g.arr[i] * (static_cast<float>(1.0) - beta1);
      vt.arr[i] = beta2 * vt.arr[i] +
                  std::pow(g.arr[i], 2) * (static_cast<float>(1.0) - beta2);
      arr[i] -= alpha * mt.arr[i] / (sqrt(vt.arr[i]) + epsilon);
    }
  } else {
    FCG_UNIMPL();
  }
}

void adam_update_complex(CFtensor& g,
                         CFtensor& mt,
                         CFtensor& vt,
                         const float beta1,
                         const float beta2,
                         const float alpha,
                         const float epsilon) {
  assert(g.asize == asize);
  assert(mt.asize == asize);
  assert(vt.asize == asize);

  if (device == 0) {
    assert(g.device == 0);
    assert(mt.device == 0);
    assert(vt.device == 0);
    for (int i = 0; i < asize; i++) {
      mt.arr[i] =
          beta1 * mt.arr[i] + g.arr[i] * (static_cast<float>(1.0) - beta1);
      // vt.arr[i]=beta2*vt.arr[i]+complex<float>(pow(g.arr[i].real(),2),pow(g.arr[i].imag(),2))*(static_cast<float>(1.0)-beta2);
      // arr[i]-=complex<float>(alpha*mt.arr[i].real()/(sqrt(vt.arr[i].real())+epsilon),
      // alpha*mt.arr[i].imag()/(sqrt(vt.arr[i].imag())+epsilon));
    }
  } else {
    FCG_UNIMPL();
  }
}

template <typename Bfloat>
void adagrad_update_complex(const Gtensor<complex<Bfloat>>& g,
                            Gtensor<complex<Bfloat>>& G,
                            const Bfloat eta,
                            const Bfloat epsilon) {
  /*
    assert(g.asize==asize);
    assert(G.asize==asize);

    if(device==0){
      assert(g.device==0);
      assert(G.device==0);

      for(int i=0; i<asize; i++){
        const Bfloat gr=g.arr[i].real();
        const Bfloat gi=g.arr[i].imag();

        G.arr[i]+=complex<Bfloat>(gr*gr,gi*gi);
        arr[i]-=eta*complex<Bfloat>(pow(epsilon+G.arr[i].real(),-0.5)*gr,pow(epsilon+G.arr[i].imag(),-0.5)*gi);
      }
    }else{
      FCG_UNIMPL();
    }

  */
}
