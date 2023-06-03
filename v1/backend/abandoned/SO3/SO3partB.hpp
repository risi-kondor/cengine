#ifndef _SO3partB
#define _SO3partB

#include "CtensorB.hpp"
#include "SO3_CGbank.hpp"
#include "SO3_SPHgen.hpp"

extern GEnet::SO3_CGbank SO3_cgbank;
extern GEnet::SO3_SPHgen SO3_sphGen;

namespace GEnet {

class SO3partB : public CtensorB {
 public:
  string classname() const { return "SO3partB"; }

  string describe() const {
    if (nbu >= 0) {
      return "SO3partB" + dims.str() + "[" + to_string(nbu) + "]";
    }
    return "SO3partB" + dims.str();
  }

  template <typename FILLTYPE>
  SO3partB(const Gdims& _dims,
           const int _nbu,
           const FILLTYPE& fill,
           const device_id& dev = 0)
      : CtensorB(_dims, _nbu, fill, dev) {}

  SO3partB(const Gtensor<complex<float>>& x, const int device = 0)
      : CtensorB(x, device) {}

 public
     :  // ---- Copying
        // -----------------------------------------------------------------------------------
  /*
  SO3partB(const SO3partB& x):
    CFtensor(x){
    COPY_WARNING;
  }

  SO3partB(const SO3partB& x, const nowarn_flag& dummy):
    CFtensor(x,dummy), dims(x.dims), nbu(x.nbu){}

  SO3partB* clone() const{
    return new SO3partB(*this, nowarn);
  }
  */

 public
     :  // ---- Conversions
        // -------------------------------------------------------------------------------
 public
     :  // ---- Access
        // -------------------------------------------------------------------------------------
  int getl() const {
    if (nbu < 1) {
      return (dims(0) - 1) / 2;
    } else {
      return (dims(2) - 1) / 2;
    }
  }

  int getn() const {
    if (nbu < 1) {
      return dims(1);
    } else {
      return dims(2);
    }
  }

  complex<float> operator()(const int i, const int m) const {
    return CFtensor::get(m, i);
  }

  complex<float> get(const int i, const int m) const {
    return CFtensor::get(m, i);
  }

  complex<float> getb(const int b, const int i, const int m) const {
    return CFtensor::get(b, m, i);
  }

  void set(const int i, const int m, complex<float> x) {
    CFtensor::set(m, i, x);
  }

  void setb(const int b, const int i, const int m, complex<float> x) {
    CFtensor::set(b, m, i, x);
  }

  void inc(const int i, const int m, complex<float> x) {
    CFtensor::inc(m, i, x);
  }

  void incb(const int b, const int i, const int m, complex<float> x) {
    CFtensor::inc(b, m, i, x);
  }

 public
     :  // ---- Operations
        // ---------------------------------------------------------------------------------
 public
     :  // ---- Cumulative Operations
        // ----------------------------------------------------------------------
  void add_CGproduct(const CtensorB& _x, const CtensorB& _y, int offs = 0) {
    const SO3partB& x = static_cast<const SO3partB&>(_x);
    const SO3partB& y = static_cast<const SO3partB&>(_y);

    if (device == 1) {
#ifdef _WITH_CUDA
      x.to_device(1);
      y.to_device(1);
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      // CGproduct_cu(x,y,offs,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      NOCUDA_ERROR;
#endif
      return;
    }

    const int l = getl();
    const int l1 = x.getl();
    const int l2 = y.getl();
    const int N1 = x.getn();
    const int N2 = y.getn();
    auto& C = SO3_cgbank.getf(CGindex(l1, l2, l));

    if (nbu < 0) {
      for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
          for (int m1 = -l1; m1 <= l1; m1++) {
            for (int m2 = std::max(-l2, -l - m1); m2 <= std::min(l2, l - m1);
                 m2++) {
              inc(offs + n2, m1 + m2 + l,
                  C(m1 + l1, m2 + l2) * x(n1, m1 + l1) * y(n2, m2 + l2));
            }
          }
        }
        offs += N2;
      }
      return;
    }

    assert(x.nbu == nbu);
    assert(y.nbu == nbu);

    for (int b = 0; b < nbu; b++) {
      int _offs = offs;
      for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
          for (int m1 = -l1; m1 <= l1; m1++) {
            for (int m2 = std::max(-l2, -l - m1); m2 <= std::min(l2, l - m1);
                 m2++) {
              incb(b, _offs + n2, m1 + m2 + l,
                   C(m1 + l1, m2 + l2) * x.getb(b, n1, m1 + l1) *
                       y.getb(b, n2, m2 + l2));
            }
          }
        }
        _offs += N2;
      }
    }
  }

  void add_CGproduct_back0(const CtensorB& _g,
                           const CtensorB& _y,
                           int offs = 0) {
    const SO3partB& g = static_cast<const SO3partB&>(_g);
    const SO3partB& y = static_cast<const SO3partB&>(_y);

    if (device == 1) {
#ifdef _WITH_CUDA
      xg.to_device(1);
      y.to_device(1);
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      // CGproduct_g1cu(xg,y,offs,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      NOCUDA_ERROR;
#endif
      return;
    }

    const int l = g.getl();
    const int l1 = getl();
    const int l2 = y.getl();
    const int N1 = getn();
    const int N2 = y.getn();
    const SO3_CGcoeffs<float>& C = SO3_cgbank.get<float>(l1, l2, l);

    if (nbu < 0) {
      for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
          for (int m1 = -l1; m1 <= l1; m1++) {
            for (int m2 = std::max(-l2, -l - m1); m2 <= std::min(l2, l - m1);
                 m2++) {
              inc(n1, m1 + l1,
                  C(m1 + l1, m2 + l2) * std::conj(y(n2, m2 + l2)) *
                      g(offs + n2, m1 + m2 + l));
            }
          }
        }
        offs += N2;
      }
      return;
    }
  }

  void add_CGproduct_back1(const CtensorB& _g,
                           const CtensorB& _x,
                           int offs = 0) {
    const SO3partB& g = static_cast<const SO3partB&>(_g);
    const SO3partB& x = static_cast<const SO3partB&>(_x);

    if (device == 1) {
#ifdef _WITH_CUDA
      x.to_device(1);
      yg.to_device(1);
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      // CGproduct_g2cu(x,yg,offs,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      NOCUDA_ERROR;
#endif
      return;
    }

    const int l = g.getl();
    const int l1 = x.getl();
    const int l2 = getl();
    const int N1 = x.getn();
    const int N2 = getn();
    const SO3_CGcoeffs<float>& C = SO3_cgbank.get<float>(l1, l2, l);

    if (nbu < 0) {
      for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
          for (int m1 = -l1; m1 <= l1; m1++) {
            for (int m2 = std::max(-l2, -l - m1); m2 <= std::min(l2, l - m1);
                 m2++) {
              inc(n2, m2 + l2,
                  C(m1 + l1, m2 + l2) * std::conj(x(n1, m1 + l1)) *
                      g(offs + n2, m1 + m2 + l));
            }
          }
        }
        offs += N2;
      }
      return;
    }
  }

 public
     :  // ---- Spherical harmonics
        // -----------------------------------------------------------------------
  SO3partB(const int l,
           const float x,
           const float y,
           const float z,
           const int nbu,
           const device_id& dev = 0)
      : SO3partB({2 * l + 1, 1}, nbu, fill::raw, dev) {
    float length = sqrt(x * x + y * y + z * z);
    float len2 = sqrt(x * x + y * y);
    complex<float> cphi(x / len2, y / len2);

    Gtensor<float> P = SO3_sphGen(l, z / length);
    cout << "P=" << P << endl;
    vector<complex<float>> phase(l + 1);
    phase[0] = complex<float>(1.0, 0);
    for (int m = 1; m <= l; m++) {
      phase[m] = cphi * phase[m - 1];
    }

    for (int m = 0; m <= l; m++) {
      (*this)(l + m, 0) = phase[m] * complex<float>(P(l, m));  // *(1-2*(m%2))
      (*this)(l - m, 0) =
          complex<float>(1 - 2 * (m % 2)) * std::conj((*this)(l + m, 0));
    }
  }

 public
     :  // ---- I/O
        // ----------------------------------------------------------------------------------------
};

inline SO3partB& asSO3partB(Cobject* x) {
  assert(x);
  if (!dynamic_cast<SO3partB*>(x)) {
    cerr << "GEnet error: Cobject is of type " << x->classname()
         << " instead of SO3partB." << endl;
  }
  assert(dynamic_cast<SO3partB*>(x));
  return static_cast<SO3partB&>(*x);
}

inline SO3partB& asSO3partB(Cnode* x) {
  assert(x->obj);
  if (!dynamic_cast<SO3partB*>(x->obj)) {
    cerr << "GEnet error: Cobject is of type " << x->obj->classname()
         << " instead of SO3partB." << endl;
  }
  assert(dynamic_cast<SO3partB*>(x->obj));
  return static_cast<SO3partB&>(*x->obj);
}

inline SO3partB& asSO3partB(Cnode& x) {
  assert(x.obj);
  if (!dynamic_cast<SO3partB*>(x.obj)) {
    cerr << "GEnet error: Cobject is of type " << x.obj->classname()
         << " instead of SO3partB." << endl;
  }
  assert(dynamic_cast<SO3partB*>(x.obj));
  return static_cast<SO3partB&>(*x.obj);
}

}  // namespace GEnet

#endif
