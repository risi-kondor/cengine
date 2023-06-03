#ifndef _CtensorArrayB
#define _CtensorArrayB

#include "CFtensorArray.hpp"
#include "Cobject.hpp"
#include "CscalarB.hpp"
#include "RscalarB.hpp"

namespace Cengine {

template <typename OBJ>
class CellRef {
 public:
  OBJ* owner;
  float* arr;
  float* arrc;
  vector<int> strides;

 public:
  CellRef(float* _arr, float* _arrc, const vector<int>& _strides)
      : arr(_arr), arrc(_arrc), strides(_strides) {}

  complex<float> operator()(const int i0, const int i1) {
    int t = strides[0] * i0 + strides[1] * i1;
    return complex<float>(arr[t], arrc[t]);
  }

  void inc(const int i0, const int i1, complex<float> x) {
    int t = strides[0] * i0 + strides[1] * i1;
    arr[t] += std::real(x);
    arrc[t] += std::imag(x);
  }
};

class CtensorArrayB : public Cobject, public CFtensorArray {
 public:
  // Gdims adims;
  Gdims dims;
  int nbu = -1;

  CtensorArrayB() { CTENSORARRAYB_CREATE(); }

  ~CtensorArrayB() { CTENSORARRAYB_DESTROY(); }

  string classname() const { return "CtensorArrayB"; }

  string describe() const {
    if (nbu >= 0) {
      return "CtensorArrayB" + adims.str() + " " + dims.str() + "[" +
             to_string(nbu) + "]";
    }
    return "CtensorArrayB" + adims.str() + " " + dims.str();
  }

 public
     :  // ---- Constructors
        // ------------------------------------------------------------------------------
  /*
  CtensorArrayB(const Gtensor<complex<float> >& x, const int dev=0):
    CFtensor(x), dims(x.dims), nbu(-1){
    CTENSORARRAYB_CREATE();
  }
  */

  /*
  CtensorArrayB(const Gtensor<complex<float> >& x, const device_id& dev=0):
    CFtensor(x), dims(x.dims), nbu(-1){
    CTENSORARRAYB_CREATE();
  }
  */

 public
     :  // ---- Filled constructors
        // -----------------------------------------------------------------------
  template <typename FILLTYPE,
            typename = typename std::enable_if<
                std::is_base_of<fill_pattern, FILLTYPE>::value,
                FILLTYPE>::type>
  CtensorArrayB(const Gdims& _adims,
                const Gdims& _dims,
                const FILLTYPE& fill,
                const int dev = 0)
      : CFtensorArray(_adims, _dims, fill, dev), dims(_dims) {
    CTENSORARRAYB_CREATE();
  }

  template <typename FILLTYPE,
            typename = typename std::enable_if<
                std::is_base_of<fill_pattern, FILLTYPE>::value,
                FILLTYPE>::type>
  CtensorArrayB(const Gdims& _adims,
                const Gdims& _dims,
                const int _nbu,
                const FILLTYPE& fill,
                const int dev = 0)
      : CFtensorArray(_adims, _dims.prepend(_nbu), fill, dev),
        dims(_dims),
        nbu(_nbu) {
    CTENSORARRAYB_CREATE();
  }

  CtensorArrayB(const Gdims& _adims,
                const Gdims& _dims,
                const int _nbu,
                const fill_gaussian& fill,
                const float c,
                const int dev = 0)
      : CFtensorArray(_adims, _dims.prepend(_nbu), fill, c, dev),
        dims(_dims),
        nbu(_nbu) {
    CTENSORARRAYB_CREATE();
  }

  /*
  CtensorArrayB(const Gdims& _adims, const Gdims& _dims, const int _nbu,
  std::function<complex<float>(const int i, const int j)> fn, const int dev=0):
    CFtensor(_dims.prepend(_nbu),fill::raw), adims(_adims), dims(_dims),
  nbu(_nbu){ if(nbu==-1){ for(int i=0; i<dims[0]; i++) for(int j=0; j<dims[1];
  j++) CFtensor::set(i,j,fn(i,j)); }else{ for(int b=0; b<nbu; b++) for(int i=0;
  i<dims[0]; i++) for(int j=0; j<dims[1]; j++) CFtensor::set(b,i,j);
    }
    if(dev>0) to_device(dev);
    CTENSORARRAYB_CREATE();
  }
  */

  /*
  CtensorArrayB(const CtensorArrayB& x, std::function<complex<float>(const
  complex<float>)> fn): CFtensor(x,fn), adims(x.adims), dims(x.dims){
    CTENSORARRAYB_CREATE();
  }
  */

  CtensorArrayB(
      const CtensorArrayB& x,
      std::function<
          complex<float>(const int i, const int j, const complex<float>)> fn)
      : CFtensorArray(x, fill::raw), dims(x.dims) {
    /*
    assert(dims.size()==2);
    if(nbu==-1){
      for(int i=0; i<dims[0]; i++)
        for(int j=0; j<dims[1]; j++)
          CFtensor::set(i,j,fn(i,j,x.CFtensor::get(i,j)));
    }else{
      for(int b=0; b<nbu; b++)
        for(int i=0; i<dims[0]; i++)
          for(int j=0; j<dims[1]; j++)
            CFtensor::set(b,i,j,fn(i,j,x.CFtensor::get(b,i,j)));
    }
    */
    CTENSORARRAYB_CREATE();
  }

 public
     :  // ---- Copying
        // -----------------------------------------------------------------------------------
  CtensorArrayB(const CtensorArrayB& x)
      : CFtensorArray(x), dims(x.dims), nbu(x.nbu) {
    COPY_WARNING;
    CTENSORARRAYB_CREATE();
  }

  CtensorArrayB(const CtensorArrayB& x, const nowarn_flag& dummy)
      : CFtensorArray(x, dummy), dims(x.dims), nbu(x.nbu) {
    CTENSORARRAYB_CREATE();
  }

  CtensorArrayB* clone() const { return new CtensorArrayB(*this, nowarn); }

 public
     :  // ---- Conversions
        // -------------------------------------------------------------------------------
  CtensorArrayB(const CFtensorArray& x)
      : CFtensorArray(x), dims(x.dims), nbu(-1) {
    CTENSORARRAYB_CREATE();
  }

  CtensorArrayB(CFtensorArray&& x)
      : CFtensorArray(std::move(x)), dims(x.dims), nbu(-1) {
    CTENSORARRAYB_CREATE();
  }

  void to_device(const int _dev) const { CFtensorArray::to_device(_dev); }

 public
     :  // ---- Access
        // -------------------------------------------------------------------------------------
  int get_nbu() const { return nbu; }

  Gdims get_adims() const { return adims; }

  Gdims get_dims() const { return dims; }

  int get_device() const { return device; }

  CellRef<CtensorArrayB> cellrf(const int i) {
    return CellRef<CtensorArrayB>(arr + i * cellstride, arrc + i * cellstride,
                                  strides);
  }

  CtensorB get_cell(const Gindex& aix) const {
    CtensorB R(dims, nbu, fill::raw, device);
    copy_cell_into(R, aix);
    return R;
  }

  void copy_cell_into(CtensorB& R, const Gindex& aix) const {
    int t = aix(astrides);
    tmpdev<CtensorB>(device, R);
    if (device == 0) {
      std::copy(arr + t, arr + t + asize, R.arr);
      std::copy(arrc + t, arrc + t + asize, R.arrc);
      return;
    }
    CUDA_SAFE(cudaMemcpy(arrg + t, R.arrg, asize * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CUDA_SAFE(cudaMemcpy(arrgc + t, R.arrgc, asize * sizeof(float),
                         cudaMemcpyDeviceToDevice));
  }

  void add_cell_into(CtensorB& R, const Gindex& aix) const {
    int t = aix(astrides);
    tmpdev<CtensorB>(device, R);
    if (device == 0) {
      float* p = arr + t;
      float* pc = arrc + t;
      for (int i = 0; i < asize; i++) {
        R.arr[i] += p[i];
      }
      for (int i = 0; i < asize; i++) {
        R.arrc[i] += pc[i];
      }
      return;
    }
    // CUDA_SAFE(cudaMemcpy(arrg+t,R.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
    // CUDA_SAFE(cudaMemcpy(arrgc+t,R.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));
  }

  void copy_into_cell(const Gindex& aix, const CtensorB& x) {
    assert(x.asize == asize);
    assert(nbu == x.nbu);
    int t = aix(astrides);
    tmpdev<CtensorB> tt(device, x);
    if (device == 0) {
      std::copy(x.arr, x.arr + asize, arr + t);
      std::copy(x.arrc, x.arrc + asize, arrc + t);
      return;
    }
    CUDA_SAFE(cudaMemcpy(x.arrg, arrg + t, asize * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CUDA_SAFE(cudaMemcpy(x.arrgc, arrgc + t, asize * sizeof(float),
                         cudaMemcpyDeviceToDevice));
  }

  void add_into_cell(const Gindex& aix, const CtensorB& x) {
    assert(x.asize == asize);
    assert(nbu == x.nbu);
    int t = aix(astrides);
    tmpdev<CtensorB> tt(device, x);
    if (device == 0) {
      float* p = arr + t;
      float* pc = arrc + t;
      for (int i = 0; i < asize; i++) {
        p[i] += x.arr[i];
      }
      for (int i = 0; i < asize; i++) {
        pc[i] += x.arrc[i];
      }
      return;
    }
    // CUDA_SAFE(cudaMemcpy(x.arrg,arrg+t,asize*sizeof(float),cudaMemcpyDeviceToDevice));
    // CUDA_SAFE(cudaMemcpy(x.arrgc,arrgc+t,asize*sizeof(float),cudaMemcpyDeviceToDevice));
  }

  void copy_into_cell(const Gindex& aix,
                      const CtensorArrayB& x,
                      const Gindex& xaix) {
    assert(x.asize == asize);
    assert(nbu == x.nbu);
    int t = aix(astrides) * cellstride;
    int xt = xaix(x.astrides) * x.cellstride;
    tmpdev<CtensorArrayB> tt(device, x);
    if (device == 0) {
      std::copy(x.arr + xt, x.arr + xt + asize, arr + t);
      std::copy(x.arrc + xt, x.arrc + xt + asize, arrc + t);
      return;
    }
    CUDA_SAFE(cudaMemcpy(x.arrg + xt, arrg + t, asize * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CUDA_SAFE(cudaMemcpy(x.arrgc + xt, arrgc + t, asize * sizeof(float),
                         cudaMemcpyDeviceToDevice));
  }

  void add_into_cell(const Gindex& aix,
                     const CtensorArrayB& x,
                     const Gindex& xaix) {
    assert(x.asize == asize);
    assert(nbu == x.nbu);
    int t = aix(astrides) * cellstride;
    int xt = xaix(x.astrides) * x.cellstride;
    tmpdev<CtensorArrayB> tt(device, x);
    if (device == 0) {
      stdadd(x.arr + xt, x.arr + xt + asize, arr + t);
      stdadd(x.arrc + xt, x.arrc + xt + asize, arrc + t);
      return;
    }
    // CUDA_SAFE(cudaMemcpy(x.arrg+xt,arrg+t,asize*sizeof(float),cudaMemcpyDeviceToDevice));
    // CUDA_SAFE(cudaMemcpy(x.arrgc+xt,arrgc+t,asize*sizeof(float),cudaMemcpyDeviceToDevice));
  }

 public
     :  // ---- Operations
        // ---------------------------------------------------------------------------------
  CtensorArrayB* conj() const {
    return new CtensorArrayB(CFtensorArray::conj());
  }

  /*
  CtensorArrayB* transp() const{
    return new CtensorArrayB(CFtensor::transp());
  }

  CtensorArrayB* herm() const{
    return new CtensorArrayB(CFtensor::herm());
  }
  */

  /*
  CtensorArrayB* divide_cols(const CtensorArrayB& N) const{
    return new CtensorArrayB(CFtensor::divide_cols(N));
  }

  CtensorArrayB* normalize_cols() const{
    return new CtensorArrayB(CFtensor::normalize_cols());
  }
  */

 public
     :  // ---- Cumulative Operations
        // ----------------------------------------------------------------------
  void add_prod(const RscalarB& c, const CtensorArrayB& A) {
    if (c.nbu == -1) {
      // CFtensorArray::add(A,c.val);
    } else {
      FCG_UNIMPL();
    }
  }

  void add_prod(const CscalarB& c, const CtensorArrayB& A) {
    if (c.nbu == -1) {
      // CFtensorArray::add(A,c.val);
    } else {
      FCG_UNIMPL();
    }
  }

  void add_prod_cconj(const CscalarB& c, const CtensorArrayB& A) {
    if (c.nbu == -1) {
      // CFtensorArray::add(A,std::conj(c.val));
    } else {
      FCG_UNIMPL();
    }
  }

  void add_prod_c_times_conj(const CscalarB& c, const CtensorArrayB& A) {
    if (c.nbu == -1) {
      // CFtensorArray::add_conj(A,c.val);
    } else {
      FCG_UNIMPL();
    }
  }

  void add_inp_into(CscalarB& r, const CtensorArrayB& A) {
    if (nbu == -1) {
      // r.val+=inp(A);
    } else {
      FCG_UNIMPL();
    }
  }

  /*
  void add_element_into(CscalarB& r, const Gindex& ix){
    if(nbu==-1){
      r.val+=get(ix);
    }else{
      FCG_UNIMPL();
    }
  }
  */

  /*
  void add_to_element(const Gindex& ix, CscalarB& r){
    assert(nbu==-1);
    if(nbu==-1){
      inc(ix,r.val);
    }else{
      FCG_UNIMPL();
    }
  }
  */

  // ---- Broadcasting and reductions
  // ----------------------------------------------------------------------

  void broadcast(const int ix, const CtensorArrayB& x) {
    assert(nbu == x.nbu);
    assert(x.ak == ak - 1);
    for (int i = 0; i < ix; i++) {
      assert(adims[i] == x.adims[i]);
    }
    for (int i = ix + 1; i < adims.size(); i++) {
      assert(adims[i] == x.adims[i - 1]);
    }
    int n = adims[ix];
    tmpdev<CtensorArrayB> tt(device, x);
    if (device == 0) {
      if (ix == 0) {
        for (int i = 0; i < n; i++) {
          // std::copy(x.arr,x.arr+x.aasize*x.cellstride,arr+i*astrides[0]*cellstride);
          // std::copy(x.arrc,x.arrc+x.aasize*x.cellstride,arrc+i*astrides[0]*cellstride);
          std::copy(x.arr, x.arr + x.aasize * x.cellstride,
                    arr + i * astrides[0]);
          std::copy(x.arrc, x.arrc + x.aasize * x.cellstride,
                    arrc + i * astrides[0]);
        }
      } else {
        const int A = (astrides[0] * adims[0]) / astrides[ix - 1];
        for (int a = 0; a < A; a++) {
          const int offs = a * astrides[ix - 1];
          const int xoffs = a * x.astrides[ix - 1];
          for (int i = 0; i < n; i++) {
            // std::copy(x.arr+xoffs,x.arr+xoffs+x.astrides[ix-1]*x.cellstride,arr+offs+astrides[ix]*cellstride);
            // std::copy(x.arrc+xoffs,x.arrc+xoffs+x.astrides[ix-1]*x.cellstride,arrc+offs+astrides[ix]*cellstride);
            std::copy(x.arr + xoffs, x.arr + xoffs + x.astrides[ix - 1],
                      arr + offs + i * astrides[ix]);
            std::copy(x.arrc + xoffs, x.arrc + xoffs + x.astrides[ix - 1],
                      arrc + offs + i * astrides[ix]);
          }
        }
      }
    }
    FCG_CPUONLY();
  }

  void add_broadcast(const int ix, const CtensorArrayB& x) {
    assert(nbu == x.nbu);
    assert(x.ak == ak - 1);
    for (int i = 0; i < ix; i++) {
      assert(adims[i] == x.adims[i]);
    }
    for (int i = ix + 1; i < adims.size(); i++) {
      assert(adims[i] == x.adims[i - 1]);
    }
    int n = adims[ix];
    tmpdev<CtensorArrayB> tt(device, x);
    if (device == 0) {
      if (ix == 0) {
        for (int i = 0; i < n; i++) {
          stdadd(x.arr, x.arr + x.aasize * x.cellstride, arr + i * astrides[0]);
          stdadd(x.arrc, x.arrc + x.aasize * x.cellstride,
                 arrc + i * astrides[0]);
        }
      } else {
        const int A = (astrides[0] * adims[0]) / astrides[ix - 1];
        for (int a = 0; a < A; a++) {
          const int offs = a * astrides[ix - 1];
          const int xoffs = a * x.astrides[ix - 1];
          for (int i = 0; i < n; i++) {
            stdadd(x.arr + xoffs, x.arr + xoffs + x.astrides[ix - 1],
                   arr + offs + i * astrides[ix]);
            stdadd(x.arrc + xoffs, x.arrc + xoffs + x.astrides[ix - 1],
                   arrc + offs + i * astrides[ix]);
          }
        }
      }
    }
    FCG_CPUONLY();
  }

  void collapse_add_into(CtensorArrayB& r, const int ix) {
    assert(nbu == r.nbu);
    assert(r.ak == ak - 1);
    for (int i = 0; i < ix; i++) {
      assert(adims[i] == r.adims[i]);
    }
    for (int i = ix + 1; i < adims.size(); i++) {
      assert(adims[i] == r.adims[i - 1]);
    }
    int n = adims[ix];

    tmpdev<CtensorArrayB> tt(device, r);
    if (device == 0) {
      if (ix == 0) {
        for (int i = 0; i < n; i++) {
          // stdadd(arr+i*astrides[0]*cellstride,arr+(i+1)*astrides[0]*cellstride,r.arr);
          // stdadd(arrc+i*astrides[0]*cellstride,arrc+(i+1)*astrides[0]*cellstride,r.arrc);
          stdadd(arr + i * astrides[0], arr + (i + 1) * astrides[0], r.arr);
          stdadd(arrc + i * astrides[0], arrc + (i + 1) * astrides[0], r.arrc);
        }
      } else {
        const int A = (astrides[0] * adims[0]) / astrides[ix - 1];
        for (int a = 0; a < A; a++) {
          const int offs = a * astrides[ix - 1];
          const int xoffs = a * r.astrides[ix - 1];
          for (int i = 0; i < n; i++) {
            // stdadd(arr+offs+i*astrides[ix]*cellstride,arr+offs+(i+1)*astrides[ix]*cellstride,r.arr+xoffs);
            // stdadd(arrc+offs+i*astrides[ix]*cellstride,arrc+offs+(i+1)*astrides[ix]*cellstride,r.arrc+xoffs);
            stdadd(arr + offs + i * astrides[ix],
                   arr + offs + (i + 1) * astrides[ix], r.arr + xoffs);
            stdadd(arrc + offs + i * astrides[ix],
                   arrc + offs + (i + 1) * astrides[ix], r.arrc + xoffs);
          }
        }
      }
    } else {
    }
    FCG_CPUONLY();
  }

  // ---- Mixing
  // -------------------------------------------------------------------------------------------

  /*
  void mix_into(CscalarB& r, const CscalarB& x) const{
    to_device(0);
    assert(dims.size()==2);
    if(r.nbu==-1){
      assert(dims[0]==1);
      if(x.nbu==-1){
        assert(dims[1]==1);
        r.val+=complex<float>(arr[0],arrc[0])*x.val;
        return;
      }else{
        assert(dims[1]==x.nbu);
        for(int i=0; i<x.nbu; i++)
          r.val+=complex<float>(arr[i],arrc[i])*x.arr[i];
        return;
      }
    }else{
      assert(dims[0]==r.nbu);
      if(x.nbu==-1){
        assert(dims[1]==1);
        for(int i=0; i<r.nbu; i++)
          r.arr[i]+=complex<float>(arr[i],arrc[i])*x.val;
      }else{
        assert(dims[1]==x.nbu);
        for(int i=0; i<r.nbu; i++){
          complex<float> t=r.arr[i];
          for(int j=0; j<x.nbu; j++)
            t+=complex<float>(arr[i*x.nbu+j],arrc[i*x.nbu+j])*x.val;
          r.arr[i]=t;
        }
      }
    }
  }
  */

  /*
  void mix_into(CtensorArrayB& r, const CtensorArrayB& x) const{
    to_device(0);
    assert(dims.size()==2);
    if(r.nbu==-1){
      assert(dims[0]==1);
      if(x.nbu==-1){
        assert(dims[1]==1);
        r.add(x,complex<float>(arr[0],arrc[0]));
        return;
      }else{
        assert(dims[1]==x.nbu);
        FCG_UNIMPL();
        return;
      }
    }else{
      FCG_UNIMPL();
    }
  }
  */

 public
     :  // ---- I/O
        // ----------------------------------------------------------------------------------------
  string str(const string indent = "") const {
    pullin<CtensorArrayB>(*this);
    ostringstream oss;

    for (int i = 0; i < aasize; i++) {
      Gindex aix(i, adims);
      oss << indent << "Cell " << aix << endl;
      oss << get_cell(aix).str(indent) << endl << endl;
    }

    return oss.str();
  }
};

inline CtensorArrayB& asCtensorArrayB(Cobject* x, const char* s) {
  return downcast<CtensorArrayB>(x, s);
}

inline CtensorArrayB& asCtensorArrayB(Cnode* x, const char* s) {
  return downcast<CtensorArrayB>(x, s);
}

inline CtensorArrayB& asCtensorArrayB(Cnode& x, const char* s) {
  return downcast<CtensorArrayB>(x, s);
}

#define CTENSORARRAYB(x) asCtensorArrayB(x, __PRETTY_FUNCTION__)

}  // namespace Cengine

#endif
