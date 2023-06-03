#ifndef _RtensorB
#define _RtensorB

#include "Cobject.hpp"
#include "RFtensor.hpp"
#include "RscalarB.hpp"

namespace Cengine {

class RtensorB : public Cobject, public RFtensor {
 public:
  Gdims dims;
  int nbu = -1;

  RtensorB() { RTENSORB_CREATE(); }

  ~RtensorB() { RTENSORB_DESTROY(); }

  string classname() const { return "RtensorB"; }

  string describe() const {
    if (nbu >= 0) {
      return "RtensorB" + dims.str() + "[" + to_string(nbu) + "]";
    }
    return "RtensorB" + dims.str();
  }

 public
     :  // ---- Constructors
        // ------------------------------------------------------------------------------
  RtensorB(const Gtensor<complex<float>>& x, const device_id& dev = 0)
      : RFtensor(x), dims(x.dims), nbu(-1) {
    RTENSORB_CREATE();
  }

 public
     :  // ---- Filled constructors
        // -----------------------------------------------------------------------
  template <typename FILLTYPE,
            typename = typename std::enable_if<
                std::is_base_of<fill_pattern, FILLTYPE>::value,
                FILLTYPE>::type>
  RtensorB(const Gdims& _dims, const FILLTYPE& fill, const device_id& dev = 0)
      : RFtensor(_dims, fill, dev), dims(_dims) {
    RTENSORB_CREATE();
  }

  template <typename FILLTYPE,
            typename = typename std::enable_if<
                std::is_base_of<fill_pattern, FILLTYPE>::value,
                FILLTYPE>::type>
  RtensorB(const Gdims& _dims,
           const int _nbu,
           const FILLTYPE& fill,
           const device_id& dev = 0)
      : RFtensor(_dims.prepend(_nbu), fill, dev), dims(_dims), nbu(_nbu) {
    RTENSORB_CREATE();
  }

 public
     :  // ---- Copying
        // -----------------------------------------------------------------------------------
  RtensorB(const RtensorB& x) : RFtensor(x), dims(x.dims), nbu(x.nbu) {
    COPY_WARNING;
    RTENSORB_CREATE();
  }

  RtensorB(const RtensorB& x, const nowarn_flag& dummy)
      : RFtensor(x, dummy), dims(x.dims), nbu(x.nbu) {
    RTENSORB_CREATE();
  }

  RtensorB* clone() const { return new RtensorB(*this, nowarn); }

 public
     :  // ---- Conversions
        // -------------------------------------------------------------------------------
  RtensorB(const RFtensor& x) : RFtensor(x), dims(x.dims), nbu(-1) {
    RTENSORB_CREATE();
  }

  RtensorB(RFtensor&& x) : RFtensor(std::move(x)), dims(x.dims), nbu(-1) {
    RTENSORB_CREATE();
  }

  void to_device(const device_id& _dev) const { RFtensor::to_device(_dev); }

 public
     :  // ---- Access
        // -------------------------------------------------------------------------------------
  int get_nbu() const { return nbu; }

  Gdims get_dims() const { return dims; }

 public
     :  // ---- Operations
        // ---------------------------------------------------------------------------------
  RtensorB* transp() const { return new RtensorB(RFtensor::transp()); }

  RtensorB* divide_cols(const RtensorB& N) const {
    return new RtensorB(RFtensor::divide_cols(N));
  }

  RtensorB* normalize_cols() const {
    return new RtensorB(RFtensor::normalize_cols());
  }

 public
     :  // ---- Cumulative Operations
        // ----------------------------------------------------------------------
  void add_prod(const RscalarB& c, const RtensorB& A) {
    if (c.nbu == -1) {
      RFtensor::add(A, c.val);
    } else {
      FCG_UNIMPL();
    }
  }

  void add_prod(const RscalarB& c, const RtensorB& A) {
    if (c.nbu == -1) {
      RFtensor::add(A, c.val);
    } else {
      FCG_UNIMPL();
    }
  }

  void add_inp_into(RscalarB& r, const RtensorB& A) {
    assert(nbu == -1);
    r.val += inp(A);
  }

  void add_element_into(RscalarB& r, const Gindex& ix) {
    assert(nbu == -1);
    r.val += get(ix);
  }

  void add_to_element(const Gindex& ix, RscalarB& r) {
    assert(nbu == -1);
    inc(ix, r.val);
  }

  void mix_into(RscalarB& r, const RscalarB& x) const {
    to_device(0);
    assert(dims.size() == 2);
    if (r.nbu == -1) {
      assert(dims[0] == 1);
      if (x.nbu == -1) {
        assert(dims[1] == 1);
        r.val += arr[0] * x.val;
        return;
      } else {
        assert(dims[1] == x.nbu);
        for (int i = 0; i < x.nbu; i++) {
          r.val += arr[i] * x.arr[i];
        }
        return;
      }
    } else {
      assert(dims[0] == r.nbu);
      if (x.nbu == -1) {
        assert(dims[1] == 1);
        for (int i = 0; i < r.nbu; i++) {
          r.arr[i] += arr[i] * x.val;
        }
      } else {
        assert(dims[1] == x.nbu);
        for (int i = 0; i < r.nbu; i++) {
          complex<float> t = r.arr[i];
          for (int j = 0; j < x.nbu; j++) {
            t += arr[i * x.nbu + j] * x.val;
          }
          r.arr[i] = t;
        }
      }
    }
  }

  void mix_into(RtensorB& r, const RtensorB& x) const {
    to_device(0);
    assert(dims.size() == 2);
    if (r.nbu == -1) {
      assert(dims[0] == 1);
      if (x.nbu == -1) {
        assert(dims[1] == 1);
        r.add(arr[0]);
        return;
      } else {
        assert(dims[1] == x.nbu);
        FCG_UNIMPL();
        return;
      }
    } else {
      FCG_UNIMPL();
    }
  }

 public
     :  // ---- I/O
        // ----------------------------------------------------------------------------------------
  string str(const string indent = "") const {
    stringstream oss;
    return oss.str();
  }
};

inline RtensorB& asRtensorB(Cobject* x, const char* s) {
  return downcast<RtensorB>(x, s);
}

inline RtensorB& asRtensorB(Cnode* x, const char* s) {
  return downcast<RtensorB>(x, s);
}

inline RtensorB& asRtensorB(Cnode& x, const char* s) {
  return downcast<RtensorB>(x, s);
}

#define RTENSORB(x) asRtensorB(x, __PRETTY_FUNCTION__)

}  // namespace Cengine

#endif
