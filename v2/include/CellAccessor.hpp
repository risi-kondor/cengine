#ifndef _CellAccessor
#define _CellAccessor

namespace Cengine {

template <typename TYPE>
class pretend_complex {
 public:
  TYPE* realp;
  TYPE* imagp;
  pretend_complex(TYPE* _realp, TYPE* _imagp) : realp(_realp), imagp(_imagp) {}

 public:
  operator complex<TYPE>() const { return complex<TYPE>(*realp, *imagp); }
  complex<TYPE> conj() const { return complex<TYPE>(*realp, -*imagp); }
  pretend_complex& operator=(const complex<TYPE> x) {
    *realp = std::real(x);
    *imagp = std::imag(x);
    return *this;
  }

 public:
  pretend_complex& operator+=(const complex<TYPE> x) {
    *realp += std::real(x);
    *imagp += std::imag(x);
    return *this;
  }
  pretend_complex& operator-=(const complex<TYPE> x) {
    *realp -= std::real(x);
    *imagp -= std::imag(x);
    return *this;
  }
  pretend_complex& operator*=(const complex<TYPE> x) {
    *realp = std::real(x) * (*realp) - std::imag(x) * (*imagp);
    *imagp = std::real(x) * (*imagp) + std::imag(x) * (*realp);
    return *this;
  }
  pretend_complex& operator/=(const complex<TYPE> x) {
    complex<float> t = complex<float>(*realp, *imagp) / x;
    *realp = std::real(t);
    *imagp = std::imag(t);
    return *this;
  }

 public:
  complex<float> operator*(const pretend_complex& y) const {
    return complex<TYPE>(*this) * complex<TYPE>(y);
  }

 public:
  friend ostream& operator<<(ostream& stream, const pretend_complex& x) {
    stream << complex<TYPE>(x);
    return stream;
  }
};

template <typename OBJ>
class CtensorAccessor {
 public:
  // OBJ* owner;
  float* arr;
  float* arrc;
  const vector<int>& strides;

 public:
  CtensorAccessor(float* _arr, float* _arrc, const vector<int>& _strides)
      : arr(_arr), arrc(_arrc), strides(_strides) {}

 public:
  pretend_complex<float> operator[](const int t) {
    return pretend_complex<float>(arr + t, arrc + t);
  }

  complex<float> geti(const int t) { return complex<float>(arr[t], arrc[t]); }

  void seti(const int t, complex<float> x) {
    arr[t] = std::real(x);
    arrc[t] = std::imag(x);
  }

  void inci(const int t, complex<float> x) {
    arr[t] += std::real(x);
    arrc[t] += std::imag(x);
  }

 public:
  complex<float> operator()(const int i0, const int i1) {
    int t = strides[0] * i0 + strides[1] * i1;
    return complex<float>(arr[t], arrc[t]);
  }

  void set(const int i0, const int i1, complex<float> x) {
    int t = strides[0] * i0 + strides[1] * i1;
    arr[t] = std::real(x);
    arrc[t] = std::imag(x);
  }

  void inc(const int i0, const int i1, complex<float> x) {
    int t = strides[0] * i0 + strides[1] * i1;
    arr[t] += std::real(x);
    arrc[t] += std::imag(x);
  }

 public:
};

}  // namespace Cengine

#endif
