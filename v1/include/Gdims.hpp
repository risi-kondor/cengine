#ifndef __Gdims
#define __Gdims

#include "Cengine_base.hpp"
// #include "Bifstream.hpp"
// #include "Bofstream.hpp"

namespace Cengine {

class Gdims : public vector<int> {
 public:
  Gdims() {}

  // Gdims(const vector<int>& x): vector<int>(x){}

  Gdims(const int k, const fill_raw& dummy) : vector<int>(k) {}

  Gdims(const vector<int>& x) {
    for (auto p : x) {
      if (p >= 0) {
        push_back(p);
      }
    }
  }

  Gdims(const initializer_list<int>& x) {
    for (auto p : x) {
      if (p >= 0) {
        push_back(p);
      }
    }
  }

  Gdims(const int i0) : vector<int>(1) { (*this)[0] = i0; }

  Gdims(const int i0, const int i1) : vector<int>(2) {
    (*this)[0] = i0;
    (*this)[1] = i1;
  }

  Gdims(const int i0, const int i1, const int i2) : vector<int>(3) {
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
  }

  Gdims(const int i0, const int i1, const int i2, const int i3)
      : vector<int>(4) {
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
    (*this)[3] = i3;
  }

  Gdims(const int i0, const int i1, const int i2, const int i3, const int i4)
      : vector<int>(5) {
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
    (*this)[3] = i3;
    (*this)[4] = i4;
  }

  Gdims(const Gdims& d1, const Gdims& d2) : vector<int>(d1.size() + d2.size()) {
    for (int i = 0; i < d1.size(); i++) {
      (*this)[i] = d1[i];
    }
    for (int i = 0; i < d2.size(); i++) {
      (*this)[i + d1.size()] = d2[i];
    }
  }

 public:
  int k() const { return size(); }

  int operator()(const int i) const { return (*this)[i]; }

  int asize() const {
    int t = 1;
    for (int i = 0; i < size(); i++) {
      t *= (*this)[i];
    }
    return t;
  }

  int first() const { return (*this)[0]; }

  int last() const { return (*this)[size() - 1]; }

 public:
  int combined(const int a, const int b) const {
    assert(a <= b);
    assert(b <= size());
    int t = 1;
    for (int i = a; i < b; i++) {
      t *= (*this)[i];
    }
    return t;
  }

  Gdims chunk(const int beg, int n = -1) const {
    if (n == -1) {
      n = size() - beg;
    }
    Gdims R;
    for (int i = 0; i < n; i++) {
      R.push_back((*this)[beg + i]);
    }
    return R;
  }

  Gdims remove(const int j) const {
    Gdims R;
    if (j < 0) {
      for (int i = 0; i < size(); i++) {
        if (i != size() + j) {
          R.push_back((*this)[i]);
        }
      }
    } else {
      for (int i = 0; i < size(); i++) {
        if (i != j) {
          R.push_back((*this)[i]);
        }
      }
    }
    return R;
  }

  Gdims insert(const int j, const int n) const {
    Gdims R;
    for (int i = 0; i < j; i++) {
      R.push_back((*this)[i]);
    }
    R.push_back(n);
    for (int i = j; i < size(); i++) {
      R.push_back((*this)[i]);
    }
    return R;
  }

  Gdims append(const int i) const {
    Gdims R = *this;
    if (i >= 0) {
      R.push_back(i);
    }
    return R;
  }

  Gdims prepend(const int i) const {
    Gdims R;
    if (i >= 0) {
      R.push_back(i);
    }
    for (auto p : *this) {
      R.push_back(p);
    }
    return R;
  }

  Gdims transpose() const {
    assert(size() == 2);
    return Gdims((*this)[1], (*this)[0]);
  }

  Gdims Mprod(const Gdims& y) const {
    Gdims R(size() + y.size() - 2, fill::raw);
    for (int i = 0; i < size() - 1; i++) {
      R[i] = (*this)[i];
    }
    for (int i = 0; i < y.size() - 1; i++) {
      R[i + size() - 1] = y[i + 1];
    }
    return R;
  }

  Gdims Mprod_AT(const Gdims& y) const {
    Gdims R(size() + y.size() - 2, fill::raw);
    for (int i = 0; i < size() - 1; i++) {
      R[i] = (*this)[i];
    }
    for (int i = 0; i < y.size() - 1; i++) {
      R[i + size() - 1] = y[i];
    }
    return R;
  }

  Gdims Mprod_TA(const Gdims& y) const {
    Gdims R(size() + y.size() - 2, fill::raw);
    for (int i = 0; i < size() - 1; i++) {
      R[i] = (*this)[i + 1];
    }
    for (int i = 0; i < y.size() - 1; i++) {
      R[i + size() - 1] = y[i + 1];
    }
    return R;
  }

  Gdims Mprod_TT(const Gdims& y) const {
    Gdims R(size() + y.size() - 2, fill::raw);
    for (int i = 0; i < size() - 1; i++) {
      R[i] = (*this)[i + 1];
    }
    for (int i = 0; i < y.size() - 1; i++) {
      R[i + size() - 1] = y[i];
    }
    return R;
  }

 public:
  /*
  Gdims(Bifstream& ifs){
    int _k=ifs.get<int>();
    resize(_k);
    for(int i=0; i<_k; i++)
      (*this)[i]=ifs.get<int>();
  }

  void serialize(Bofstream& ofs) const{
    const int k=size();
    ofs.write(k);
    for(int i=0; i<k; i++)
      ofs.write((*this)[i]);
  }
  */

  string str() const {
    ostringstream oss;
    int k = size();
    oss << "(";
    for (int i = 0; i < k; i++) {
      oss << (*this)[i];
      if (i < k - 1) {
        oss << ",";
      }
    }
    oss << ")";
    return oss.str();
  }

  friend ostream& operator<<(ostream& stream, const Gdims& x) {
    stream << x.str();
    return stream;
  }
};

inline Gdims dims(const int i0) {
  return Gdims(i0);
}
inline Gdims dims(const int i0, const int i1) {
  return Gdims(i0, i1);
}
inline Gdims dims(const int i0, const int i1, const int i2) {
  return Gdims(i0, i1, i2);
}
inline Gdims dims(const int i0, const int i1, const int i2, const int i3) {
  return Gdims(i0, i1, i2, i3);
}
inline Gdims dims(const int i0,
                  const int i1,
                  const int i2,
                  const int i3,
                  const int i4) {
  return Gdims(i0, i1, i2, i3, i4);
}

}  // namespace Cengine

namespace std {

template <>
struct hash<Cengine::Gdims> {
 public:
  size_t operator()(const Cengine::Gdims& dims) const {
    size_t t = 0;
    for (int i = 0; i < dims.size(); i++) {
      t = (t ^ hash<int>()(dims[i])) << 1;
    }
    return t;
  }
};

}  // namespace std

#endif
