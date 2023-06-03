public:  // k=1 special cases
float operator()(const int i0) const {
  FCG_ASSERT(device == 0, "RFtensor::operator() not implemented for GPU.\n");
  assert(k == 1);
  int t = i0 * strides[0];
  return arr[t];
}

float get(const int i0) const {
  FCG_ASSERT(device == 0, "RFtensor::get not implemented for GPU.\n");
  assert(k == 1);
  int t = i0 * strides[0];
  return arr[t];
}

void set(const int i0, const float x) {
  FCG_ASSERT(device == 0, "RFtensor::set not implemented for GPU.\n");
  assert(k == 1);
  int t = i0 * strides[0];
  arr[t] = x;
}

void inc(const int i0, const float x) {
  FCG_ASSERT(device == 0, "RFtensor::inc not implemented for GPU.\n");
  assert(k == 1);
  int t = i0 * strides[0];
  arr[t] += x;
}

public:  // k=2 special cases
float operator()(const int i0, const int i1) const {
  FCG_ASSERT(device == 0, "RFtensor::operator() not implemented for GPU.\n");
  assert(k == 2);
  int t = i0 * strides[0] + i1 * strides[1];
  return arr[t];
}

float get(const int i0, const int i1) const {
  FCG_ASSERT(device == 0, "RFtensor::get not implemented for GPU.\n");
  assert(k == 2);
  int t = i0 * strides[0] + i1 * strides[1];
  return arr[t];
}

void set(const int i0, const int i1, const float x) {
  FCG_ASSERT(device == 0, "RFtensor::set not implemented for GPU.\n");
  assert(k == 2);
  int t = i0 * strides[0] + i1 * strides[1];
  arr[t] = x;
}

void inc(const int i0, const int i1, const float x) {
  FCG_ASSERT(device == 0, "RFtensor::inc not implemented for GPU.\n");
  assert(k == 2);
  int t = i0 * strides[0] + i1 * strides[1];
  arr[t] += x;
}

public:  // k=3 special cases
float operator()(const int i0, const int i1, const int i2) const {
  FCG_ASSERT(device == 0, "RFtensor::operator() not implemented for GPU.\n");
  assert(k == 3);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2];
  return arr[t];
}

float get(const int i0, const int i1, const int i2) const {
  FCG_ASSERT(device == 0, "RFtensor::get not implemented for GPU.\n");
  assert(k == 3);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2];
  return arr[t];
}

void set(const int i0, const int i1, const int i2, const float x) {
  FCG_ASSERT(device == 0, "RFtensor::set not implemented for GPU.\n");
  assert(k == 3);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2];
  arr[t] = x;
}

void inc(const int i0, const int i1, const int i2, const float x) {
  FCG_ASSERT(device == 0, "RFtensor::inc not implemented for GPU.\n");
  assert(k == 3);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2];
  arr[t] += x;
}

public:  // k=4 special cases
float operator()(const int i0, const int i1, const int i2, const int i3) const {
  FCG_ASSERT(device == 0, "RFtensor::operator() not implemented for GPU.\n");
  assert(k == 4);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
  return arr[t];
}

float get(const int i0, const int i1, const int i2, const int i3) const {
  FCG_ASSERT(device == 0, "RFtensor::get not implemented for GPU.\n");
  assert(k == 4);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
  return arr[t];
}

void set(const int i0,
         const int i1,
         const int i2,
         const int i3,
         const float x) {
  FCG_ASSERT(device == 0, "RFtensor::set not implemented for GPU.\n");
  assert(k == 4);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
  arr[t] = x;
}

void inc(const int i0,
         const int i1,
         const int i2,
         const int i3,
         const float x) {
  FCG_ASSERT(device == 0, "RFtensor::inc not implemented for GPU.\n");
  assert(k == 4);
  int t = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
  arr[t] += x;
}
