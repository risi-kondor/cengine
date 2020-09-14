  public: // k=1 special cases

    complex<float> operator()(const int i0) const{
      FCG_ASSERT(device==0,"CFtensor::operator() not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get(const int i0) const{
      FCG_ASSERT(device==0,"CFtensor::get not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const complex<float> x){
      FCG_ASSERT(device==0,"CFtensor::set not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const complex<float> x){
      FCG_ASSERT(device==0,"CFtensor::inc not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // k=2 special cases


    complex<float> operator()(const int i0, const int i1) const{
      FCG_ASSERT(device==0,"CFtensor::operator() not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get(const int i0, const int i1) const{
      FCG_ASSERT(device==0,"CFtensor::get not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return complex<float>(arr[t],arr[t+cst]);
    }

    void set(const int i0, const int i1, const complex<float> x){
      FCG_ASSERT(device==0,"CFtensor::set not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const complex<float> x){
      FCG_ASSERT(device==0,"CFtensor::inc not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // k=3 special cases


    complex<float> operator()(const int i0, const int i1, const int i2) const{
      FCG_ASSERT(device==0, "CFtensor::operator() not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arr[t+cst]);
    }

    complex<float> get(const int i0, const int i1, const int i2) const{
      FCG_ASSERT(device==0, "CFtensor::get not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arr[t+cst]);
    }

    void set(const int i0, const int i1, const int i2, const complex<float> x){
      FCG_ASSERT(device==0, "CFtensor::set not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const complex<float> x){
      FCG_ASSERT(device==0, "CFtensor::inc not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


  public: // k=4 special cases


    complex<float> operator()(const int i0, const int i1, const int i2, const int i3) const{
      FCG_ASSERT(device==0, "CFtensor::operator() not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];
      return complex<float>(arr[t],arr[t+cst]);
    }

    complex<float> get(const int i0, const int i1, const int i2, const int i3) const{
      FCG_ASSERT(device==0, "CFtensor::get not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];
      return complex<float>(arr[t],arr[t+cst]);
    }

    void set(const int i0, const int i1, const int i2, const int i3, const complex<float> x){
      FCG_ASSERT(device==0, "CFtensor::set not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const complex<float> x){
      FCG_ASSERT(device==0, "CFtensor::inc not implemented for GPU.\n");
      assert(k==4);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }


