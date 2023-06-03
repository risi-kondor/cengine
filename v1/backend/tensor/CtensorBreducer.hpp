#ifndef _CtensorBreducer
#define _CtensorBreducer

#include "CtensorBpack.hpp"

namespace Cengine {

class CtensorBreducer : public CtensorBpack {
 public:
  CtensorB& target;

  CtensorBreducer(const int _N, CtensorB& _target)
      : CtensorBpack(_N, _target.dims, _target.nbu, 1), target(_target) {
    N = _N;
    int cst = target.cst;
    int memsize = target.memsize;
    CUDA_SAFE(cudaMalloc((void**)&arrg, memsize * N * sizeof(float)));
    CUDA_SAFE(cudaMemset(arrg, 0, N * memsize * sizeof(float)));
    arrgc = arrg + target.cst;

    float* arr[N];
    float* arrc[N];
    for (int i = 0; i < N; i++) {
      arr[i] = arrg + i * memsize;
      arrc[i] = arrgc + i * memsize;
    }

    // CUDA_SAFE(cudaMalloc((void ***)&parr, N*sizeof(float*)));
    // CUDA_SAFE(cudaMalloc((void ***)&parrc, N*sizeof(float*)));
    CUDA_SAFE(
        cudaMemcpy(parr, arr, N * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_SAFE(
        cudaMemcpy(parrc, arrc, N * sizeof(float*), cudaMemcpyHostToDevice));

    parr_valid = true;
  }

  ~CtensorBreducer() {
#ifdef _WITH_CUDA
    target.to_device(1);
    cudaStream_t stream;
    cudaDeviceSynchronize();
    CUDA_SAFE(cudaStreamCreate(&stream));
    sum_into_cu(target, stream);
    CUDA_SAFE(cudaStreamSynchronize(stream));
    CUDA_SAFE(cudaStreamDestroy(stream));
    cudaDeviceSynchronize();
#else
    NOCUDA_ERROR;
#endif
  }

 public:
};

}  // namespace Cengine

#endif
