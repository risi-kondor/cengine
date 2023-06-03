#ifndef _CscalarBreducer
#define _CscalarBreducer

#include "CscalarBpack.hpp"

namespace Cengine {

class CscalarBreducer : public CscalarBpack {
 public:
  CscalarB& target;

  CscalarBreducer(const int _N, CscalarB& _target)
      : CscalarBpack(_N), target(_target) {}

  ~CscalarBreducer() {
#ifdef _WITH_CUDA
    target.to_device(1);
    cudaStream_t stream;
    CUDA_SAFE(cudaStreamCreate(&stream));
    // reduce_cu(target,stream);
    CUDA_SAFE(cudaStreamSynchronize(stream));
    CUDA_SAFE(cudaStreamDestroy(stream));
#else
    NOCUDA_ERROR;
#endif
    CUDA_SAFE(cudaMemcpy(&target.val, arrg, 2 * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_SAFE(cudaFree(arrg));
  }
};

}  // namespace Cengine

#endif
