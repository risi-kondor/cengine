#ifndef _RscalarInterface
#define _RscalarInterface

#include "Cengine.hpp"
#include "InterfaceBase.hpp"
#include "RscalarB_ops.hpp"

extern ::Cengine::Cengine* Cscalar_engine;

namespace Cengine {

namespace engine {

Chandle* new_rscalar(const int nbd = -1, const int device = 0) {
  Cnode* node = Cengine_engine->enqueue(new new_rscalar_op(nbd, device));
  return new_handle(node);
}

Chandle* new_rscalar_zero(const int nbd = -1, const int device = 0) {
  Cnode* node = Cengine_engine->enqueue(new new_rscalar_zero_op(nbd, device));
  return new_handle(node);
}

Chandle* new_rscalar_gaussian(const int nbd = -1, const int device = 0) {
  Cnode* node =
      Cengine_engine->enqueue(new new_rscalar_gaussian_op(nbd, device));
  return new_handle(node);
}

Chandle* new_rscalar_set(const float x,
                         const int nbd = -1,
                         const int device = 0) {
  Cnode* node = Cengine_engine->enqueue(new new_rscalar_set_op(nbd, x, device));
  return new_handle(node);
}

Chandle* rscalar_copy(Chandle* x) {
  Cnode* node = Cengine_engine->enqueue(new rscalar_copy_op(nodeof(x)));
  return new_handle(node);
}

// ---- In-place operations
// ------------------------------------------------------------------------------

Chandle* rscalar_zero(Chandle* r) {
  return new_handle(
      Cengine_engine->enqueue(new rscalar_set_zero_op(nodeof(r))));
}

// ---- Cumulative operations
// ----------------------------------------------------------------------------

Chandle* rscalar_add(Chandle* r, Chandle* x) {
  return new_handle(
      Cengine_engine->enqueue(new rscalar_add_op(nodeof(r), nodeof(x))));
}

Chandle* rscalar_add_times_real(Chandle* r, Chandle* x, float c) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_times_real_op(nodeof(r), nodeof(x), c)));
}

Chandle* rscalar_subtract(Chandle* r, Chandle* x) {
  return new_handle(
      Cengine_engine->enqueue(new rscalar_subtract_op(nodeof(r), nodeof(x))));
}

Chandle* rscalar_add_prod(Chandle* r, Chandle* x, Chandle* y) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_prod_op(nodeof(r), nodeof(x), nodeof(y))));
}

Chandle* rscalar_add_div(Chandle* r, Chandle* x, Chandle* y) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_div_op(nodeof(r), nodeof(x), nodeof(y))));
}

Chandle* rscalar_add_div_back0(Chandle* r, Chandle* g, Chandle* y) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_div_back0_op(nodeof(r), nodeof(g), nodeof(y))));
}

Chandle* rscalar_add_div_back1(Chandle* r, Chandle* g, Chandle* x, Chandle* y) {
  return new_handle(Cengine_engine->enqueue(new rscalar_add_div_back1_op(
      nodeof(r), nodeof(g), nodeof(x), nodeof(y))));
}

Chandle* rscalar_add_abs(Chandle* r, Chandle* x) {
  return new_handle(
      Cengine_engine->enqueue(new rscalar_add_abs_op(nodeof(r), nodeof(x))));
}

Chandle* rscalar_add_abs_back(Chandle* r, Chandle* g, Chandle* x) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_abs_back_op(nodeof(r), nodeof(g), nodeof(x))));
}

Chandle* rscalar_add_pow(Chandle* r, Chandle* x, float p, float c) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_pow_op(nodeof(r), nodeof(x), p, c)));
}

Chandle* rscalar_add_exp(Chandle* r, Chandle* x) {
  return new_handle(
      Cengine_engine->enqueue(new rscalar_add_exp_op(nodeof(r), nodeof(x))));
}

Chandle* rscalar_add_ReLU(Chandle* r, Chandle* x, const float c) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_ReLU_op(nodeof(r), nodeof(x), c)));
}

Chandle* rscalar_add_ReLU_back(Chandle* r,
                               Chandle* g,
                               Chandle* x,
                               const float c) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_ReLU_back_op(nodeof(r), nodeof(g), nodeof(x), c)));
}

Chandle* rscalar_add_sigmoid(Chandle* r, Chandle* x) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_sigmoid_op(nodeof(r), nodeof(x))));
}

Chandle* rscalar_add_sigmoid_back(Chandle* r, Chandle* g, Chandle* x) {
  return new_handle(Cengine_engine->enqueue(
      new rscalar_add_sigmoid_back_op(nodeof(r), nodeof(g), nodeof(x))));
}

// ---- Output operations
// -------------------------------------------------------------------------

}  // namespace engine

}  // namespace Cengine

#endif
