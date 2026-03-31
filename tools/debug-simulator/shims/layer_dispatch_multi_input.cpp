#include "../../headers/layer_dispatch_multi_input.h"

#include <cstdio>

#ifdef DEBUG_MODE
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName)                                      \
  std::printf("[debug-simulator shim] enter %s\n", fnName)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName)                                 \
  std::printf("[debug-simulator shim] exit %s\n", fnName)
#else
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName) ((void)0)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName) ((void)0)
#endif

extern "C" __attribute__((weak)) Tensor2DF32 analog_run_layer_2d_from_3d_2d(
    float *allocated0, float *aligned0, int64_t offset0, int64_t size00,
    int64_t size01, int64_t size02, int64_t stride00, int64_t stride01,
    int64_t stride02, float *allocated1, float *aligned1, int64_t offset1,
    int64_t size10, int64_t size11, int64_t stride10, int64_t stride11,
    int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_run_layer_2d_from_3d_2d");
  (void)allocated0;
  (void)aligned0;
  (void)offset0;
  (void)size00;
  (void)size01;
  (void)size02;
  (void)stride00;
  (void)stride01;
  (void)stride02;
  (void)allocated1;
  (void)aligned1;
  (void)offset1;
  (void)size10;
  (void)size11;
  (void)stride10;
  (void)stride11;
  (void)layerId;
  Tensor2DF32 result = Tensor2DF32{nullptr, nullptr, -1, {-1, -1}, {-1, -1}};
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_layer_2d_from_3d_2d");
  return result;
}

extern "C" __attribute__((weak)) void analog_dispatch_layer_2d_from_3d_2d(
    float *allocated0, float *aligned0, int64_t offset0, int64_t size00,
    int64_t size01, int64_t size02, int64_t stride00, int64_t stride01,
    int64_t stride02, float *allocated1, float *aligned1, int64_t offset1,
    int64_t size10, int64_t size11, int64_t stride10, int64_t stride11,
    int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_2d_from_3d_2d");
  (void)allocated0;
  (void)aligned0;
  (void)offset0;
  (void)size00;
  (void)size01;
  (void)size02;
  (void)stride00;
  (void)stride01;
  (void)stride02;
  (void)allocated1;
  (void)aligned1;
  (void)offset1;
  (void)size10;
  (void)size11;
  (void)stride10;
  (void)stride11;
  (void)layerId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_2d_from_3d_2d");
}

extern "C" Tensor2DF32 analog_wait_layers_2d_from_3d_2d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_2d_from_3d_2d");
  Tensor2DF32 result = Tensor2DF32{nullptr, nullptr, -1, {-1, -1}, {-1, -1}};
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_2d_from_3d_2d");
  return result;
}

