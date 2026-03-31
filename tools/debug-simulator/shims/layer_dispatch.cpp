#include "../../headers/layer_dispatch.h"

#include <cstdio>

namespace {

#ifdef DEBUG_MODE
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName)                                      \
  std::printf("[debug-simulator shim] enter %s\n", fnName)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName)                                 \
  std::printf("[debug-simulator shim] exit %s\n", fnName)
#else
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName) ((void)0)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName) ((void)0)
#endif


template <typename TensorT>
TensorT makeInvalidTensor();

template <>
Tensor2DF32 makeInvalidTensor<Tensor2DF32>() {
  return Tensor2DF32{nullptr, nullptr, -1, {-1, -1}, {-1, -1}};
}

template <>
Tensor3DF32 makeInvalidTensor<Tensor3DF32>() {
  return Tensor3DF32{nullptr, nullptr, -1, {-1, -1, -1}, {-1, -1, -1}};
}

template <>
Tensor4DF32 makeInvalidTensor<Tensor4DF32>() {
  return Tensor4DF32{nullptr, nullptr, -1, {-1, -1, -1, -1},
                     {-1, -1, -1, -1}};
}

template <>
Tensor5DF32 makeInvalidTensor<Tensor5DF32>() {
  return Tensor5DF32{nullptr, nullptr, -1, {-1, -1, -1, -1, -1},
                     {-1, -1, -1, -1, -1}};
}

} // namespace

extern "C" __attribute__((weak)) Tensor2DF32
analog_run_layer_2d(float *allocated, float *aligned,
                   int64_t offset, int64_t size0,
                   int64_t size1, int64_t stride0,
                   int64_t stride1, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_run_layer_2d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)stride0;
  (void)stride1;
  (void)layerId;
  Tensor2DF32 result = makeInvalidTensor<Tensor2DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_layer_2d");
  return result;
}

extern "C" __attribute__((weak)) Tensor3DF32
analog_run_layer_3d(float *allocated, float *aligned,
                   int64_t offset, int64_t size0,
                   int64_t size1, int64_t size2,
                   int64_t stride0, int64_t stride1,
                   int64_t stride2, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_run_layer_3d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)layerId;
  Tensor3DF32 result = makeInvalidTensor<Tensor3DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_layer_3d");
  return result;
}

extern "C" __attribute__((weak)) Tensor4DF32
analog_run_layer_4d(float *allocated, float *aligned,
                   int64_t offset, int64_t size0,
                   int64_t size1, int64_t size2,
                   int64_t size3, int64_t stride0,
                   int64_t stride1, int64_t stride2,
                   int64_t stride3, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_run_layer_4d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)size3;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)stride3;
  (void)layerId;
  Tensor4DF32 result = makeInvalidTensor<Tensor4DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_layer_4d");
  return result;
}

extern "C" __attribute__((weak)) Tensor5DF32
analog_run_layer_5d(float *allocated, float *aligned,
                   int64_t offset, int64_t size0,
                   int64_t size1, int64_t size2,
                   int64_t size3, int64_t size4,
                   int64_t stride0, int64_t stride1,
                   int64_t stride2, int64_t stride3,
                   int64_t stride4, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_run_layer_5d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)size3;
  (void)size4;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)stride3;
  (void)stride4;
  (void)layerId;
  Tensor5DF32 result = makeInvalidTensor<Tensor5DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_layer_5d");
  return result;
}

extern "C" void analog_dispatch_layer_2d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t stride0,
                                         int64_t stride1, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_2d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)stride0;
  (void)stride1;
  (void)layerId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_2d");
}

extern "C" void analog_dispatch_layer_3d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_3d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)layerId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_3d");
}

extern "C" void analog_dispatch_layer_4d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t size3, int64_t stride0,
                                         int64_t stride1, int64_t stride2,
                                         int64_t stride3, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_4d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)size3;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)stride3;
  (void)layerId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_4d");
}

extern "C" void analog_dispatch_layer_5d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t size3, int64_t size4,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, int64_t stride3,
                                         int64_t stride4, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_5d");
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)size3;
  (void)size4;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)stride3;
  (void)stride4;
  (void)layerId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_5d");
}

extern "C" Tensor2DF32 analog_wait_layers_2d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_2d");
  Tensor2DF32 result = makeInvalidTensor<Tensor2DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_2d");
  return result;
}

extern "C" Tensor3DF32 analog_wait_layers_3d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_3d");
  Tensor3DF32 result = makeInvalidTensor<Tensor3DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_3d");
  return result;
}

extern "C" Tensor4DF32 analog_wait_layers_4d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_4d");
  Tensor4DF32 result = makeInvalidTensor<Tensor4DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_4d");
  return result;
}

extern "C" Tensor5DF32 analog_wait_layers_5d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_5d");
  Tensor5DF32 result = makeInvalidTensor<Tensor5DF32>();
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_5d");
  return result;
}
