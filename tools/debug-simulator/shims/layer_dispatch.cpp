#include "../../headers/layer_dispatch.h"
#include "python_bridge.h"

#include <cstdio>
#include <cstdlib>

#include "../../headers/analog_operations.h"

extern "C" Tensor2DF32 analog_run_layer_2d(float *allocated, float *aligned,
                                           int64_t offset, int64_t size0,
                                           int64_t size1, int64_t stride0,
                                           int64_t stride1, int32_t layerId);

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

Tensor2DF32 lastDispatchedLayer2dResult = {
    nullptr, nullptr, -1, {-1, -1}, {-1, -1}};

}


extern "C" void analog_dispatch_layer_2d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t stride0,
                                         int64_t stride1, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_2d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
  lastDispatchedLayer2dResult =
      analog_run_layer_2d(allocated, aligned, offset, size0, size1, stride0,
                          stride1, layerId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_2d");
}

extern "C" void analog_dispatch_layer_3d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_3d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
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
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
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
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
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
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_2d");
  return lastDispatchedLayer2dResult;
}

extern "C" Tensor3DF32 analog_wait_layers_3d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_3d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  Tensor3DF32 result;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_3d");
  return result;
}

extern "C" Tensor4DF32 analog_wait_layers_4d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_4d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  Tensor4DF32 result;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_4d");
  return result;
}

extern "C" Tensor5DF32 analog_wait_layers_5d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_5d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  Tensor5DF32 result;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_5d");
  return result;
}
