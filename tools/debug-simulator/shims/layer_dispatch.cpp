#include "../../headers/layer_dispatch.h"
#include "python_bridge.h"

#include <cstdio>
#include <cstdlib>

extern "C" void analog_run_layer_2d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t stride0, int64_t stride1, float *outAllocated,
    float *outAligned, int64_t outOffset, int64_t outSize0, int64_t outSize1,
    int64_t outStride0, int64_t outStride1, int32_t layerId);
extern "C" void analog_run_layer_3d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t stride0, int64_t stride1,
    int64_t stride2, float *outAllocated, float *outAligned,
    int64_t outOffset, int64_t outSize0, int64_t outSize1, int64_t outSize2,
    int64_t outStride0, int64_t outStride1, int64_t outStride2,
    int32_t layerId);
extern "C" void analog_run_layer_4d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t size3, int64_t stride0,
    int64_t stride1, int64_t stride2, int64_t stride3, float *outAllocated,
    float *outAligned, int64_t outOffset, int64_t outSize0, int64_t outSize1,
    int64_t outSize2, int64_t outSize3, int64_t outStride0,
    int64_t outStride1, int64_t outStride2, int64_t outStride3,
    int32_t layerId);
extern "C" void analog_run_layer_5d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, float *outAllocated, float *outAligned,
    int64_t outOffset, int64_t outSize0, int64_t outSize1, int64_t outSize2,
    int64_t outSize3, int64_t outSize4, int64_t outStride0,
    int64_t outStride1, int64_t outStride2, int64_t outStride3,
    int64_t outStride4, int32_t layerId);

extern "C" __attribute__((weak)) void analog_run_layer_3d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t stride0, int64_t stride1,
    int64_t stride2, float *outAllocated, float *outAligned,
    int64_t outOffset, int64_t outSize0, int64_t outSize1, int64_t outSize2,
    int64_t outStride0, int64_t outStride1, int64_t outStride2,
    int32_t layerId) {
  (void)allocated;
  (void)aligned;
  (void)offset;
  (void)size0;
  (void)size1;
  (void)size2;
  (void)stride0;
  (void)stride1;
  (void)stride2;
  (void)outAllocated;
  (void)outAligned;
  (void)outOffset;
  (void)outSize0;
  (void)outSize1;
  (void)outSize2;
  (void)outStride0;
  (void)outStride1;
  (void)outStride2;
  (void)layerId;
}

extern "C" __attribute__((weak)) void analog_run_layer_4d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t size3, int64_t stride0,
    int64_t stride1, int64_t stride2, int64_t stride3, float *outAllocated,
    float *outAligned, int64_t outOffset, int64_t outSize0, int64_t outSize1,
    int64_t outSize2, int64_t outSize3, int64_t outStride0,
    int64_t outStride1, int64_t outStride2, int64_t outStride3,
    int32_t layerId) {
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
  (void)outAllocated;
  (void)outAligned;
  (void)outOffset;
  (void)outSize0;
  (void)outSize1;
  (void)outSize2;
  (void)outSize3;
  (void)outStride0;
  (void)outStride1;
  (void)outStride2;
  (void)outStride3;
  (void)layerId;
}

extern "C" __attribute__((weak)) void analog_run_layer_5d(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, float *outAllocated, float *outAligned,
    int64_t outOffset, int64_t outSize0, int64_t outSize1, int64_t outSize2,
    int64_t outSize3, int64_t outSize4, int64_t outStride0,
    int64_t outStride1, int64_t outStride2, int64_t outStride3,
    int64_t outStride4, int32_t layerId) {
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
  (void)outAllocated;
  (void)outAligned;
  (void)outOffset;
  (void)outSize0;
  (void)outSize1;
  (void)outSize2;
  (void)outSize3;
  (void)outSize4;
  (void)outStride0;
  (void)outStride1;
  (void)outStride2;
  (void)outStride3;
  (void)outStride4;
  (void)layerId;
}

#ifdef DEBUG_MODE
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName)                                      \
  std::printf("[debug-simulator shim] enter %s\n", fnName)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName)                                 \
  std::printf("[debug-simulator shim] exit %s\n", fnName)
#else
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName) ((void)0)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName) ((void)0)
#endif

extern "C" void analog_dispatch_layer_2d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t stride0,
                                         int64_t stride1, float *outAllocated,
                                         float *outAligned, int64_t outOffset,
                                         int64_t outSize0, int64_t outSize1,
                                         int64_t outStride0,
                                         int64_t outStride1, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_2d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
  analog_run_layer_2d(allocated, aligned, offset, size0, size1, stride0,
                      stride1, outAllocated, outAligned, outOffset, outSize0,
                      outSize1, outStride0, outStride1, layerId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_2d");
}

extern "C" void analog_dispatch_layer_3d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, float *outAllocated,
                                         float *outAligned, int64_t outOffset,
                                         int64_t outSize0, int64_t outSize1,
                                         int64_t outSize2,
                                         int64_t outStride0,
                                         int64_t outStride1,
                                         int64_t outStride2, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_3d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
  analog_run_layer_3d(allocated, aligned, offset, size0, size1, size2,
                      stride0, stride1, stride2, outAllocated, outAligned,
                      outOffset, outSize0, outSize1, outSize2, outStride0,
                      outStride1, outStride2, layerId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_3d");
}

extern "C" void analog_dispatch_layer_4d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t size3, int64_t stride0,
                                         int64_t stride1, int64_t stride2,
                                         int64_t stride3, float *outAllocated,
                                         float *outAligned, int64_t outOffset,
                                         int64_t outSize0, int64_t outSize1,
                                         int64_t outSize2, int64_t outSize3,
                                         int64_t outStride0,
                                         int64_t outStride1,
                                         int64_t outStride2,
                                         int64_t outStride3, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_4d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
  analog_run_layer_4d(allocated, aligned, offset, size0, size1, size2, size3,
                      stride0, stride1, stride2, stride3, outAllocated,
                      outAligned, outOffset, outSize0, outSize1, outSize2,
                      outSize3, outStride0, outStride1, outStride2,
                      outStride3, layerId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_4d");
}

extern "C" void analog_dispatch_layer_5d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t size3, int64_t size4,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, int64_t stride3,
                                         int64_t stride4, float *outAllocated,
                                         float *outAligned, int64_t outOffset,
                                         int64_t outSize0, int64_t outSize1,
                                         int64_t outSize2, int64_t outSize3,
                                         int64_t outSize4,
                                         int64_t outStride0,
                                         int64_t outStride1,
                                         int64_t outStride2,
                                         int64_t outStride3,
                                         int64_t outStride4, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_5d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
  analog_run_layer_5d(allocated, aligned, offset, size0, size1, size2, size3,
                      size4, stride0, stride1, stride2, stride3, stride4,
                      outAllocated, outAligned, outOffset, outSize0, outSize1,
                      outSize2, outSize3, outSize4, outStride0, outStride1,
                      outStride2, outStride3, outStride4, layerId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_5d");
}

extern "C" void analog_wait_layers_2d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_2d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_2d");
}

extern "C" void analog_wait_layers_3d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_3d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_3d");
}

extern "C" void analog_wait_layers_4d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_4d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_4d");
}

extern "C" void analog_wait_layers_5d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_5d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_5d");
}
