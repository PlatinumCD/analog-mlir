#include "../../headers/layer_dispatch_multi_input.h"
#include "python_bridge.h"

#include <cstdio>
#include <cstdlib>

#ifdef DEBUG_MODE
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName)                                      \
  std::printf("[debug-simulator shim] enter %s\n", fnName)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName)                                 \
  std::printf("[debug-simulator shim] exit %s\n", fnName)
#else
#define ANALOG_DEBUG_SIM_SHIM_TRACE(fnName) ((void)0)
#define ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT(fnName) ((void)0)
#endif

extern "C" __attribute__((weak)) void analog_run_layer_2d_from_3d_2d(
    float *allocated0, float *aligned0, int64_t offset0, int64_t size00,
    int64_t size01, int64_t size02, int64_t stride00, int64_t stride01,
    int64_t stride02, float *allocated1, float *aligned1, int64_t offset1,
    int64_t size10, int64_t size11, int64_t stride10, int64_t stride11,
    float *outAllocated, float *outAligned, int64_t outOffset,
    int64_t outSize0, int64_t outSize1, int64_t outStride0,
    int64_t outStride1, int32_t layerId) {
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
  (void)outAllocated;
  (void)outAligned;
  (void)outOffset;
  (void)outSize0;
  (void)outSize1;
  (void)outStride0;
  (void)outStride1;
  (void)layerId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_layer_2d_from_3d_2d");
}

extern "C" __attribute__((weak)) void analog_dispatch_layer_2d_from_3d_2d(
    float *allocated0, float *aligned0, int64_t offset0, int64_t size00,
    int64_t size01, int64_t size02, int64_t stride00, int64_t stride01,
    int64_t stride02, float *allocated1, float *aligned1, int64_t offset1,
    int64_t size10, int64_t size11, int64_t stride10, int64_t stride11,
    float *outAllocated, float *outAligned, int64_t outOffset,
    int64_t outSize0, int64_t outSize1, int64_t outStride0,
    int64_t outStride1, int32_t layerId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_layer_2d_from_3d_2d");
  if (!analog_debug_python_bridge_dispatch_layer(layerId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] dispatch_layer failed for %d\n",
                 static_cast<int>(layerId));
    std::abort();
  }
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
  analog_run_layer_2d_from_3d_2d(
      allocated0, aligned0, offset0, size00, size01, size02, stride00,
      stride01, stride02, allocated1, aligned1, offset1, size10, size11,
      stride10, stride11, outAllocated, outAligned, outOffset, outSize0,
      outSize1, outStride0, outStride1, layerId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_layer_2d_from_3d_2d");
}

extern "C" void analog_wait_layers_2d_from_3d_2d() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_layers_2d_from_3d_2d");
  if (!analog_debug_python_bridge_wait_layers()) {
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_layers_2d_from_3d_2d");
}
