#ifndef ANALOG_MLIR_HEADERS_LAYER_DISPATCH_MULTI_INPUT_H
#define ANALOG_MLIR_HEADERS_LAYER_DISPATCH_MULTI_INPUT_H

#include "tensor_types.h"

/*
  analog_run_layer_2d_from_3d_2d

  Synchronously executes a generated multi-input layer routine with the current
  supported signature:
  - operand 0: rank-3 tensor
  - operand 1: rank-2 tensor
  - result: rank-2 tensor

  This ABI shape is currently used by full-sequence RNN dispatch.
*/
extern "C" void analog_run_layer_2d_from_3d_2d(
    float *allocated0, float *aligned0, int64_t offset0, int64_t size00,
    int64_t size01, int64_t size02, int64_t stride00, int64_t stride01,
    int64_t stride02, float *allocated1, float *aligned1, int64_t offset1,
    int64_t size10, int64_t size11, int64_t stride10, int64_t stride11,
    float *outAllocated, float *outAligned, int64_t outOffset,
    int64_t outSize0, int64_t outSize1, int64_t outStride0,
    int64_t outStride1, int32_t layerId);

/*
  analog_dispatch_layer_2d_from_3d_2d

  Asynchronously dispatches a generated multi-input layer routine with the
  signature (3D, 2D) -> 2D and writes its result into the provided output
  buffer. Completion is synchronized via
  analog_wait_layers_2d_from_3d_2d().
*/
extern "C" void analog_dispatch_layer_2d_from_3d_2d(
    float *allocated0, float *aligned0, int64_t offset0, int64_t size00,
    int64_t size01, int64_t size02, int64_t stride00, int64_t stride01,
    int64_t stride02, float *allocated1, float *aligned1, int64_t offset1,
    int64_t size10, int64_t size11, int64_t stride10, int64_t stride11,
    float *outAllocated, float *outAligned, int64_t outOffset,
    int64_t outSize0, int64_t outSize1, int64_t outStride0,
    int64_t outStride1, int32_t layerId);

/*
  analog_wait_layers_2d_from_3d_2d

  Blocks until all outstanding multi-input layer dispatches with signature
  (3D, 2D) -> 2D have completed.
*/
extern "C" void analog_wait_layers_2d_from_3d_2d();

#endif // ANALOG_MLIR_HEADERS_LAYER_DISPATCH_MULTI_INPUT_H
