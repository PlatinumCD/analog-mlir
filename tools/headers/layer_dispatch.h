#ifndef ANALOG_MLIR_HEADERS_LAYER_DISPATCH_H
#define ANALOG_MLIR_HEADERS_LAYER_DISPATCH_H

#include "tensor_types.h"

/*
  analog_dispatch_layer_2d

  Asynchronously dispatches a generated single-input layer routine with a
  rank-2 tensor operand. Completion is synchronized via analog_wait_layers_2d().
*/
extern "C" void analog_dispatch_layer_2d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t stride0,
                                         int64_t stride1, int32_t layerId);
/*
  analog_dispatch_layer_3d

  Asynchronously dispatches a generated single-input layer routine with a
  rank-3 tensor operand. Completion is synchronized via analog_wait_layers_3d().
*/
extern "C" void analog_dispatch_layer_3d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, int32_t layerId);
/*
  analog_dispatch_layer_4d

  Asynchronously dispatches a generated single-input layer routine with a
  rank-4 tensor operand. Completion is synchronized via analog_wait_layers_4d().
*/
extern "C" void analog_dispatch_layer_4d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t size3, int64_t stride0,
                                         int64_t stride1, int64_t stride2,
                                         int64_t stride3, int32_t layerId);
/*
  analog_dispatch_layer_5d

  Asynchronously dispatches a generated single-input layer routine with a
  rank-5 tensor operand. Completion is synchronized via analog_wait_layers_5d().
*/
extern "C" void analog_dispatch_layer_5d(float *allocated, float *aligned,
                                         int64_t offset, int64_t size0,
                                         int64_t size1, int64_t size2,
                                         int64_t size3, int64_t size4,
                                         int64_t stride0, int64_t stride1,
                                         int64_t stride2, int64_t stride3,
                                         int64_t stride4, int32_t layerId);

/*
  analog_wait_layers_2d

  Blocks until all outstanding rank-2 single-input layer dispatches have
  completed and returns the resulting rank-2 tensor.
*/
extern "C" Tensor2DF32 analog_wait_layers_2d();
/*
  analog_wait_layers_3d

  Blocks until all outstanding rank-3 single-input layer dispatches have
  completed and returns the resulting rank-3 tensor.
*/
extern "C" Tensor3DF32 analog_wait_layers_3d();
/*
  analog_wait_layers_4d

  Blocks until all outstanding rank-4 single-input layer dispatches have
  completed and returns the resulting rank-4 tensor.
*/
extern "C" Tensor4DF32 analog_wait_layers_4d();
/*
  analog_wait_layers_5d

  Blocks until all outstanding rank-5 single-input layer dispatches have
  completed and returns the resulting rank-5 tensor.
*/
extern "C" Tensor5DF32 analog_wait_layers_5d();

#endif // ANALOG_MLIR_HEADERS_LAYER_DISPATCH_H
