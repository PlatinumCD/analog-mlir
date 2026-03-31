#ifndef ANALOG_MLIR_HEADERS_LAYER_DISPATCH_MULTI_INPUT_H
#define ANALOG_MLIR_HEADERS_LAYER_DISPATCH_MULTI_INPUT_H

#include <cstdint>

/*
  analog_run_weight

  Synchronously executes the generated weight routine identified by weightId.
*/
extern "C" void analog_run_weight(int32_t weightId);
/*
  analog_dispatch_weight

  Asynchronously dispatches the generated weight routine identified by
  weightId. Completion is synchronized via analog_wait_weights().
*/
extern "C" void analog_dispatch_weight(int32_t weightId);
/*
  analog_wait_weights

  Blocks until all outstanding asynchronous weight dispatches have completed.
*/
extern "C" void analog_wait_weights();

#endif // ANALOG_MLIR_HEADERS_LAYER_DISPATCH_MULTI_INPUT_H
