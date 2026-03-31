#ifndef ANALOG_MLIR_HEADERS_PYTHON_WEIGHT_BRIDGE_H
#define ANALOG_MLIR_HEADERS_PYTHON_WEIGHT_BRIDGE_H

#include <cstdint>

/*
  analog_debug_python_bridge_initialize

  Initializes the debug-simulator Python bridge for weight dispatch.
*/
bool analog_debug_python_bridge_initialize();

/*
  analog_debug_python_bridge_bind_current_test

  Binds the Python bridge to the current test's shared object, derived from the
  running executable name.
*/
bool analog_debug_python_bridge_bind_current_test();

/*
  analog_debug_python_bridge_dispatch_weight

  Forwards one weight-dispatch request to the Python bridge.
*/
bool analog_debug_python_bridge_dispatch_weight(int32_t weightId);

/*
  analog_debug_python_bridge_wait_weights

  Drains all queued weight-dispatch requests through the Python bridge.
*/
bool analog_debug_python_bridge_wait_weights();

#endif // ANALOG_MLIR_HEADERS_PYTHON_WEIGHT_BRIDGE_H
