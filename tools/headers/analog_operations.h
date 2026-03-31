#ifndef ANALOG_MLIR_HEADERS_ANALOG_OPERATIONS_H
#define ANALOG_MLIR_HEADERS_ANALOG_OPERATIONS_H

#include <cstdint>

/*
  golem_debug_mvm_set

  Programs a simulated analog array with matrix data from host memory.
*/
extern "C" void golem_debug_mvm_set(void *data, int32_t rawArrayId);

/*
  golem_debug_mvm_load

  Loads a simulated analog array input vector from host memory.
*/
extern "C" void golem_debug_mvm_load(void *data, int32_t rawArrayId);

/*
  golem_debug_mvm_compute

  Executes the simulated analog array compute step for the specified array.
*/
extern "C" void golem_debug_mvm_compute(int32_t rawArrayId);

/*
  golem_debug_mvm_store

  Stores the simulated analog array output vector back to host memory.
*/
extern "C" void golem_debug_mvm_store(void *data, int32_t rawArrayId);

#endif // ANALOG_MLIR_HEADERS_ANALOG_OPERATIONS_H
