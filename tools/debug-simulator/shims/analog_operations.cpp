#include "../../headers/analog_operations.h"

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

extern "C" void golem_debug_mvm_set(void *data, int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_set");
  (void)data;
  (void)rawArrayId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_set");
}

extern "C" void golem_debug_mvm_load(void *data, int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_load");
  (void)data;
  (void)rawArrayId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_load");
}

extern "C" void golem_debug_mvm_compute(int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_compute");
  (void)rawArrayId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_compute");
}

extern "C" void golem_debug_mvm_store(void *data, int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_store");
  (void)data;
  (void)rawArrayId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_store");
}
