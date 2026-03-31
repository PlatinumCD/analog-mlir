#include <cstdio>
#include <cstdlib>

#include "../../headers/analog_operations.h"
#include "python_bridge.h"

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
  if (!analog_debug_python_bridge_record_mvm_set(data, rawArrayId)) {
    std::fprintf(stderr, "[analog ops] mvm_set failed for array %d\n",
                 static_cast<int>(rawArrayId));
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_set");
}

extern "C" void golem_debug_mvm_load(void *data, int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_load");
  if (!analog_debug_python_bridge_record_mvm_load(data, rawArrayId)) {
    std::fprintf(stderr, "[analog ops] mvm_load failed for array %d\n",
                 static_cast<int>(rawArrayId));
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_load");
}

extern "C" void golem_debug_mvm_compute(int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_compute");
  if (!analog_debug_python_bridge_record_mvm_compute(rawArrayId)) {
    std::fprintf(stderr, "[analog ops] mvm_compute failed for array %d\n",
                 static_cast<int>(rawArrayId));
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_compute");
}

extern "C" void golem_debug_mvm_store(void *data, int32_t rawArrayId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("golem_debug_mvm_store");
  if (!analog_debug_python_bridge_record_mvm_store(data, rawArrayId)) {
    std::fprintf(stderr, "[analog ops] mvm_store failed for array %d\n",
                 static_cast<int>(rawArrayId));
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("golem_debug_mvm_store");
}
