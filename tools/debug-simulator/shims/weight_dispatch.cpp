#include "../../headers/weight_dispatch.h"
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

namespace {

void ensureBridgeInitialized() {
  if (!analog_debug_python_bridge_initialize()) {
    std::fprintf(stderr,
                 "[debug-simulator shim] bridge initialization failed\n");
    std::abort();
  }
}

} // namespace

/*extern "C" __attribute__((weak)) void analog_run_weight(int32_t weightId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_run_weight");
  (void)weightId;
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_run_weight");
}
*/

extern "C" void analog_dispatch_weight(int32_t weightId) {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_dispatch_weight");
  ensureBridgeInitialized();
  if (!analog_debug_python_bridge_dispatch_weight(weightId)) {
    std::fprintf(stderr,
                 "[debug-simulator shim] analog_dispatch_weight failed for weight %d\n",
                 static_cast<int>(weightId));
    std::abort();
  }
  analog_run_weight(weightId);
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_dispatch_weight");
}

extern "C" void analog_wait_weights() {
  ANALOG_DEBUG_SIM_SHIM_TRACE("analog_wait_weights");
  if (!analog_debug_python_bridge_wait_weights()) {
    std::fprintf(stderr,
                 "[debug-simulator shim] analog_wait_weights failed\n");
    std::abort();
  }
  ANALOG_DEBUG_SIM_SHIM_TRACE_EXIT("analog_wait_weights");
}
