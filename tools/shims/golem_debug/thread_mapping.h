#ifndef ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_THREAD_MAPPING_H
#define ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_THREAD_MAPPING_H

#include <cstdint>

// Runtime orchestration policy:
// maps compiler-provided weight ids to logical worker slots.
int32_t mapWeightToWorkerSlot(int32_t weightId);

#endif // ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_THREAD_MAPPING_H
