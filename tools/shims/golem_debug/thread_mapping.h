#ifndef ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_THREAD_MAPPING_H
#define ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_THREAD_MAPPING_H

#include <cstdint>

// Shared runtime orchestration policy:
// maps logical task ids to worker slots.
int32_t mapTaskToWorkerSlot(int32_t taskId);

// Runtime orchestration policy:
// maps compiler-provided weight ids to logical worker slots.
int32_t mapWeightToWorkerSlot(int32_t weightId);

// Runtime orchestration policy:
// maps compiler-provided layer ids to logical worker slots.
int32_t mapLayerToWorkerSlot(int32_t layerId);

// Per-thread runtime context used by the shim layer.
void setCurrentWorkerSlot(int32_t workerSlot);
int32_t getCurrentWorkerSlot();

#endif // ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_THREAD_MAPPING_H
