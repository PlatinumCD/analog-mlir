#include "thread_mapping.h"

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

#if NUM_LAYERS <= 0
#error "NUM_LAYERS must be > 0"
#endif

static thread_local int32_t currentWorkerSlot = -1;

/*
  mapTaskToWorkerSlot(int32_t taskId)

  Shared mapping policy used by both weight and layer dispatch shims.
  Update this single function to change worker assignment behavior.
*/
int32_t mapTaskToWorkerSlot(int32_t taskId) {
  int32_t slot = taskId % static_cast<int32_t>(NUM_LAYERS);
  if (slot < 0) {
    slot += static_cast<int32_t>(NUM_LAYERS);
  }
  return slot;
}

/*
  mapWeightToWorkerSlot(int32_t weightId)

  Weight-specific view over the shared task mapping policy.
*/
int32_t mapWeightToWorkerSlot(int32_t weightId) {
  return mapTaskToWorkerSlot(weightId);
}

/*
  mapLayerToWorkerSlot(int32_t layerId)

  Layer-specific view over the shared task mapping policy.
*/
int32_t mapLayerToWorkerSlot(int32_t layerId) {
  return mapTaskToWorkerSlot(layerId);
}

/*
  setCurrentWorkerSlot(int32_t workerSlot)

  Records the logical worker slot for the current thread.
*/
void setCurrentWorkerSlot(int32_t workerSlot) {
  currentWorkerSlot = workerSlot;
}

/*
  getCurrentWorkerSlot()

  Returns the logical worker slot for the current thread.
*/
int32_t getCurrentWorkerSlot() {
  return currentWorkerSlot;
}
