#include "thread_mapping.h"

/*
  mapWeightToWorkerSlot(int32_t weightId)

  Determines the logical worker slot responsible for executing a given
  weight. This function encodes the runtime’s orchestration policy,
  mapping weight identifiers to specific worker indices.

  The current implementation is a fixed example mapping used for
  demonstration and testing purposes.
*/
int32_t mapWeightToWorkerSlot(int32_t weightId) {
  // Example orchestration:
  //   weight 0 -> worker 2
  //   weight 1 -> worker 1
  if (weightId == 0) {
    return 2;
  }
  if (weightId == 1) {
    return 1;
  }
  return 1; // default fallback
}
