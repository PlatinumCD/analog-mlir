#include "thread_mapping.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <vector>

extern "C" void analog_run_weight(int32_t weightId);

struct WeightTask {
  int32_t weightId;
  int32_t workerSlot;
};

static std::vector<pthread_t> weightThreads;


/*
  weightThread(void* context)

  pthread entry point for analog weight execution.

  This function serves as the worker routine for threads created by
  analog_dispatch_weight(). The incoming context is expected to be a
  heap-allocated WeightTask describing which weight to execute and
  which logical worker slot it has been assigned to.

  The routine invokes analog_run_weight(), which is generated from
  MLIR and represents the lowered analog compute kernel for that
  weight. After execution completes, the task object is destroyed and
  the thread exits.

  Ownership of the WeightTask is transferred to this thread.
*/
static void *weightThread(void *context) {
  WeightTask *task = static_cast<WeightTask *>(context);
  std::printf("[dispatch shim] worker[%d] running weight %d\n",
              task->workerSlot,
              task->weightId);

  setCurrentWorkerSlot(task->workerSlot);
  analog_run_weight(task->weightId);
  setCurrentWorkerSlot(-1);

  delete task;
  return nullptr;
}


/*
  analog_dispatch_weight(int32_t weightId)

  Asynchronously dispatches execution of a single analog weight.

  This function creates a new pthread that runs the specified weight
  routine (analog_run_weight). The weight is first mapped to a logical
  worker slot, packaged into a WeightTask, and passed to the thread
  entry point.

  The created thread is recorded in the internal weightThreads list so
  that analog_wait_weights() can later join all outstanding executions.

  This function does not block. Synchronization is the responsibility
  of analog_wait_weights().
*/
extern "C" void analog_dispatch_weight(int32_t weightId) {
  pthread_t thread;
  int32_t workerSlot = mapTaskToWorkerSlot(weightId);
  WeightTask *task = new WeightTask{weightId, workerSlot};

  int rc = pthread_create(&thread, nullptr, &weightThread, task);
  if (rc != 0) {
    std::fprintf(stderr,
                 "pthread_create failed for weight %d (rc=%d)\n",
                 static_cast<int>(weightId),
                 rc);
    delete task;
    std::abort();
  }

  weightThreads.push_back(thread);
}


/*
  analog_wait_weights()

  Runtime barrier for asynchronously dispatched weight executions.

  This function blocks until all weight threads previously launched via
  analog_dispatch_weight() have completed. It joins each outstanding
  pthread and then clears the internal tracking list.

  This is exported from analog-mlir and serves as the synchronization
  point for analog weight execution.
*/
extern "C" void analog_wait_weights() {
  for (pthread_t thread : weightThreads) {
    int rc = pthread_join(thread, nullptr);
    if (rc != 0) {
      std::fprintf(stderr, "pthread_join failed (rc=%d)\n", rc);
      std::abort();
    }
  }
  weightThreads.clear();
}
