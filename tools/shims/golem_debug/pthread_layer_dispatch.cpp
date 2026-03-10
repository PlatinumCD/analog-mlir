#include "thread_mapping.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <vector>

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

#if NUM_LAYERS <= 0
#error "NUM_LAYERS must be > 0"
#endif

struct Tensor2DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

extern "C" Tensor2DF32 analog_run_layer(float *allocated, float *aligned,
                                        int64_t offset, int64_t size0,
                                        int64_t size1, int64_t stride0,
                                        int64_t stride1, int32_t layerId);

struct LayerTask {
  Tensor2DF32 input;
  int32_t layerId;
  int32_t workerSlot;
  int32_t activeWorkerSlot;
};

static std::vector<pthread_t> layerThreads;

static void printTensorResult(const Tensor2DF32 &tensor) {
  std::printf("[dispatch shim] layer result (%lld x %lld)\n",
              static_cast<long long>(tensor.sizes[0]),
              static_cast<long long>(tensor.sizes[1]));

  for (int64_t row = 0; row < tensor.sizes[0]; ++row) {
    std::printf("  [");
    for (int64_t col = 0; col < tensor.sizes[1]; ++col) {
      int64_t index = tensor.offset + row * tensor.strides[0] +
                      col * tensor.strides[1];
      std::printf("%s%.6f",
                  col == 0 ? "" : ", ",
                  tensor.aligned[index]);
    }
    std::printf("]\n");
  }
}


/*
  layerThread(void* context)

  pthread entry point for analog layer execution.

  This function serves as the worker routine for threads created by
  analog_dispatch_layer(). The incoming context is expected to be a
  heap-allocated LayerTask describing the input tensor, the layer to
  execute, the worker's logical slot, and which worker is responsible
  for actively running the layer.

  Only the active worker invokes analog_run_layer(), which is generated
  from MLIR and represents the lowered analog compute kernel for that
  layer. The remaining workers simply participate in the dispatch/join
  structure and return without running the layer.

  Ownership of the LayerTask is transferred to this thread.
*/
static void *layerThread(void *context) {
  LayerTask *task = static_cast<LayerTask *>(context);

  if (task->workerSlot == task->activeWorkerSlot) {
    std::printf("[dispatch shim] worker[%d] running layer %d\n",
                task->workerSlot,
                task->layerId);

    setCurrentWorkerSlot(task->workerSlot);
    Tensor2DF32 result = analog_run_layer(
        task->input.allocated, task->input.aligned, task->input.offset,
        task->input.sizes[0], task->input.sizes[1], task->input.strides[0],
        task->input.strides[1], task->layerId);
    setCurrentWorkerSlot(-1);

    Tensor2DF32 *heapResult = new Tensor2DF32(result);
    delete task;
    return heapResult;
  } else {
    std::printf("[dispatch shim] worker[%d] waiting for layer %d\n",
                task->workerSlot,
                task->layerId);
  }

  delete task;
  return nullptr;
}


/*
  analog_dispatch_layer(...)

  Asynchronously dispatches execution of a single analog layer.

  This function creates NUM_LAYERS pthreads. The layer is first mapped
  to a logical worker slot, then each worker receives a LayerTask. Only
  the mapped worker actually runs analog_run_layer(); all others simply
  wait at the later join point.

  The created threads are recorded in the internal layerThreads list so
  that analog_wait_layers() can later join the full dispatch group and
  return the active worker's result.

  This function does not block. Synchronization is the responsibility
  of analog_wait_layers().
*/
extern "C" void analog_dispatch_layer(float *allocated, float *aligned,
                                      int64_t offset, int64_t size0,
                                      int64_t size1, int64_t stride0,
                                      int64_t stride1, int32_t layerId) {
  if (!layerThreads.empty()) {
    std::fprintf(stderr,
                 "analog_dispatch_layer called with pending layer threads\n");
    std::abort();
  }

  Tensor2DF32 input{allocated, aligned, offset, {size0, size1},
                    {stride0, stride1}};

  int32_t activeWorkerSlot = mapTaskToWorkerSlot(layerId);
  if (activeWorkerSlot < 0 ||
      activeWorkerSlot >= static_cast<int32_t>(NUM_LAYERS)) {
    std::fprintf(stderr,
                 "invalid active worker slot %d for layer %d (NUM_LAYERS=%d)\n",
                 static_cast<int>(activeWorkerSlot),
                 static_cast<int>(layerId),
                 static_cast<int>(NUM_LAYERS));
    std::abort();
  }

  layerThreads.reserve(static_cast<size_t>(NUM_LAYERS));
  for (int32_t workerSlot = 0; workerSlot < static_cast<int32_t>(NUM_LAYERS);
       ++workerSlot) {
    LayerTask *task =
        new LayerTask{input, layerId, workerSlot, activeWorkerSlot};
    pthread_t thread;

    int rc = pthread_create(&thread, nullptr, &layerThread, task);
    if (rc != 0) {
      std::fprintf(stderr,
                   "pthread_create failed for layer %d worker %d (rc=%d)\n",
                   static_cast<int>(layerId),
                   static_cast<int>(workerSlot),
                   rc);
      delete task;
      std::abort();
    }

    layerThreads.push_back(thread);
  }
}


/*
  analog_wait_layers()

  Runtime barrier for asynchronously dispatched layer executions.

  This function blocks until all layer threads previously launched via
  analog_dispatch_layer() have completed. It joins each outstanding
  pthread, clears the internal tracking list, and returns the result
  produced by the active worker.

  This is exported from analog-mlir and serves as the synchronization
  point for analog layer execution.
*/
extern "C" Tensor2DF32 analog_wait_layers() {
  Tensor2DF32 *joinedResult = nullptr;

  for (pthread_t thread : layerThreads) {
    void *threadResult = nullptr;
    int rc = pthread_join(thread, &threadResult);
    if (rc != 0) {
      std::fprintf(stderr, "pthread_join failed (rc=%d)\n", rc);
      std::abort();
    }

    if (threadResult != nullptr) {
      joinedResult = static_cast<Tensor2DF32 *>(threadResult);
    }
  }
  layerThreads.clear();

  if (joinedResult == nullptr) {
    std::fprintf(stderr, "analog_wait_layers completed with no layer result\n");
    std::abort();
  }

  Tensor2DF32 result = *joinedResult;
  delete joinedResult;

  printTensorResult(result);
  return result;
}
