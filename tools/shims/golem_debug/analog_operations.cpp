#include "weight_emulator.h"
#include "thread_mapping.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <sched.h>

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

#ifndef GOLEM_DEBUG_NUM_ARRAYS
#define GOLEM_DEBUG_NUM_ARRAYS 2
#endif

#ifndef GOLEM_DEBUG_ARRAY_ROWS
#define GOLEM_DEBUG_ARRAY_ROWS 8
#endif

#ifndef GOLEM_DEBUG_ARRAY_COLS
#define GOLEM_DEBUG_ARRAY_COLS 8
#endif

#ifndef GOLEM_DEBUG_SLEEP_MS
#define GOLEM_DEBUG_SLEEP_MS 0
#endif

#ifndef GOLEM_DEBUG_LOG_OPERATIONS
#define GOLEM_DEBUG_LOG_OPERATIONS 0
#endif

#ifndef GOLEM_DEBUG_DUMP_ARRAY_STATE
#define GOLEM_DEBUG_DUMP_ARRAY_STATE 0
#endif

using analog::shims::ComputeArray;

/*
  createWorkerLocalArrays()

  Builds one local array bank per worker slot. Each worker gets its own
  0-based array namespace.
*/
static std::vector<ComputeArray> createWorkerLocalArrays() {
  std::vector<ComputeArray> arrays;
  arrays.reserve(static_cast<size_t>(NUM_LAYERS) *
                 static_cast<size_t>(GOLEM_DEBUG_NUM_ARRAYS));
  for (int32_t workerSlot = 0; workerSlot < static_cast<int32_t>(NUM_LAYERS);
       ++workerSlot) {
    for (int32_t arrayId = 0; arrayId < GOLEM_DEBUG_NUM_ARRAYS; ++arrayId) {
      arrays.emplace_back(GOLEM_DEBUG_ARRAY_ROWS, GOLEM_DEBUG_ARRAY_COLS,
                          arrayId);
    }
  }
  return arrays;
}


/*
  logCoreAndMaybeSleep(const char* fnName)

  Debug helper to show which CPU core is executing the shim and make
  concurrent execution easier to observe in logs.
*/
static int logCoreAndMaybeSleep(const char *fnName) {
  int core = sched_getcpu();
#if GOLEM_DEBUG_LOG_OPERATIONS
  std::printf("[operation shim] %s running on CORE#%d\n", fnName, core);
#endif
#if GOLEM_DEBUG_SLEEP_MS > 0
  std::this_thread::sleep_for(std::chrono::milliseconds(GOLEM_DEBUG_SLEEP_MS));
#endif
  return core;
}


/*
  getWorkerLocalArray(int32_t rawArrayId)

  Resolves the current worker-local array instance for a packed/raw id.
*/
static ComputeArray &getWorkerLocalArray(int32_t rawArrayId,
                                         int32_t &workerSlot,
                                         int32_t &arrayId) {
  static std::vector<ComputeArray> arrays = createWorkerLocalArrays();

  workerSlot = getCurrentWorkerSlot();
  if (workerSlot < 0 || workerSlot >= static_cast<int32_t>(NUM_LAYERS)) {
    std::fprintf(stderr,
                 "[operation shim] worker slot %d is invalid for raw=%d (valid: 0..%d)\n",
                 static_cast<int>(workerSlot),
                 static_cast<int>(rawArrayId),
                 static_cast<int>(NUM_LAYERS - 1));
    std::abort();
  }

  arrayId = ComputeArray::arrayIndexFromPackedId(rawArrayId);
  if (arrayId < 0 || arrayId >= GOLEM_DEBUG_NUM_ARRAYS) {
    std::fprintf(stderr,
                 "[operation shim] array id %d out of range (raw=%d, valid: 0..%d)\n",
                 static_cast<int>(arrayId),
                 static_cast<int>(rawArrayId),
                 static_cast<int>(GOLEM_DEBUG_NUM_ARRAYS - 1));
    std::abort();
  }

  size_t flatIndex = static_cast<size_t>(workerSlot) *
                         static_cast<size_t>(GOLEM_DEBUG_NUM_ARRAYS) +
                     static_cast<size_t>(arrayId);
  return arrays[flatIndex];
}


/*
  golem_debug_mvm_set(void* data, int32_t packedArrayId)

  Debug shim implementation for the analog matrix-programming operation.

  The lowered MLIR passes a host pointer to matrix data and a packed
  array id. This shim decodes the target array, determines the source
  row stride from the packed id, programs the simulated ComputeArray
  matrix state, and prints a debug dump.
*/
extern "C" void golem_debug_mvm_set(void *data, int32_t packedArrayId) {
  int core = logCoreAndMaybeSleep("mvm.set");

  int32_t workerSlot = -1;
  int32_t arrayId = -1;
  ComputeArray &array =
      getWorkerLocalArray(packedArrayId, workerSlot, arrayId);
  float *src = static_cast<float *>(data);

  const int32_t rows = array.rows();
  const int32_t cols = array.cols();
  const int32_t matrixWidth = ComputeArray::matrixWidthFromPackedId(packedArrayId);
  const int32_t srcStride = matrixWidth > 0 ? matrixWidth : cols;

  array.setMatrixFromRowMajor(src, srcStride);

#if GOLEM_DEBUG_LOG_OPERATIONS
  std::printf(
      "[operation shim] mvm.set   ptr=%p worker=%d array=%d raw=%d matrix_width=%d rows=%d cols=%d core=%d\n",
      data,
      workerSlot,
      arrayId,
      static_cast<int>(packedArrayId),
      srcStride,
      rows,
      cols,
      core);
#endif
#if GOLEM_DEBUG_DUMP_ARRAY_STATE
  array.dumpMatrix("matrix");
#endif
}


/*
  golem_debug_mvm_load(void* data, int32_t rawArrayId)

  Debug shim implementation for the analog vector-load operation.
*/
extern "C" void golem_debug_mvm_load(void *data, int32_t rawArrayId) {
  int core = logCoreAndMaybeSleep("mvm.load");

  int32_t workerSlot = -1;
  int32_t arrayId = -1;
  ComputeArray &array = getWorkerLocalArray(rawArrayId, workerSlot, arrayId);
  float *src = static_cast<float *>(data);

  array.loadVector(src);

#if GOLEM_DEBUG_LOG_OPERATIONS
  std::printf(
      "[operation shim] mvm.load  ptr=%p worker=%d array=%d raw=%d cols=%d core=%d\n",
      data,
      workerSlot,
      arrayId,
      static_cast<int>(rawArrayId),
      array.cols(),
      core);
#endif
#if GOLEM_DEBUG_DUMP_ARRAY_STATE
  array.dumpInputVector("input");
#endif
}


/*
  golem_debug_mvm_compute(int32_t rawArrayId)

  Debug shim implementation for the analog compute operation.
*/
extern "C" void golem_debug_mvm_compute(int32_t rawArrayId) {
  int core = logCoreAndMaybeSleep("mvm.compute");

  int32_t workerSlot = -1;
  int32_t arrayId = -1;
  ComputeArray &array = getWorkerLocalArray(rawArrayId, workerSlot, arrayId);

  array.compute();

#if GOLEM_DEBUG_LOG_OPERATIONS
  std::printf("[operation shim] mvm.compute worker=%d array=%d raw=%d core=%d\n",
              workerSlot,
              arrayId,
              static_cast<int>(rawArrayId),
              core);
#endif
#if GOLEM_DEBUG_DUMP_ARRAY_STATE
  array.dumpOutputVector("output");
#endif
}


/*
  golem_debug_mvm_store(void* data, int32_t rawArrayId)

  Debug shim implementation for the analog output-store operation.
*/
extern "C" void golem_debug_mvm_store(void *data, int32_t rawArrayId) {
  int core = logCoreAndMaybeSleep("mvm.store");

  int32_t workerSlot = -1;
  int32_t arrayId = -1;
  ComputeArray &array = getWorkerLocalArray(rawArrayId, workerSlot, arrayId);
  float *dst = static_cast<float *>(data);

  array.storeOutput(dst);

#if GOLEM_DEBUG_LOG_OPERATIONS
  std::printf(
      "[operation shim] mvm.store ptr=%p worker=%d array=%d raw=%d rows=%d core=%d\n",
      data,
      workerSlot,
      arrayId,
      static_cast<int>(rawArrayId),
      array.rows(),
      core);
#endif
}
