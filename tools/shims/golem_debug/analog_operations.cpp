#include "weight_emulator.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <sched.h>

#ifndef GOLEM_DEBUG_NUM_ARRAYS
#define GOLEM_DEBUG_NUM_ARRAYS 2
#endif

#ifndef GOLEM_DEBUG_ARRAY_ROWS
#define GOLEM_DEBUG_ARRAY_ROWS 8
#endif

#ifndef GOLEM_DEBUG_ARRAY_COLS
#define GOLEM_DEBUG_ARRAY_COLS 8
#endif

using analog::shims::ComputeArray;

static std::vector<ComputeArray> arrays;
static std::vector<std::unique_ptr<std::mutex>> arrayLocks;
static std::once_flag initOnce;


/*
  initializeArrays()

  Lazily constructs the fixed set of simulated analog arrays used by
  the debug shim layer. Array ids are 0-based to match the packed-id
  convention used by the lowered MLIR debug shim ABI.
*/
static void initializeArrays() {
  arrays.reserve(GOLEM_DEBUG_NUM_ARRAYS);
  arrayLocks.reserve(GOLEM_DEBUG_NUM_ARRAYS);
  for (int32_t i = 0; i < GOLEM_DEBUG_NUM_ARRAYS; ++i) {
    arrays.emplace_back(GOLEM_DEBUG_ARRAY_ROWS, GOLEM_DEBUG_ARRAY_COLS, i);
    arrayLocks.push_back(std::make_unique<std::mutex>());
  }
}


/*
  logCoreAndSleep(const char* fnName)

  Debug helper to show which CPU core is executing the shim and make
  concurrent execution easier to observe in logs.
*/
static int logCoreAndSleep(const char *fnName) {
  int core = sched_getcpu();
  std::printf("[operation shim] %s running on CORE#%d\n", fnName, core);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return core;
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
  int core = logCoreAndSleep("mvm.set");

  std::call_once(initOnce, initializeArrays);
  int32_t arrayId = ComputeArray::arrayIndexFromPackedId(packedArrayId);
  if (arrayId < 0 || arrayId >= GOLEM_DEBUG_NUM_ARRAYS) {
    std::fprintf(stderr,
                 "[operation shim] array id %d out of range (valid: 0..%d)\n",
                 static_cast<int>(arrayId),
                 static_cast<int>(GOLEM_DEBUG_NUM_ARRAYS - 1));
    std::abort();
  }

  std::lock_guard<std::mutex> lock(*arrayLocks[static_cast<size_t>(arrayId)]);
  ComputeArray &array = arrays[static_cast<size_t>(arrayId)];
  float *src = static_cast<float *>(data);

  const int32_t rows = array.rows();
  const int32_t cols = array.cols();
  const int32_t matrixWidth = ComputeArray::matrixWidthFromPackedId(packedArrayId);
  const int32_t srcStride = matrixWidth > 0 ? matrixWidth : cols;

  array.setMatrixFromRowMajor(src, srcStride);

  std::printf(
      "[operation shim] mvm.set   ptr=%p array=%d raw=%d matrix_width=%d rows=%d cols=%d core=%d\n",
      data,
      array.arrayId(),
      static_cast<int>(packedArrayId),
      srcStride,
      rows,
      cols,
      core);
  array.dumpMatrix("matrix");
}
