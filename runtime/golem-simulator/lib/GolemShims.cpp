#include "GolemShims.h"

#include "SimBridge.h"
#include "runtime_utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

[[noreturn]] void fail(const char *context, const std::string &detail) {
  std::fprintf(stderr, "golem-simulator shim error: %s", context);
  if (!detail.empty())
    std::fprintf(stderr, ": %s", detail.c_str());
  std::fprintf(stderr, "\n");
  std::abort();
}

MemRef2D buildMemRef2D(float *basePtr, float *data, std::int64_t offset,
                       std::int64_t size0, std::int64_t size1,
                       std::int64_t stride0, std::int64_t stride1) {
  MemRef2D memref{};
  memref.basePtr = basePtr;
  memref.data = data;
  memref.offset = offset;
  memref.sizes[0] = size0;
  memref.sizes[1] = size1;
  memref.strides[0] = stride0;
  memref.strides[1] = stride1;
  return memref;
}

MemRef3D buildMemRef3D(float *basePtr, float *data, std::int64_t offset,
                       std::int64_t size0, std::int64_t size1,
                       std::int64_t size2, std::int64_t stride0,
                       std::int64_t stride1, std::int64_t stride2) {
  MemRef3D memref{};
  memref.basePtr = basePtr;
  memref.data = data;
  memref.offset = offset;
  memref.sizes[0] = size0;
  memref.sizes[1] = size1;
  memref.sizes[2] = size2;
  memref.strides[0] = stride0;
  memref.strides[1] = stride1;
  memref.strides[2] = stride2;
  return memref;
}

float *requireContiguous2D(const MemRef2D &memref, const char *context) {
  if (!memref.data)
    fail(context, "null memref data pointer");
  if (memref.strides[1] != 1 || memref.strides[0] != memref.sizes[1])
    fail(context, "expected contiguous row-major 2D memref");
  return memref.data + memref.offset;
}

float *requireContiguous3D(const MemRef3D &memref, const char *context) {
  if (!memref.data)
    fail(context, "null memref data pointer");
  if (memref.strides[2] != 1 ||
      memref.strides[1] != memref.sizes[2] ||
      memref.strides[0] != memref.sizes[1] * memref.sizes[2]) {
    fail(context, "expected contiguous row-major 3D memref");
  }
  return memref.data + memref.offset;
}

void requireLocalArrayId(std::int32_t arrayId, const char *context) {
  if (arrayId < 0)
    fail(context, "expected non-negative local array id");
}

} // namespace

extern "C" {

void golem_analog_mvm_set(float *basePtr, float *data, std::int64_t offset,
                          std::int64_t size0, std::int64_t size1,
                          std::int64_t stride0, std::int64_t stride1,
                          std::int32_t arrayId) {
  (void)basePtr;
  requireLocalArrayId(arrayId, "golem_analog_mvm_set");
  MemRef2D memref =
      buildMemRef2D(basePtr, data, offset, size0, size1, stride0, stride1);
  analog::golem_sim::recordMvmSet(
      requireContiguous2D(memref, "golem_analog_mvm_set"), arrayId);
}

void golem_analog_mvm_load(float *basePtr, float *data, std::int64_t offset,
                           std::int64_t size0, std::int64_t size1,
                           std::int64_t stride0, std::int64_t stride1,
                           std::int32_t arrayId) {
  (void)basePtr;
  requireLocalArrayId(arrayId, "golem_analog_mvm_load");
  MemRef2D memref =
      buildMemRef2D(basePtr, data, offset, size0, size1, stride0, stride1);
  analog::golem_sim::recordMvmLoad(
      requireContiguous2D(memref, "golem_analog_mvm_load"), arrayId);
}

void golem_analog_mvm_compute(std::int32_t arrayId) {
  requireLocalArrayId(arrayId, "golem_analog_mvm_compute");
  analog::golem_sim::recordMvmCompute(arrayId);
}

void golem_analog_mvm_store(float *basePtr, float *data, std::int64_t offset,
                            std::int64_t size0, std::int64_t size1,
                            std::int64_t size2, std::int64_t stride0,
                            std::int64_t stride1, std::int64_t stride2,
                            std::int32_t arrayId) {
  (void)basePtr;
  requireLocalArrayId(arrayId, "golem_analog_mvm_store");
  MemRef3D memref = buildMemRef3D(basePtr, data, offset, size0, size1, size2,
                                  stride0, stride1, stride2);
  if (memref.sizes[0] != 1 || memref.sizes[1] != 1)
    fail("golem_analog_mvm_store",
         "expected store scratch memref with shape [1, 1, lanes]");
  analog::golem_sim::copyMvmStore(
      requireContiguous3D(memref, "golem_analog_mvm_store"),
      static_cast<std::uint64_t>(memref.sizes[2]), arrayId);
}

} // extern "C"
