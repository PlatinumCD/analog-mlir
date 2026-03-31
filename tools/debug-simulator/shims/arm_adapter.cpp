#include "../../headers/tensor_types.h"

#include <cstdint>

extern "C" Tensor2DF32 forward(float *allocated, float *aligned, int64_t offset,
                               int64_t size0, int64_t size1, int64_t stride0,
                               int64_t stride1);

extern "C" Tensor2DF32
forward_arm_adapter(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t stride0,
                    int64_t stride1) {
#if defined(__aarch64__)
  register float *x0 asm("x0") = allocated;
  register float *x1 asm("x1") = aligned;
  register int64_t x2 asm("x2") = offset;
  register int64_t x3 asm("x3") = size0;
  register int64_t x4 asm("x4") = size1;
  register int64_t x5 asm("x5") = stride0;
  register int64_t x6 asm("x6") = stride1;

  asm volatile("bl forward"
               : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3), "+r"(x4), "+r"(x5),
                 "+r"(x6)
               :
               : "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
                 "x16", "x17", "x18", "x30", "cc", "memory");

  return Tensor2DF32{x0, x1, x2, {x3, x4}, {x5, x6}};
#else
  return forward(allocated, aligned, offset, size0, size1, stride0, stride1);
#endif
}
