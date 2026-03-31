#ifndef ANALOG_MLIR_HEADERS_TENSOR_TYPES_H
#define ANALOG_MLIR_HEADERS_TENSOR_TYPES_H

#include <cstdint>
#include <cstdio>
 
/*
  Tensor2DF32

  Canonical rank-2 floating-point tensor ABI used by the debug simulator
  boundary.

  This struct mirrors the memref-style tensor representation expected by the
  generated layer dispatch entrypoints and the simulator shims:
  - allocated: original allocation base pointer
  - aligned: aligned data pointer used for indexed access
  - offset: element offset from the aligned pointer
  - sizes: extents for each logical dimension
  - strides: per-dimension element strides
*/
struct Tensor2DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

/*
  Tensor3DF32

  Canonical rank-3 floating-point tensor ABI used by the debug simulator
  boundary.

  The layout matches the generated runtime-facing tensor representation used by
  simulator dispatch/run/wait functions.
*/
struct Tensor3DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[3];
  int64_t strides[3];
};

/*
  Tensor4DF32

  Canonical rank-4 floating-point tensor ABI used by the debug simulator
  boundary.

  This definition is intended to be shared by simulator test harnesses, shim
  implementations, and generated runtime entrypoints.
*/
struct Tensor4DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
};

/*
  Tensor5DF32

  Canonical rank-5 floating-point tensor ABI used by the debug simulator
  boundary.

  This is the highest-rank tensor ABI currently required by the existing layer
  dispatch surface.
*/
struct Tensor5DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[5];
  int64_t strides[5];
};


/*
  printTensor

  Prints the full descriptor and all element values of a rank-2 tensor.
*/
inline void printTensor(const char *label, const Tensor2DF32 &tensor) {
  std::printf("%s (%lld x %lld)\n", label,
              static_cast<long long>(tensor.sizes[0]),
              static_cast<long long>(tensor.sizes[1]));
#ifdef DEBUG_MODE
  std::printf("  allocated=%p aligned=%p offset=%lld\n",
              static_cast<void *>(tensor.allocated),
              static_cast<void *>(tensor.aligned),
              static_cast<long long>(tensor.offset));
  std::printf("  strides=(%lld x %lld)\n",
              static_cast<long long>(tensor.strides[0]),
              static_cast<long long>(tensor.strides[1]));
#endif

  for (int64_t row = 0; row < tensor.sizes[0]; ++row) {
    std::printf("  [");
    for (int64_t col = 0; col < tensor.sizes[1]; ++col) {
      int64_t index = tensor.offset + row * tensor.strides[0] +
                      col * tensor.strides[1];
      std::printf("%s%.6f", col == 0 ? "" : ", ", tensor.aligned[index]);
    }
    std::printf("]\n");
  }
}

/*
  printTensor

  Prints the full descriptor and all element values of a rank-3 tensor.
*/
inline void printTensor(const char *label, const Tensor3DF32 &tensor) {
  std::printf("%s (%lld x %lld x %lld)\n", label,
              static_cast<long long>(tensor.sizes[0]),
              static_cast<long long>(tensor.sizes[1]),
              static_cast<long long>(tensor.sizes[2]));
#ifdef DEBUG_MODE
  std::printf("  allocated=%p aligned=%p offset=%lld\n",
              static_cast<void *>(tensor.allocated),
              static_cast<void *>(tensor.aligned),
              static_cast<long long>(tensor.offset));
  std::printf("  strides=(%lld x %lld x %lld)\n",
              static_cast<long long>(tensor.strides[0]),
              static_cast<long long>(tensor.strides[1]),
              static_cast<long long>(tensor.strides[2]));
#endif

  for (int64_t d0 = 0; d0 < tensor.sizes[0]; ++d0) {
    for (int64_t d1 = 0; d1 < tensor.sizes[1]; ++d1) {
      std::printf("  [d0=%lld, d1=%lld] [", static_cast<long long>(d0),
                  static_cast<long long>(d1));
      for (int64_t d2 = 0; d2 < tensor.sizes[2]; ++d2) {
        int64_t index = tensor.offset + d0 * tensor.strides[0] +
                        d1 * tensor.strides[1] + d2 * tensor.strides[2];
        std::printf("%s%.6f", d2 == 0 ? "" : ", ", tensor.aligned[index]);
      }
      std::printf("]\n");
    }
  }
}

/*
  printTensor

  Prints the full descriptor and all element values of a rank-4 tensor.
*/
inline void printTensor(const char *label, const Tensor4DF32 &tensor) {
  std::printf("%s (%lld x %lld x %lld x %lld)\n", label,
              static_cast<long long>(tensor.sizes[0]),
              static_cast<long long>(tensor.sizes[1]),
              static_cast<long long>(tensor.sizes[2]),
              static_cast<long long>(tensor.sizes[3]));
#ifdef DEBUG_MODE
  std::printf("  allocated=%p aligned=%p offset=%lld\n",
              static_cast<void *>(tensor.allocated),
              static_cast<void *>(tensor.aligned),
              static_cast<long long>(tensor.offset));
  std::printf("  strides=(%lld x %lld x %lld x %lld)\n",
              static_cast<long long>(tensor.strides[0]),
              static_cast<long long>(tensor.strides[1]),
              static_cast<long long>(tensor.strides[2]),
              static_cast<long long>(tensor.strides[3]));
#endif

  for (int64_t d0 = 0; d0 < tensor.sizes[0]; ++d0) {
    for (int64_t d1 = 0; d1 < tensor.sizes[1]; ++d1) {
      for (int64_t d2 = 0; d2 < tensor.sizes[2]; ++d2) {
        std::printf("  [d0=%lld, d1=%lld, d2=%lld] [",
                    static_cast<long long>(d0), static_cast<long long>(d1),
                    static_cast<long long>(d2));
        for (int64_t d3 = 0; d3 < tensor.sizes[3]; ++d3) {
          int64_t index = tensor.offset + d0 * tensor.strides[0] +
                          d1 * tensor.strides[1] + d2 * tensor.strides[2] +
                          d3 * tensor.strides[3];
          std::printf("%s%.6f", d3 == 0 ? "" : ", ", tensor.aligned[index]);
        }
        std::printf("]\n");
      }
    }
  }
}

/*
  printTensor

  Prints the full descriptor and all element values of a rank-5 tensor.
*/
inline void printTensor(const char *label, const Tensor5DF32 &tensor) {
  std::printf("%s (%lld x %lld x %lld x %lld x %lld)\n", label,
              static_cast<long long>(tensor.sizes[0]),
              static_cast<long long>(tensor.sizes[1]),
              static_cast<long long>(tensor.sizes[2]),
              static_cast<long long>(tensor.sizes[3]),
              static_cast<long long>(tensor.sizes[4]));
#ifdef DEBUG_MODE
  std::printf("  allocated=%p aligned=%p offset=%lld\n",
              static_cast<void *>(tensor.allocated),
              static_cast<void *>(tensor.aligned),
              static_cast<long long>(tensor.offset));
  std::printf("  strides=(%lld x %lld x %lld x %lld x %lld)\n",
              static_cast<long long>(tensor.strides[0]),
              static_cast<long long>(tensor.strides[1]),
              static_cast<long long>(tensor.strides[2]),
              static_cast<long long>(tensor.strides[3]),
              static_cast<long long>(tensor.strides[4]));
#endif

  for (int64_t d0 = 0; d0 < tensor.sizes[0]; ++d0) {
    for (int64_t d1 = 0; d1 < tensor.sizes[1]; ++d1) {
      for (int64_t d2 = 0; d2 < tensor.sizes[2]; ++d2) {
        for (int64_t d3 = 0; d3 < tensor.sizes[3]; ++d3) {
          std::printf("  [d0=%lld, d1=%lld, d2=%lld, d3=%lld] [",
                      static_cast<long long>(d0), static_cast<long long>(d1),
                      static_cast<long long>(d2), static_cast<long long>(d3));
          for (int64_t d4 = 0; d4 < tensor.sizes[4]; ++d4) {
            int64_t index = tensor.offset + d0 * tensor.strides[0] +
                            d1 * tensor.strides[1] + d2 * tensor.strides[2] +
                            d3 * tensor.strides[3] + d4 * tensor.strides[4];
            std::printf("%s%.6f", d4 == 0 ? "" : ", ",
                        tensor.aligned[index]);
          }
          std::printf("]\n");
        }
      }
    }
  }
}

#endif // ANALOG_MLIR_HEADERS_TENSOR_TYPES_H
