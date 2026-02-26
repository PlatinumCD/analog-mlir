#include "weight_emulator.h"

#include <algorithm>
#include <cstdio>

namespace analog {
namespace shims {

// ---------------------------------------------------------------------------
// Construction / Reset
// ---------------------------------------------------------------------------
// Owns the simulated array buffers and resets their contents.
ComputeArray::ComputeArray(int32_t rows, int32_t cols, int32_t arrayId)
    : rows_(rows),
      cols_(cols),
      arrayId_(arrayId),
      matrix_(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f),
      inputVector_(static_cast<size_t>(cols), 0.0f),
      outputVector_(static_cast<size_t>(rows), 0.0f) {}

void ComputeArray::clear() {
  std::fill(matrix_.begin(), matrix_.end(), 0.0f);
  std::fill(inputVector_.begin(), inputVector_.end(), 0.0f);
  std::fill(outputVector_.begin(), outputVector_.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// Matrix Programming Path
// ---------------------------------------------------------------------------
// Programs the simulated matrix state from host memory.
void ComputeArray::setMatrixFromRowMajor(const float *src, int32_t srcStride) {
  if (!src) {
    return;
  }
  if (srcStride <= 0) {
    srcStride = cols_;
  }

  for (int32_t r = 0; r < rows_; ++r) {
    for (int32_t c = 0; c < cols_; ++c) {
      matrix_[matrixIndex(r, c)] =
          src[static_cast<size_t>(r) * static_cast<size_t>(srcStride) +
              static_cast<size_t>(c)];
    }
  }
}

void ComputeArray::setMatrixFromPackedId(const float *src, int32_t packedArrayId) {
  int32_t stride = matrixWidthFromPackedId(packedArrayId);
  setMatrixFromRowMajor(src, stride > 0 ? stride : cols_);
}

// ---------------------------------------------------------------------------
// Input Load Path
// ---------------------------------------------------------------------------
// Loads the host input vector into the simulated array input state.
void ComputeArray::loadVector(const float *src) {
  if (!src) {
    return;
  }
  for (int32_t c = 0; c < cols_; ++c) {
    inputVector_[static_cast<size_t>(c)] = src[static_cast<size_t>(c)];
  }
}

// ---------------------------------------------------------------------------
// Compute Path
// ---------------------------------------------------------------------------
// Simulates the array's matrix-vector multiply.
void ComputeArray::compute() {
  for (int32_t r = 0; r < rows_; ++r) {
    float acc = 0.0f;
    for (int32_t c = 0; c < cols_; ++c) {
      acc += matrix_[matrixIndex(r, c)] * inputVector_[static_cast<size_t>(c)];
    }
    outputVector_[static_cast<size_t>(r)] = acc;
  }
}

// ---------------------------------------------------------------------------
// Output Store Path
// ---------------------------------------------------------------------------
// Writes the simulated output state back to host memory.
void ComputeArray::storeOutput(float *dst) const {
  if (!dst) {
    return;
  }
  for (int32_t r = 0; r < rows_; ++r) {
    dst[static_cast<size_t>(r)] = outputVector_[static_cast<size_t>(r)];
  }
}

// ---------------------------------------------------------------------------
// Debug Helpers
// ---------------------------------------------------------------------------
// Pretty-printers for inspecting simulated array state while debugging.
void ComputeArray::dumpMatrix(const char *label) const {
  std::printf("[sim] array=%d %s (%d x %d)\n", arrayId_, label, rows_, cols_);
  for (int32_t r = 0; r < rows_; ++r) {
    std::printf("  [");
    for (int32_t c = 0; c < cols_; ++c) {
      std::printf("%s%.3f", (c == 0 ? "" : ", "), matrix_[matrixIndex(r, c)]);
    }
    std::printf("]\n");
  }
}

void ComputeArray::dumpInputVector(const char *label) const {
  std::printf("[sim] array=%d %s [", arrayId_, label);
  for (int32_t c = 0; c < cols_; ++c) {
    std::printf("%s%.3f", (c == 0 ? "" : ", "),
                inputVector_[static_cast<size_t>(c)]);
  }
  std::printf("]\n");
}

void ComputeArray::dumpOutputVector(const char *label) const {
  std::printf("[sim] array=%d %s [", arrayId_, label);
  for (int32_t r = 0; r < rows_; ++r) {
    std::printf("%s%.3f", (r == 0 ? "" : ", "),
                outputVector_[static_cast<size_t>(r)]);
  }
  std::printf("]\n");
}

// ---------------------------------------------------------------------------
// Packed-ID Helpers (Debug Shim ABI)
// ---------------------------------------------------------------------------
// Encodes/decodes the packed array id used by the lowered debug shim calls.
int32_t ComputeArray::packArrayId(int32_t arrayId, int32_t matrixWidth) {
  uint32_t lo = static_cast<uint32_t>(arrayId) & 0xFFFFU;
  uint32_t hi = (static_cast<uint32_t>(matrixWidth) & 0xFFFFU) << 16;
  return static_cast<int32_t>(hi | lo);
}

int32_t ComputeArray::arrayIndexFromPackedId(int32_t packedArrayId) {
  uint32_t raw = static_cast<uint32_t>(packedArrayId);
  return static_cast<int32_t>(raw & 0xFFFFU);
}

int32_t ComputeArray::matrixWidthFromPackedId(int32_t packedArrayId) {
  uint32_t raw = static_cast<uint32_t>(packedArrayId);
  return static_cast<int32_t>((raw >> 16) & 0xFFFFU);
}

// ---------------------------------------------------------------------------
// Internal Indexing Helper
// ---------------------------------------------------------------------------
// Computes row-major offsets into the matrix buffer.
size_t ComputeArray::matrixIndex(int32_t row, int32_t col) const {
  return static_cast<size_t>(row) * static_cast<size_t>(cols_) +
         static_cast<size_t>(col);
}

} // namespace shims
} // namespace analog
