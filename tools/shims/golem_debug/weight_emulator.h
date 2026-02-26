#ifndef ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_WEIGHT_EMULATOR_H
#define ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_WEIGHT_EMULATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace analog {
namespace shims {

class ComputeArray {
public:
  // Owns one simulated analog array instance.
  ComputeArray(int32_t rows, int32_t cols, int32_t arrayId = -1);

  int32_t rows() const { return rows_; }
  int32_t cols() const { return cols_; }
  int32_t arrayId() const { return arrayId_; }
  void setArrayId(int32_t arrayId) { arrayId_ = arrayId; }

  const std::vector<float> &matrix() const { return matrix_; }
  const std::vector<float> &inputVector() const { return inputVector_; }
  const std::vector<float> &outputVector() const { return outputVector_; }

  // Reset all internal buffers.
  void clear();

  // Matrix programming path (host buffer -> simulated array matrix).
  void setMatrixFromRowMajor(const float *src, int32_t srcStride);
  void setMatrixFromPackedId(const float *src, int32_t packedArrayId);

  // Input load path (host vector -> simulated array input register).
  void loadVector(const float *src);

  // Array compute path (matrix-vector multiply).
  void compute();

  // Output store path (simulated array output register -> host buffer).
  void storeOutput(float *dst) const;

  // Debug helpers.
  void dumpMatrix(const char *label = "matrix") const;
  void dumpInputVector(const char *label = "vector") const;
  void dumpOutputVector(const char *label = "output") const;

  // Packed-id helpers used by the debug shim ABI.
  static int32_t packArrayId(int32_t arrayId, int32_t matrixWidth);
  static int32_t arrayIndexFromPackedId(int32_t packedArrayId);
  static int32_t matrixWidthFromPackedId(int32_t packedArrayId);

private:
  // Row-major matrix indexing helper.
  size_t matrixIndex(int32_t row, int32_t col) const;

  int32_t rows_;
  int32_t cols_;
  int32_t arrayId_;
  std::vector<float> matrix_;
  std::vector<float> inputVector_;
  std::vector<float> outputVector_;
};

} // namespace shims
} // namespace analog

#endif // ANALOG_MLIR_TOOLS_SHIMS_GOLEM_DEBUG_WEIGHT_EMULATOR_H
