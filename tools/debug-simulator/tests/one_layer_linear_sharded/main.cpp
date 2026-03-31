#include "../../../headers/tensor_types.h"

#include <cstdint>

extern "C" void analog_init_weights() __attribute__((weak));

extern "C" Tensor2DF32 forward(float *allocated, float *aligned, int64_t offset,
                               int64_t size0, int64_t size1, int64_t stride0,
                               int64_t stride1);
extern "C" Tensor2DF32
forward_arm_adapter(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t stride0,
                    int64_t stride1);

int main() {
  alignas(64) float inputData[8];
  for (int64_t i = 0; i < 8; ++i) {
    inputData[i] = static_cast<float>(i + 1);
  }

  if (analog_init_weights) {
    analog_init_weights();
  }

  Tensor2DF32 output = forward_arm_adapter(inputData, inputData, 0, 1, 8, 8, 1);
  printTensor("output", output);
  return output.aligned != nullptr ? 0 : 1;
}
