#include <cstdint>
#include <cstdio>

extern "C" void analog_init_weights();

struct Tensor2DF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

extern "C" Tensor2DF32 forward(float *allocated, float *aligned, int64_t offset,
                               int64_t size0, int64_t size1, int64_t stride0,
                               int64_t stride1);

static void printTensor(const char *label, const Tensor2DF32 &tensor) {
  std::printf("%s (%lld x %lld)\n",
              label,
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

int main() {
  alignas(64) float inputData[8] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };

  analog_init_weights();

  Tensor2DF32 output =
      forward(inputData, inputData, 0, 1, 8, 8, 1);

  printTensor("forward output", output);
  return 0;
}
