#include "analog_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  std::array<float, 25> inputStorage = {
      1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
      6.0f,  7.0f,  8.0f,  9.0f,  10.0f,
      11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
      16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
      21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
  };
  std::array<float, 54> outputStorage = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  };

  const void *inputs[] = {inputStorage.data()};
  void *outputs[] = {outputStorage.data()};

  void *runtime = runtime_init();
  if (!runtime) {
    std::fprintf(stderr, "runtime_init returned null\n");
    return 1;
  }

  const std::int32_t rc = runtime_execute(runtime, inputs, outputs);
  runtime_destroy(runtime);
  if (rc != 0) {
    std::fprintf(stderr, "runtime_execute failed with status %d\n", rc);
    return 1;
  }

  for (std::size_t i = 0; i < outputStorage.size(); ++i) {
    if (i != 0)
      std::printf(" ");
    std::printf("%.6f", outputStorage[i]);
  }
  std::printf("\n");
  return 0;
}
