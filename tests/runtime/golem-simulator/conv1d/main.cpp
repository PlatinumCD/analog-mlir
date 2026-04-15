#include "analog_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  std::array<float, 6> inputStorage = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
  };
  std::array<float, 12> outputStorage = {
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
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
