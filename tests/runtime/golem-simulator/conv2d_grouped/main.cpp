#include "analog_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  std::array<float, 100> inputStorage;
  std::array<float, 54> outputStorage;

  for (std::size_t i = 0; i < inputStorage.size(); ++i)
    inputStorage[i] = static_cast<float>(i + 1);
  outputStorage.fill(0.0f);

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
