#include "analog_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  std::array<float, 4> inputStorage = {
      1.0f, 2.0f, 3.0f, 4.0f,
  };
  std::array<float, 3> hidden0Storage = {
      0.1f, 0.2f, 0.3f,
  };
  std::array<float, 3> hidden1Storage = {
      0.4f, 0.5f, 0.6f,
  };
  std::array<float, 3> outputStorage = {
      0.0f, 0.0f, 0.0f,
  };

  const void *inputs[] = {
      inputStorage.data(),
      hidden0Storage.data(),
      hidden1Storage.data(),
  };
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
