#ifndef ANALOG_TESTS_RUNTIME_GOLEM_BRIDGE_H
#define ANALOG_TESTS_RUNTIME_GOLEM_BRIDGE_H

#include <cstdint>

extern "C" {

void *runtime_init();
std::int32_t runtime_execute(void *runtime, const void *const *inputs,
                             void *const *outputs);
void runtime_destroy(void *runtime);

}

#endif // ANALOG_TESTS_RUNTIME_GOLEM_BRIDGE_H
