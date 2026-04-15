#ifndef ANALOG_GOLEM_SIMULATOR_GOLEMSHIMS_H
#define ANALOG_GOLEM_SIMULATOR_GOLEMSHIMS_H

#include <cstdint>

extern "C" {

void golem_analog_mvm_set(float *basePtr, float *data, std::int64_t offset,
                          std::int64_t size0, std::int64_t size1,
                          std::int64_t stride0, std::int64_t stride1,
                          std::int32_t arrayId);

void golem_analog_mvm_load(float *basePtr, float *data, std::int64_t offset,
                           std::int64_t size0, std::int64_t size1,
                           std::int64_t stride0, std::int64_t stride1,
                           std::int32_t arrayId);

void golem_analog_mvm_compute(std::int32_t arrayId);

void golem_analog_mvm_store(float *basePtr, float *data, std::int64_t offset,
                            std::int64_t size0, std::int64_t size1,
                            std::int64_t size2, std::int64_t stride0,
                            std::int64_t stride1, std::int64_t stride2,
                            std::int32_t arrayId);

} // extern "C"

#endif // ANALOG_GOLEM_SIMULATOR_GOLEMSHIMS_H
