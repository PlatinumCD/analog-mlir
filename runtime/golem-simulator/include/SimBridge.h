#ifndef ANALOG_GOLEM_SIMULATOR_SIMBRIDGE_H
#define ANALOG_GOLEM_SIMULATOR_SIMBRIDGE_H

#include <cstdint>

namespace analog::golem_sim {

void initializeBridge(std::int32_t numCores, std::int32_t arraysPerCore,
                      std::int32_t arrayRows, std::int32_t arrayCols);
void shutdownBridge();

void setActiveCore(std::int32_t coreIndex);
void clearActiveCore();

void recordMvmSet(const float *data, std::int32_t rawArrayId);
void recordMvmLoad(const float *data, std::int32_t rawArrayId);
void recordMvmCompute(std::int32_t rawArrayId);
void copyMvmStore(float *dst, std::uint64_t elementCount,
                  std::int32_t rawArrayId);

} // namespace analog::golem_sim

#endif // ANALOG_GOLEM_SIMULATOR_SIMBRIDGE_H
