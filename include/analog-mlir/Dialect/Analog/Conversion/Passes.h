#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_PASSES_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_PASSES_H

namespace mlir {
namespace analog {

void registerAnalogConversionPasses();
void registerLowerToGolemPipeline();
void registerLowerToDebugShimsPipeline();

} // namespace analog
} // namespace mlir

#endif
