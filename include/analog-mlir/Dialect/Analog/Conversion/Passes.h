#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_PASSES_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_PASSES_H

namespace mlir {
namespace analog {

void registerEmitRuntimeGraphPass();

// Registers every Analog conversion pass exposed to textual pipelines and
// pass managers.
void registerAnalogConversionPasses();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_PASSES_H
