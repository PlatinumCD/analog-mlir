#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PASSES_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PASSES_H

namespace mlir {
namespace analog {

// Registers every Analog transform pass exposed to textual pipelines and pass
// managers.
void registerAnalogPasses();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PASSES_H
