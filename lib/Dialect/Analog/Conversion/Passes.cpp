#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"
#include "analog-mlir/Dialect/Analog/Conversion/EmitRuntimeGraph.h"
#include "analog-mlir/Dialect/Analog/Conversion/Passes.h"

namespace mlir {
namespace analog {

// Registers the conversion pass bundle exposed by this library entry point.
void registerAnalogConversionPasses() {
  registerConvertAnalogToGolemBackendPass();
  registerEmitRuntimeGraphPass();
}

} // namespace analog
} // namespace mlir
