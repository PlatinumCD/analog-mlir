#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"
#include "analog-mlir/Dialect/Analog/Conversion/EmitRuntimeGraph.h"
#include "analog-mlir/Dialect/Analog/Conversion/FinalizeGolemIntrinsics.h"
#include "analog-mlir/Dialect/Analog/Conversion/Passes.h"

namespace mlir {
namespace analog {

// Registers the conversion pass bundle exposed by this library entry point.
void registerAnalogConversionPasses() {
  registerConvertAnalogToGolemBackendPass();
  registerFinalizeGolemIntrinsicsPass();
  registerEmitRuntimeGraphPass();
}

} // namespace analog
} // namespace mlir
