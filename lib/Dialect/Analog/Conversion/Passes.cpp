#include "analog-mlir/Dialect/Analog/Conversion/Passes.h"
#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"
#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToDebugShims.h"
#include "analog-mlir/Dialect/Analog/Conversion/FinalizeGolemIntrinsics.h"

#include <mlir/Pass/PassRegistry.h>

namespace mlir {
namespace analog {

void registerAnalogConversionPasses() {
  PassRegistration<ConvertAnalogToGolemBackendPass>();
  PassRegistration<ConvertAnalogToDebugShimsPass>();
  PassRegistration<FinalizeGolemIntrinsicsPass>();
}

} // namespace analog
} // namespace mlir
