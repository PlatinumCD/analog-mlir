#include "analog-mlir/Dialect/Analog/Conversion/Passes.h"
#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"
#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToDebugShims.h"
#include "analog-mlir/Dialect/Analog/Conversion/FinalizeGolemIntrinsics.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <mlir/Pass/PassRegistry.h>

namespace mlir {
namespace analog {

struct LowerToDebugShimsPipelineOptions
    : public PassPipelineOptions<LowerToDebugShimsPipelineOptions> {};

void registerLowerToDebugShimsPipeline() {
  PassPipelineRegistration<LowerToDebugShimsPipelineOptions>(
      "analog-lower-to-debug-shims",
      "Lower analog IR to the golem backend and rewrite calls to debug shims",
      [](OpPassManager &pm, const LowerToDebugShimsPipelineOptions &) {
        pm.addPass(createConvertAnalogToGolemBackendPass());
        pm.addPass(createConvertAnalogToDebugShimsPass());
      });
}

void registerAnalogConversionPasses() {
  PassRegistration<ConvertAnalogToGolemBackendPass>();
  PassRegistration<ConvertAnalogToDebugShimsPass>();
  PassRegistration<FinalizeGolemIntrinsicsPass>();
  registerLowerToDebugShimsPipeline();
}

} // namespace analog
} // namespace mlir
