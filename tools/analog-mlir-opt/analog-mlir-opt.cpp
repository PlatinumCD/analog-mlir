#include "analog-mlir/Dialect/Analog/IR/AnalogDialect.h"
#include "analog-mlir/Dialect/Analog/Conversion/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char** argv) {
  DialectRegistry registry;

  registerAllDialects(registry);
  registry.insert<analog::AnalogDialect>();

  mlir::analog::registerAnalogPasses();
  mlir::analog::registerAnalogConversionPasses();
  registerAllPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Analog MLIR modular optimizer\n", registry));

}
