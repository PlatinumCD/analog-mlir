#include "analog-mlir/Dialect/Analog/IR/AnalogDialect.h"
#include "analog-mlir/Transforms/NN/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char** argv) {
  DialectRegistry registry;

  registerAllDialects(registry);
  registry.insert<analog::AnalogDialect>();

  mlir::nn::registerNNPasses();
//  registerAllPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Analog MLIR modular optimizer\n", registry));

}
