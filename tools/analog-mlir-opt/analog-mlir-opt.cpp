

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char** argv) {


  DialectRegistry registry;
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Analog MLIR modular optimizer\n", registry));

}
