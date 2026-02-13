#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>


using namespace mlir;

namespace mlir {
namespace analog {

// =====--------------------------------=====
//   PartitionMatrixPass - Pass
// =====--------------------------------=====

llvm::StringRef PartitionMatrixPass::getArgument() const {
  return "analog-partition-matrix";
}

llvm::StringRef PartitionMatrixPass::getDescription() const {
  return "Partition analog matrices into array-grid views using configurable array dimensions";
}

void PartitionMatrixPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::MatrixFromTensorOp op) {

    Value output = op.getResult();
    auto matrixTy = llvm::dyn_cast<analog::MatrixType>(output.getType());
    if (!matrixTy) {
      return;
    }

    int64_t arrayRows   = array_rows;
    int64_t arrayCols   = array_cols;

    auto matrixShape = matrixTy.getShape();
    int64_t matrixRows = matrixShape[0];
    int64_t matrixCols = matrixShape[1];

    int64_t numArrayRows = (matrixRows + arrayRows - 1) / arrayRows;
    int64_t numArrayCols = (matrixCols + arrayCols - 1) / arrayCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto arrayGridTy = analog::MatrixGridType::get(
      builder.getContext(),
      {numArrayRows, numArrayCols},
      {arrayRows, arrayCols},
      matrixTy
    );

    builder.create<analog::MatrixPartitionOp>(
      op.getLoc(),
      arrayGridTy,
      op.getResult()
    );
  });
}

void PartitionMatrixPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createPartitionMatrixPass() {
  return std::make_unique<PartitionMatrixPass>();
}

std::unique_ptr<mlir::Pass> createPartitionMatrixPass(int64_t arrayRows, int64_t arrayCols) {
  auto pass = std::make_unique<PartitionMatrixPass>();
  pass->array_rows = arrayRows;
  pass->array_cols = arrayCols;
  return pass;
}

} // namespace analog
} // namespace mlir
