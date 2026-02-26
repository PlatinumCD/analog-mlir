#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformUtils.h"

#include "llvm/Support/Casting.h"
#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>

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

    auto tiling = detail::computeGridTiling2D(
      matrixRows, matrixCols, arrayRows, arrayCols);

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto arrayGridTy = analog::MatrixGridType::get(
      builder.getContext(),
      {tiling.rows, tiling.cols},
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
