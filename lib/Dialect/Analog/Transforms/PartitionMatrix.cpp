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
  return "TODO";
}

void PartitionMatrixPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::MatrixFromTensorOp op) {

    Value output = op.getResult();
    auto matrixTy = llvm::dyn_cast<analog::MatrixType>(output.getType());
    if (!matrixTy) {
      return;
    }

    int64_t tileRows   = tile_rows;
    int64_t tileCols   = tile_cols;

    auto matrixShape = matrixTy.getShape();
    int64_t matrixRows = matrixShape[0];
    int64_t matrixCols = matrixShape[1];

    int64_t numTileRows = (matrixRows + tileRows - 1) / tileRows;
    int64_t numTileCols = (matrixCols + tileCols - 1) / tileCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto tileGridTy = analog::TileGridType::get(
      builder.getContext(),
      {numTileRows, numTileCols},
      {tileRows, tileCols},
      matrixTy
    );

    builder.create<analog::TilePartitionOp>(
      op.getLoc(),
      tileGridTy,
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

std::unique_ptr<mlir::Pass> createPartitionMatrixPass(int64_t tileRows, int64_t tileCols) {
  auto pass = std::make_unique<PartitionMatrixPass>();
  pass->tile_rows = tileRows;
  pass->tile_cols = tileCols;
  return pass;
}

} // namespace analog
} // namespace mlir
