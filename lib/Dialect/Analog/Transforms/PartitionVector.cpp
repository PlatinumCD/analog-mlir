#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
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
//   PartitionVectorPass - Pass
// =====--------------------------------=====

llvm::StringRef PartitionVectorPass::getArgument() const {
  return "analog-partition-vector";
}

llvm::StringRef PartitionVectorPass::getDescription() const {
  return "TODO";
}

void PartitionVectorPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::VectorFromTensorOp op) {
    Value output = op.getResult();
    auto vectorTy = llvm::dyn_cast<analog::VectorType>(output.getType());
    if (!vectorTy) {
      return;
    }

    int64_t tileRows = tile_rows;
    int64_t tileCols = tile_cols;

    Operation *next = op->getNextNode();
    if (!next) {
      return;
    }

    auto matmulOp = llvm::dyn_cast<linalg::MatmulOp>(next);
    if (!matmulOp) {
      return;
    }

    Value matrixTransposeInput = matmulOp.getInputs()[1]; 
    auto matrixTansposeInputTy = llvm::dyn_cast<RankedTensorType>(matrixTransposeInput.getType());
    if (!matrixTansposeInputTy) {
      return;
    }

    auto matrixTransposeShape = matrixTansposeInputTy.getShape();
    int64_t matrixRows = matrixTransposeShape[1];
    int64_t matrixCols = matrixTransposeShape[0];

    int64_t numTileRows = (matrixRows + tileRows - 1) / tileRows;
    int64_t numTileCols = (matrixCols + tileCols - 1) / tileCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto vtileSliceTy = analog::VTileSliceType::get(
      builder.getContext(),
      {numTileRows, numTileCols},
      {tileRows, tileCols},
      vectorTy
    );

    builder.create<analog::VTilePartitionOp>(
      op.getLoc(),
      vtileSliceTy,
      op.getResult()
    );
  });
}

void PartitionVectorPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createPartitionVectorPass() {
  return std::make_unique<PartitionVectorPass>();
}

std::unique_ptr<mlir::Pass> createPartitionVectorPass(int64_t tileRows, int64_t tileCols) {
  auto pass = std::make_unique<PartitionVectorPass>();
  pass->tile_rows = tileRows;
  pass->tile_cols = tileCols;
  return pass;
}

} // namespace analog
} // namespace mlir
