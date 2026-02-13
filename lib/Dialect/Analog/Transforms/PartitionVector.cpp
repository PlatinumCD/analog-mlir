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
  return "Partition analog vectors into varray-slice views derived from tiling geometry";
}

void PartitionVectorPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::VectorFromTensorOp op) {
    Value output = op.getResult();
    auto vectorTy = llvm::dyn_cast<analog::VectorType>(output.getType());
    if (!vectorTy) {
      return;
    }

    int64_t arrayRows = array_rows;
    int64_t arrayCols = array_cols;

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

    int64_t numArrayRows = (matrixRows + arrayRows - 1) / arrayRows;
    int64_t numArrayCols = (matrixCols + arrayCols - 1) / arrayCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto varraySliceTy = analog::VectorSliceType::get(
      builder.getContext(),
      {numArrayRows, numArrayCols},
      {arrayRows, arrayCols},
      vectorTy
    );

    builder.create<analog::VectorPartitionOp>(
      op.getLoc(),
      varraySliceTy,
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

std::unique_ptr<mlir::Pass> createPartitionVectorPass(int64_t arrayRows, int64_t arrayCols) {
  auto pass = std::make_unique<PartitionVectorPass>();
  pass->array_rows = arrayRows;
  pass->array_cols = arrayCols;
  return pass;
}

} // namespace analog
} // namespace mlir
