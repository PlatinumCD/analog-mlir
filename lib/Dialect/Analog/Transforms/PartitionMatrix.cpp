#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformAttrs.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformUtils.h"

#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

using detail::kMatrixSourceIdAttr;


// Returns the matrix type only for values already materialized as
// analog matrices.

analog::MatrixType getPartitionableMatrixType(Value value) {
  return llvm::dyn_cast<analog::MatrixType>(value.getType());
}


// Finds the next free matrix source id so existing matrix materializations
// can be linked to later matmul execution.
int64_t getNextMatrixSourceId(func::FuncOp func) {
  int64_t nextMatrixSourceId = 0;
  func.walk([&](Operation *op) {
    auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId)
      return;

    nextMatrixSourceId =
        std::max(nextMatrixSourceId, matrixSourceId.getInt() + 1);
  });
  return nextMatrixSourceId;
}


// Ensures the source matrix and its partition share a stable source id even
// when the IR already contained handwritten matrix materializations.
IntegerAttr getOrCreateMatrixSourceId(analog::MatrixFromTensorOp op,
                                      int64_t &nextMatrixSourceId) {
  if (auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr))
    return matrixSourceId;

  if (auto constantOp = op.getInput().getDefiningOp<arith::ConstantOp>()) {
    if (auto matrixSourceId =
            constantOp->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr)) {
      op->setAttr(kMatrixSourceIdAttr, matrixSourceId);
      return matrixSourceId;
    }
  }

  auto matrixSourceId = IntegerAttr::get(
      IntegerType::get(op.getContext(), 64), nextMatrixSourceId++);
  op->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  if (auto constantOp = op.getInput().getDefiningOp<arith::ConstantOp>())
    constantOp->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  return matrixSourceId;
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef PartitionMatrixPass::getArgument() const {
  return "analog-partition-matrix";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef PartitionMatrixPass::getDescription() const {
  return "Partition analog matrices into array-grid views using configurable array dimensions";
}


// Partitions each analog matrix into a grid type sized by the
// configured array dimensions and forwards its source id.

void PartitionMatrixPass::runOnOperation() {
  auto func = getOperation();
  int64_t nextMatrixSourceId = getNextMatrixSourceId(func);

  func.walk([&](analog::MatrixFromTensorOp op) {
    Value output = op.getResult();
    analog::MatrixType matrixTy = getPartitionableMatrixType(output);
    if (!matrixTy) {
      return;
    }

    int64_t arrayRows  = array_rows;
    int64_t arrayCols  = array_cols;

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

    auto partition = builder.create<analog::MatrixPartitionOp>(
      op.getLoc(),
      arrayGridTy,
      op.getResult()
    );
    partition->setAttr(kMatrixSourceIdAttr,
                       getOrCreateMatrixSourceId(op, nextMatrixSourceId));
  });
}


// Declares the analog dialect required for the matrix partition op
// inserted by this pass.

void PartitionMatrixPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass using the default array
// dimensions.

std::unique_ptr<mlir::Pass> createPartitionMatrixPass() {
  return std::make_unique<PartitionMatrixPass>();
}


// Builds a new instance of the pass with explicit array dimensions for
// pipeline construction.

std::unique_ptr<mlir::Pass> createPartitionMatrixPass(int64_t arrayRows, int64_t arrayCols) {
  auto pass = std::make_unique<PartitionMatrixPass>();
  pass->array_rows = arrayRows;
  pass->array_cols = arrayCols;
  return pass;
}
} // namespace analog
} // namespace mlir
