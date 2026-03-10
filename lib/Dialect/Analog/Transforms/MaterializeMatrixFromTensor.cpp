#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"
#include <algorithm>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>


using namespace mlir;

namespace mlir {
namespace analog {

namespace {

constexpr StringLiteral kMatrixSourceIdAttr = "analog.matrix_source_id";


// Returns the ranked tensor type only for rank-2 floating-point tensor
// constants that can become analog matrices.

RankedTensorType getMaterializableMatrixTensorType(arith::ConstantOp op) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(op.getType());
  if (!tensorTy || tensorTy.getRank() != 2) {
    return {};
  }
  if (!llvm::isa<FloatType>(tensorTy.getElementType())) {
    return {};
  }

  return tensorTy;
}


// Finds the next free matrix source id so newly discovered matrix constants
// can participate in later analog execution passes.
int64_t getNextMatrixSourceId(func::FuncOp func) {
  int64_t nextMatrixSourceId = 0;
  func.walk([&](arith::ConstantOp op) {
    auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId)
      return;

    nextMatrixSourceId =
        std::max(nextMatrixSourceId, matrixSourceId.getInt() + 1);
  });
  return nextMatrixSourceId;
}


// Ensures each materializable matrix constant has a stable source id so
// partitioning and execution can reconnect it to later matmuls.
IntegerAttr getOrCreateMatrixSourceId(arith::ConstantOp op,
                                      int64_t &nextMatrixSourceId) {
  if (auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr))
    return matrixSourceId;

  auto matrixSourceId = IntegerAttr::get(
      IntegerType::get(op.getContext(), 64), nextMatrixSourceId++);
  op->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  return matrixSourceId;
}


// Copies the matrix source id onto the newly inserted materialization
// op so later passes can keep tracking it.

void propagateMatrixSourceId(arith::ConstantOp op, Operation *materializedOp) {
  if (!materializedOp) {
    return;
  }

  if (auto matrixSourceId = op->getAttr(kMatrixSourceIdAttr)) {
    materializedOp->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  }
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef MaterializeMatrixFromTensorPass::getArgument() const {
  return "analog-materialize-matrix";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef MaterializeMatrixFromTensorPass::getDescription() const {
  return "Transform dense resources into analog matrix types";
}


// Converts eligible rank-2 floating-point tensor constants into analog
// matrix materializations and forwards their source ids.

void MaterializeMatrixFromTensorPass::runOnOperation() {
  auto func = getOperation();
  int64_t nextMatrixSourceId = getNextMatrixSourceId(func);

  func.walk([&](arith::ConstantOp op) {
    RankedTensorType tensorTy = getMaterializableMatrixTensorType(op);
    if (!tensorTy) {
      return;
    }

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto matrixTy = analog::MatrixType::get(
        builder.getContext(),
        tensorTy.getShape(),
        tensorTy.getElementType()
    );

    auto materialized = builder.create<analog::MatrixFromTensorOp>(
        op.getLoc(),
        matrixTy,
        op.getResult()
    );
    getOrCreateMatrixSourceId(op, nextMatrixSourceId);
    propagateMatrixSourceId(op, materialized.getOperation());
  });
}


// Declares the analog dialect required for the matrix materialization
// op inserted by this pass.

void MaterializeMatrixFromTensorPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createMaterializeMatrixFromTensorPass() {
  return std::make_unique<MaterializeMatrixFromTensorPass>();
}


} // namespace analog
} // namespace mlir
