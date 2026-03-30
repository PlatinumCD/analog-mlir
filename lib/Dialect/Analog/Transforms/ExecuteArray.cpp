#include "analog-mlir/Dialect/Analog/Transforms/ExecuteArray.h"
#include "analog-mlir/Dialect/Analog/Transforms/SourceTrackingUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformAttrs.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/DialectRegistry.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace analog {

using detail::kMatmulExecIdAttr;
using detail::collectMatrixGridsBySourceId;
using detail::getOrInferMatmulSourceId;

struct MatmulExecutionPlan {
  analog::MatrixGridType gridTy;
  int64_t arrayRows;
  int64_t numArrayRows;
  int64_t numArrayCols;
};

// Resolves the execution plan for a matmul tagged with a matrix source
// id and reports whether this matmul should be rewritten.

static FailureOr<std::optional<MatmulExecutionPlan>> buildExecutionPlan(
    linalg::MatmulOp op,
    const llvm::DenseMap<int64_t, analog::MatrixGridType>
        &gridByMatrixSourceId) {
  if (op.getInputs().size() < 2) {
    op.emitError("expected matmul with two inputs");
    return failure();
  }

  auto matrixSourceId = getOrInferMatmulSourceId(op);
  if (!matrixSourceId) {
    return std::optional<MatmulExecutionPlan>{};
  }
  auto it = gridByMatrixSourceId.find(matrixSourceId.getInt());
  if (it == gridByMatrixSourceId.end()) {
    op.emitError("could not find analog matrix partition for matmul RHS");
    return failure();
  }

  analog::MatrixGridType gridTy = it->second;
  auto arrayShape = gridTy.getArrayShape();
  if (arrayShape.size() != 2) {
    op.emitError("expected matrix grid array shape to be rank-2");
    return failure();
  }

  auto gridShape = gridTy.getGridShape();
  if (gridShape.size() != 2) {
    op.emitError("expected matrix grid shape to be rank-2");
    return failure();
  }

  return std::optional<MatmulExecutionPlan>{MatmulExecutionPlan{
      gridTy,
      arrayShape[0],
      gridShape[0],
      gridShape[1],
  }};
}


// Allocates the execution buffer for one matmul and tags both the
// buffer and source matmul with a shared execution id.

static Value allocateArrayOutputBuffers(OpBuilder &builder, Location loc,
                                        int64_t numArrayRows,
                                        int64_t numArrayCols,
                                        int64_t arrayRows, int64_t execId,
                                        linalg::MatmulOp op) {
  auto arrayOutputBuffersTy =
      mlir::MemRefType::get({numArrayRows, numArrayCols, arrayRows},
                            builder.getF32Type());
  Value arrayOutputBuffers =
      builder.create<memref::AllocOp>(loc, arrayOutputBuffersTy);

  // Tag both the produced buffer and the source matmul so ReduceResults can
  // reconnect them without relying on neighboring ops in the block.
  auto arrayOutputBuffersOp =
      arrayOutputBuffers.getDefiningOp<memref::AllocOp>();
  arrayOutputBuffersOp->setAttr(kMatmulExecIdAttr,
                                builder.getI64IntegerAttr(execId));
  op->setAttr(kMatmulExecIdAttr, builder.getI64IntegerAttr(execId));
  return arrayOutputBuffers;
}


// Emits the nested loops that execute each placed array and store its
// output into the allocated buffer grid.

static void emitExecuteAndStoreLoops(OpBuilder &builder, Location loc,
                                     analog::MatrixGridType gridTy,
                                     Value arrayOutputBuffers,
                                     int64_t numArrayRows,
                                     int64_t numArrayCols) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value ubArrayRows = builder.create<arith::ConstantIndexOp>(loc, numArrayRows);
  Value ubArrayCols = builder.create<arith::ConstantIndexOp>(loc, numArrayCols);

  builder.create<scf::ForOp>(
      loc, zero, ubArrayRows, one, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value tr, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, zero, ubArrayCols, one, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value tc, ValueRange) {
              Value array =
                  colBuilder.create<analog::ArrayExecuteOp>(colLoc, gridTy,
                                                            ValueRange{tr, tc});
              colBuilder.create<analog::ArrayStoreOp>(
                  colLoc, array, arrayOutputBuffers, ValueRange{tr, tc});
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}


// Rewrites one tagged matmul into array execution loops using the
// precomputed execution plan for its matrix grid.

static LogicalResult rewriteTaggedMatmul(OpBuilder &builder, linalg::MatmulOp op,
                                         const MatmulExecutionPlan &plan,
                                         int64_t execId) {
  Location loc = op.getLoc();
  Value arrayOutputBuffers = allocateArrayOutputBuffers(
      builder, loc, plan.numArrayRows, plan.numArrayCols, plan.arrayRows,
      execId, op);
  emitExecuteAndStoreLoops(builder, loc, plan.gridTy, arrayOutputBuffers,
                           plan.numArrayRows, plan.numArrayCols);
  return success();
}


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef ExecuteArrayPass::getArgument() const {
  return "analog-execute-array";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef ExecuteArrayPass::getDescription() const {
  return "Insert ExecuteArray ops";
}


// Rewrites tagged matmuls into analog array execution loops and
// allocates the buffers needed to capture per-array outputs.

void ExecuteArrayPass::runOnOperation() {
  auto func = getOperation();
  auto gridByMatrixSourceId = collectMatrixGridsBySourceId(func);
  bool hadError = false;

  int64_t nextMatmulExecId = 0;
  func.walk([&](mlir::linalg::MatmulOp op) {
    if (hadError) {
      return;
    }

    FailureOr<std::optional<MatmulExecutionPlan>> maybePlan =
        buildExecutionPlan(op, gridByMatrixSourceId);
    if (failed(maybePlan)) {
      hadError = true;
      return;
    }
    if (!*maybePlan) {
      return;
    }

    OpBuilder builder(op);
    builder.setInsertionPoint(op);
    if (failed(rewriteTaggedMatmul(builder, op, **maybePlan,
                                   nextMatmulExecId))) {
      hadError = true;
      return;
    }
    ++nextMatmulExecId;
  });

  if (hadError) {
    signalPassFailure();
  }
}


// Declares the analog dialect required for the execute and store ops
// inserted by this pass.

void ExecuteArrayPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createExecuteArrayPass() {
  return std::make_unique<ExecuteArrayPass>();
}


} // namespace analog
} // namespace mlir
