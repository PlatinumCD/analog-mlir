#include "analog-mlir/Dialect/Analog/Transforms/ReduceResults.h"
#include "analog-mlir/Dialect/Analog/Transforms/SourceTrackingUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformAttrs.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

struct ReductionPlan {
  analog::MatrixGridType gridTy;
  Value resultBuffer;
  int64_t gridRows;
  int64_t gridCols;
  int64_t matrixRows;
  int64_t laneWidth;
};

struct ReductionConstants {
  Value c0;
  Value c1;
  Value c0f;
};

// Builds a lookup from execution ids to the allocated result buffers
// created by the execute-array pass.

static llvm::DenseMap<int64_t, Value>
collectResultBuffersByExecId(func::FuncOp func) {
  llvm::DenseMap<int64_t, Value> resultBufferByExecId;
  func.walk([&](memref::AllocOp op) {
    auto execId = op->getAttrOfType<IntegerAttr>(kMatmulExecIdAttr);
    if (!execId) {
      return;
    }
    resultBufferByExecId.try_emplace(execId.getInt(), op.getResult());
    op->removeAttr(kMatmulExecIdAttr);
  });
  return resultBufferByExecId;
}

// Resolves the grid and buffer inputs needed to reduce one tagged
// matmul result back into a logical tensor.

static FailureOr<std::optional<ReductionPlan>> buildReductionPlan(
    linalg::MatmulOp op,
    const llvm::DenseMap<int64_t, analog::MatrixGridType> &gridByMatrixSourceId,
    const llvm::DenseMap<int64_t, Value> &resultBufferByExecId) {
  if (op.getInputs().size() < 2) {
    op.emitError("expected matmul with two inputs");
    return failure();
  }

  auto matrixSourceId = detail::getMatrixSourceIdAttr(op);
  if (!matrixSourceId) {
    return std::optional<ReductionPlan>{};
  }

  auto gridIt = gridByMatrixSourceId.find(matrixSourceId.getInt());
  if (gridIt == gridByMatrixSourceId.end()) {
    op.emitError("could not find analog matrix partition for matmul RHS");
    return failure();
  }
  analog::MatrixGridType gridTy = gridIt->second;

  auto gridShape = gridTy.getGridShape();
  if (gridShape.size() != 2) {
    op.emitError("expected matrix grid shape to be rank-2");
    return failure();
  }
  int64_t gridRows = gridShape[0];
  int64_t gridCols = gridShape[1];

  auto matrixTy = gridTy.getMatrix();
  auto matrixShape = matrixTy.getShape();
  if (matrixShape.size() != 2) {
    op.emitError("expected matrix type to be rank-2");
    return failure();
  }
  int64_t matrixRows = matrixShape[0];

  auto execId = op->getAttrOfType<IntegerAttr>(kMatmulExecIdAttr);
  if (!execId) {
    op.emitError("matmul is missing analog execution id");
    return failure();
  }

  auto memrefIt = resultBufferByExecId.find(execId.getInt());
  if (memrefIt == resultBufferByExecId.end()) {
    op.emitError("could not find analog result buffer for matmul");
    return failure();
  }
  Value resultBuffer = memrefIt->second;

  auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(resultBuffer.getType());
  if (!memrefTy) {
    op.emitError("expected analog result buffer to be a memref");
    return failure();
  }
  auto memrefShape = memrefTy.getShape();
  if (memrefShape.size() != 3) {
    op.emitError("expected memref<gridRows x gridCols x lanes x elem> result buffer");
    return failure();
  }
  int64_t laneWidth = memrefShape[2];

  return std::optional<ReductionPlan>{ReductionPlan{
      gridTy,
      resultBuffer,
      gridRows,
      gridCols,
      matrixRows,
      laneWidth,
  }};
}


// Builds the constants reused across the reduction loops for one
// rewritten matmul.

static ReductionConstants buildReductionConstants(OpBuilder &builder,
                                                  Location loc) {
  return ReductionConstants{
      builder.create<arith::ConstantIndexOp>(loc, 0),
      builder.create<arith::ConstantIndexOp>(loc, 1),
      builder.create<arith::ConstantFloatOp>(
          loc, builder.getF32Type(), llvm::APFloat(0.0f)),
  };
}


// Allocates and zero-initializes the temporary row buffers used to
// accumulate reduced array results.

static Value allocateZeroedRowBuffers(OpBuilder &builder, Location loc,
                                      const ReductionPlan &plan,
                                      const ReductionConstants &constants) {
  auto rowBufTy =
      MemRefType::get({plan.gridRows, plan.laneWidth}, builder.getF32Type());
  Value rowBufs = builder.create<memref::AllocOp>(loc, rowBufTy);
  Value cGridRows = builder.create<arith::ConstantIndexOp>(loc, plan.gridRows);
  Value cLane = builder.create<arith::ConstantIndexOp>(loc, plan.laneWidth);

  builder.create<scf::ForOp>(
      loc, constants.c0, cGridRows, constants.c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value r, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, constants.c0, cLane, constants.c1, ValueRange{},
            [&](OpBuilder &laneBuilder, Location laneLoc, Value j, ValueRange) {
              laneBuilder.create<memref::StoreOp>(laneLoc, constants.c0f, rowBufs,
                                                  ValueRange{r, j});
              laneBuilder.create<scf::YieldOp>(laneLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
  return rowBufs;
}


// Sums the per-array partial results across each grid row into the
// temporary reduction buffers.

static void emitRowReduction(OpBuilder &builder, Location loc, Value rowBufs,
                             const ReductionPlan &plan,
                             const ReductionConstants &constants) {
  Value cGridRows = builder.create<arith::ConstantIndexOp>(loc, plan.gridRows);
  Value cGridCols = builder.create<arith::ConstantIndexOp>(loc, plan.gridCols);
  Value cLane = builder.create<arith::ConstantIndexOp>(loc, plan.laneWidth);

  builder.create<scf::ForOp>(
      loc, constants.c0, cGridRows, constants.c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value r, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, constants.c0, cGridCols, constants.c1, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value c, ValueRange) {
              colBuilder.create<scf::ForOp>(
                  colLoc, constants.c0, cLane, constants.c1, ValueRange{},
                  [&](OpBuilder &laneBuilder, Location laneLoc, Value j,
                      ValueRange) {
                    Value acc = laneBuilder.create<memref::LoadOp>(
                        laneLoc, rowBufs, ValueRange{r, j});
                    Value val = laneBuilder.create<memref::LoadOp>(
                        laneLoc, plan.resultBuffer, ValueRange{r, c, j});
                    Value sum =
                        laneBuilder.create<arith::AddFOp>(laneLoc, acc, val);
                    laneBuilder.create<memref::StoreOp>(
                        laneLoc, sum, rowBufs, ValueRange{r, j});
                    laneBuilder.create<scf::YieldOp>(laneLoc);
                  });
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}


// Materializes the reduced row buffers back into a rank-2 tensor while
// trimming away any padded lanes.

static Value materializeReducedTensor(OpBuilder &builder, Location loc,
                                      Value rowBufs, const ReductionPlan &plan,
                                      const ReductionConstants &constants) {
  auto f32Ty = builder.getF32Type();
  auto outTy = MemRefType::get({1, plan.matrixRows}, f32Ty);
  Value out = builder.create<memref::AllocOp>(loc, outTy);
  Value cGridRows = builder.create<arith::ConstantIndexOp>(loc, plan.gridRows);
  Value cLane = builder.create<arith::ConstantIndexOp>(loc, plan.laneWidth);
  Value cMatrixRows =
      builder.create<arith::ConstantIndexOp>(loc, plan.matrixRows);

  builder.create<scf::ForOp>(
      loc, constants.c0, cGridRows, constants.c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value r, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, constants.c0, cLane, constants.c1, ValueRange{},
            [&](OpBuilder &laneBuilder, Location laneLoc, Value j, ValueRange) {
              Value v = laneBuilder.create<memref::LoadOp>(
                  laneLoc, rowBufs, ValueRange{r, j});
              Value colOffset =
                  laneBuilder.create<arith::MulIOp>(laneLoc, r, cLane);
              Value col =
                  laneBuilder.create<arith::AddIOp>(laneLoc, colOffset, j);
              Value inBounds = laneBuilder.create<arith::CmpIOp>(
                  laneLoc, arith::CmpIPredicate::slt, col, cMatrixRows);
              laneBuilder.create<scf::IfOp>(
                  laneLoc, inBounds,
                  [&](OpBuilder &ifBuilder, Location ifLoc) {
                    ifBuilder.create<memref::StoreOp>(ifLoc, v, out,
                                                      ValueRange{constants.c0, col});
                    ifBuilder.create<scf::YieldOp>(ifLoc);
                  });
              laneBuilder.create<scf::YieldOp>(laneLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });

  auto resultTy = RankedTensorType::get({1, plan.matrixRows}, f32Ty);
  auto toTensor = builder.create<bufferization::ToTensorOp>(loc, resultTy, out);
  toTensor->setAttr("restrict", builder.getUnitAttr());
  return toTensor.getResult();
}


// Rewrites one tagged matmul by reducing its per-array execution
// buffers back into the final logical tensor result.

static LogicalResult rewriteReducedMatmul(OpBuilder &builder, linalg::MatmulOp op,
                                          const ReductionPlan &plan) {
  Location loc = op.getLoc();
  ReductionConstants constants = buildReductionConstants(builder, loc);
  Value rowBufs = allocateZeroedRowBuffers(builder, loc, plan, constants);
  emitRowReduction(builder, loc, rowBufs, plan, constants);
  materializeReducedTensor(builder, loc, rowBufs, plan, constants);
  return success();
}


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef ReduceResultsPass::getArgument() const {
  return "analog-reduce-results";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef ReduceResultsPass::getDescription() const {
  return "Reduce array outputs into final tensor results";
}


// Reduces the per-array execution buffers associated with tagged
// matmuls back into their final tensor results.

void ReduceResultsPass::runOnOperation() {
  auto func = getOperation();
  auto gridByMatrixSourceId = collectMatrixGridsBySourceId(func);
  auto resultBufferByExecId = collectResultBuffersByExecId(func);
  bool hadError = false;

  func.walk([&](mlir::linalg::MatmulOp op) {
    if (hadError) {
      return;
    }

    OpBuilder builder(op);
    builder.setInsertionPoint(op);
    FailureOr<std::optional<ReductionPlan>> maybePlan =
        buildReductionPlan(op, gridByMatrixSourceId, resultBufferByExecId);
    if (failed(maybePlan)) {
      hadError = true;
      return;
    }
    if (!*maybePlan) {
      return;
    }
    if (failed(rewriteReducedMatmul(builder, op, **maybePlan))) {
      hadError = true;
    }
  });

  if (hadError) {
    signalPassFailure();
  }
}


// Declares the dialects this pass may create while reducing memref
// buffers back into tensor values.

void ReduceResultsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createReduceResultsPass() {
  return std::make_unique<ReduceResultsPass>();
}


} // namespace analog
} // namespace mlir
