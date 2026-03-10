#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformUtils.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <utility>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

enum class MatmulMatchStatus {
  NoMatch,
  Match,
  Error,
};


// Returns the vector type only for values already materialized as
// analog vectors.

analog::VectorType getPartitionableVectorType(Value value) {
  return llvm::dyn_cast<analog::VectorType>(value.getType());
}


// Finds the unique matmul that consumes the original tensor input for
// this vector materialization.

MatmulMatchStatus getUniqueInputMatmulUser(analog::VectorFromTensorOp op,
                                           linalg::MatmulOp &matchedMatmul) {
  SmallVector<linalg::MatmulOp> matmulUsers;
  for (Operation *user : op.getInput().getUsers()) {
    auto matmulOp = llvm::dyn_cast<linalg::MatmulOp>(user);
    if (!matmulOp) {
      continue;
    }
    if (matmulOp.getInputs().empty() || matmulOp.getInputs()[0] != op.getInput()) {
      continue;
    }
    matmulUsers.push_back(matmulOp);
  }

  if (matmulUsers.empty()) {
    return MatmulMatchStatus::NoMatch;
  }
  if (matmulUsers.size() != 1) {
    op.emitError("expected exactly one matmul user of vector input for partitioning");
    return MatmulMatchStatus::Error;
  }

  matchedMatmul = matmulUsers.front();
  return MatmulMatchStatus::Match;
}


// Extracts the matrix dimensions that determine the shared array-grid
// tiling for the vector partition.

FailureOr<std::pair<int64_t, int64_t>>
getPartitionedMatrixShape(analog::VectorFromTensorOp op, linalg::MatmulOp matmulOp) {
  Value matrixTransposeInput = matmulOp.getInputs()[1];
  auto matrixTransposeInputTy =
      llvm::dyn_cast<RankedTensorType>(matrixTransposeInput.getType());
  if (!matrixTransposeInputTy || matrixTransposeInputTy.getRank() != 2) {
    op.emitError("expected rank-2 matrix operand on matched matmul");
    return failure();
  }

  auto matrixTransposeShape = matrixTransposeInputTy.getShape();
  return std::make_pair(matrixTransposeShape[1], matrixTransposeShape[0]);
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef PartitionVectorPass::getArgument() const {
  return "analog-partition-vector";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef PartitionVectorPass::getDescription() const {
  return "Partition analog vectors into vector-slice views derived from tiling geometry";
}


// Partitions each eligible analog vector using the matched matrix
// geometry so both operands share the same array grid.

void PartitionVectorPass::runOnOperation() {
  auto func = getOperation();
  bool hadError = false;

  func.walk([&](analog::VectorFromTensorOp op) {
    if (hadError) {
      return;
    }

    Value output = op.getResult();
    analog::VectorType vectorTy = getPartitionableVectorType(output);
    if (!vectorTy) {
      return;
    }

    int64_t arrayRows = array_rows;
    int64_t arrayCols = array_cols;

    auto vectorShape = vectorTy.getShape();
    if (vectorShape.size() != 2) {
      op.emitError("expected rank-2 analog vector type");
      hadError = true;
      return;
    }

    linalg::MatmulOp matchedMatmul;
    MatmulMatchStatus matmulMatch = getUniqueInputMatmulUser(op, matchedMatmul);
    if (matmulMatch == MatmulMatchStatus::NoMatch) {
      return;
    }
    if (matmulMatch == MatmulMatchStatus::Error) {
      hadError = true;
      return;
    }

    FailureOr<std::pair<int64_t, int64_t>> maybeMatrixShape =
        getPartitionedMatrixShape(op, matchedMatmul);
    if (failed(maybeMatrixShape)) {
      hadError = true;
      return;
    }

    auto [matrixRows, matrixCols] = *maybeMatrixShape;

    auto tiling = detail::computeGridTiling2D(
      matrixRows, matrixCols, arrayRows, arrayCols);

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto vectorSliceTy = analog::VectorSliceType::get(
      builder.getContext(),
      {tiling.rows, tiling.cols},
      {arrayRows, arrayCols},
      vectorTy
    );

    builder.create<analog::VectorPartitionOp>(
      op.getLoc(),
      vectorSliceTy,
      op.getResult()
    );
  });

  if (hadError) {
    signalPassFailure();
  }
}


// Declares the analog dialect required for the vector partition op
// inserted by this pass.

void PartitionVectorPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass using the default array
// dimensions.

std::unique_ptr<mlir::Pass> createPartitionVectorPass() {
  return std::make_unique<PartitionVectorPass>();
}


// Builds a new instance of the pass with explicit array dimensions for
// pipeline construction.

std::unique_ptr<mlir::Pass> createPartitionVectorPass(int64_t arrayRows, int64_t arrayCols) {
  auto pass = std::make_unique<PartitionVectorPass>();
  pass->array_rows = arrayRows;
  pass->array_cols = arrayCols;
  return pass;
}
} // namespace analog
} // namespace mlir
