#include "analog-mlir/Dialect/Analog/Transforms/ExtractLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorImplementationUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/MatchUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/RewriteUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <memory>
#include <optional>

namespace extractor_impl = mlir::analog::extractor_impl;
namespace extractor_utils = mlir::analog::extractor_utils;
namespace match_utils = mlir::analog::match_utils;
namespace rewrite_utils = mlir::analog::rewrite_utils;

namespace {

using mlir::arith::ConstantOp;
using mlir::func::FuncOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

// Captures one transposed-weight matmul branch of the LSTM pre-activation.
struct LSTMCellMatmulBranchMatch {
  mlir::Value activationInput;
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *transposeEmpty = nullptr;
  mlir::Operation *transpose = nullptr;
  mlir::Operation *matmul = nullptr;
  mlir::Operation *fill = nullptr;
  mlir::Operation *fillConstant = nullptr;
  mlir::Operation *biasAdd = nullptr;
  mlir::Operation *biasConstant = nullptr;
};

// Captures the shared index-building scaffold used to slice the 4H preactivation.
struct LSTMIndexingScaffoldMatch {
  mlir::Operation *zeroIndexConstant = nullptr;
  mlir::Operation *indexExtentConstant = nullptr;
  mlir::Operation *zeroIndexEmpty = nullptr;
  mlir::Operation *zeroIndexGeneric = nullptr;
  mlir::Operation *zeroIndexExpand = nullptr;
  mlir::Operation *baseOffsetEmpty = nullptr;
  mlir::Operation *baseOffsetGeneric = nullptr;
  mlir::Operation *rangeEmpty = nullptr;
  mlir::Operation *rangeGeneric = nullptr;
  mlir::Operation *rangeExpand = nullptr;
  mlir::Operation *combinedIndicesEmpty = nullptr;
  mlir::Operation *combinedIndicesGeneric = nullptr;
  mlir::Operation *collapsedIndices = nullptr;
  mlir::Operation *gatherEmpty = nullptr;
};

// Captures one gate slice from the fused preactivation through activation.
struct LSTMGateSliceMatch {
  mlir::Operation *offsetConstant = nullptr;
  mlir::Operation *offsetAdd = nullptr;
  mlir::Operation *gather = nullptr;
  mlir::Operation *expand = nullptr;
  mlir::Operation *activation = nullptr;
};

// Captures the full LSTM cell slice that will be outlined into a layer func.
struct LSTMCellMatch {
  mlir::Operation *sharedMatmulOutputEmpty = nullptr;
  mlir::Operation *sharedFillOp = nullptr;
  mlir::Operation *sharedFillConstant = nullptr;
  mlir::Operation *preActivationAddOp = nullptr;
  mlir::Operation *preActivationCollapseOp = nullptr;

  LSTMCellMatmulBranchMatch inputBranch;
  LSTMCellMatmulBranchMatch hiddenBranch;

  LSTMIndexingScaffoldMatch indexing;
  LSTMGateSliceMatch inputGate;
  LSTMGateSliceMatch forgetGate;
  LSTMGateSliceMatch candidateGate;
  LSTMGateSliceMatch outputGate;

  mlir::Operation *sharedHiddenOutputEmpty = nullptr;
  mlir::Operation *sigmoidOneConstant = nullptr;
  mlir::Operation *forgetCellMulOp = nullptr;
  mlir::Operation *inputCandidateMulOp = nullptr;
  mlir::Operation *cellAddOp = nullptr;
  mlir::Operation *cellTanhOp = nullptr;
  mlir::Operation *hiddenMulOp = nullptr;

  mlir::Operation *root = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
};

template <typename BodyOpTy>
static bool genericYieldsSingleOpResult(mlir::Operation *op) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic)
    return false;

  mlir::Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return false;

  mlir::Block &block = region.front();
  if (block.empty())
    return false;

  auto it = block.begin();
  auto e = block.end();
  auto bodyOp = llvm::dyn_cast<BodyOpTy>(&*it++);
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  return bodyOp && yield && it == e && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == bodyOp.getResult();
}

static bool isAddiGeneric(mlir::Operation *op) {
  return genericYieldsSingleOpResult<mlir::arith::AddIOp>(op);
}

static bool isMuliGeneric(mlir::Operation *op) {
  return genericYieldsSingleOpResult<mlir::arith::MulIOp>(op);
}

static bool constantOpHasI64Value(mlir::Operation *op, int64_t expected) {
  auto constant = llvm::dyn_cast_or_null<ConstantOp>(op);
  if (!constant)
    return false;

  auto attr = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return attr && attr.getInt() == expected;
}

// Extracts the shared unit constant referenced inside a sigmoid generic body.
static mlir::Operation *getSigmoidUnitConstant(mlir::Operation *op) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic)
    return nullptr;

  mlir::Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return nullptr;

  mlir::Block &block = region.front();
  if (block.empty())
    return nullptr;

  auto it = block.begin();
  auto e = block.end();
  auto neg = llvm::dyn_cast<mlir::arith::NegFOp>(&*it++);
  auto exp = (it != e) ? llvm::dyn_cast<mlir::math::ExpOp>(&*it++)
                       : mlir::math::ExpOp();
  auto add =
      (it != e) ? llvm::dyn_cast<mlir::arith::AddFOp>(&*it++)
                : mlir::arith::AddFOp();
  auto div =
      (it != e) ? llvm::dyn_cast<mlir::arith::DivFOp>(&*it++)
                : mlir::arith::DivFOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!neg || !exp || !add || !div || !yield || it != e)
    return nullptr;

  if (exp.getOperand() != neg.getResult())
    return nullptr;

  mlir::Value unitConstant;
  if (add.getLhs() == exp.getResult())
    unitConstant = add.getRhs();
  else if (add.getRhs() == exp.getResult())
    unitConstant = add.getLhs();
  else
    return nullptr;

  if (div.getLhs() != unitConstant || div.getRhs() != add.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != div.getResult())
    return nullptr;

  return unitConstant.getDefiningOp();
}

// Matches the no-input generic that seeds the gate-index scaffold with zero.
static mlir::LogicalResult
matchYieldingConstantGeneric(mlir::Operation *op,
                             mlir::Operation *expectedConstant) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic)
    return mlir::failure();

  if (!extractor_utils::hasOperands(op, 1) || !extractor_utils::hasInputs(op, 0))
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return mlir::failure();

  mlir::Block &block = region.front();
  if (!llvm::hasSingleElement(block))
    return mlir::failure();

  auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(&block.front());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0).getDefiningOp() != expectedConstant)
    return mlir::failure();

  return mlir::success();
}

// Matches the no-input generic that materializes [0, 1, ..., H-1] indices.
static mlir::LogicalResult matchIndexRangeGeneric(mlir::Operation *op) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic)
    return mlir::failure();

  if (!extractor_utils::hasOperands(op, 1) || !extractor_utils::hasInputs(op, 0))
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return mlir::failure();

  mlir::Block &block = region.front();
  if (block.empty())
    return mlir::failure();

  auto it = block.begin();
  auto e = block.end();
  auto index = llvm::dyn_cast<mlir::linalg::IndexOp>(&*it++);
  auto cast = (it != e) ? llvm::dyn_cast<mlir::arith::IndexCastOp>(&*it++)
                        : mlir::arith::IndexCastOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!index || !cast || !yield || it != e || cast.getIn() != index.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != cast.getResult())
    return mlir::failure();

  return mlir::success();
}

// Matches a one-input generic that multiplies its input by a captured constant.
static mlir::LogicalResult
matchMultiplyByConstantGeneric(mlir::Operation *op,
                               mlir::Operation *expectedConstant) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !isMuliGeneric(op))
    return mlir::failure();

  if (!extractor_utils::hasOperands(op, 2) || !extractor_utils::hasInputs(op, 1))
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  mlir::Block &block = region.front();
  auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(&block.front());
  if (!mul)
    return mlir::failure();

  mlir::Operation *lhsDef = mul.getLhs().getDefiningOp();
  mlir::Operation *rhsDef = mul.getRhs().getDefiningOp();
  if (lhsDef != expectedConstant && rhsDef != expectedConstant)
    return mlir::failure();

  return mlir::success();
}

// Matches a one-input generic that adds a captured offset to each index.
static mlir::LogicalResult
matchOffsetAddGeneric(mlir::Operation *op, mlir::Operation *expectedBaseIndices,
                      mlir::Operation *expectedOutputEmpty, int64_t expectedOffset,
                      mlir::Operation *&offsetConstant) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !isAddiGeneric(op))
    return mlir::failure();

  if (!extractor_utils::hasOperands(op, 2) || !extractor_utils::hasInputs(op, 1))
    return mlir::failure();

  if (extractor_utils::defOp(op, 0) != expectedBaseIndices)
    return mlir::failure();

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 1);
  if (!outputEmpty || outputEmpty.getOperation() != expectedOutputEmpty)
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  mlir::Block &block = region.front();
  auto add = llvm::dyn_cast<mlir::arith::AddIOp>(&block.front());
  if (!add)
    return mlir::failure();

  offsetConstant = add.getLhs().getDefiningOp();
  if (!offsetConstant)
    offsetConstant = add.getRhs().getDefiningOp();
  if (!constantOpHasI64Value(offsetConstant, expectedOffset))
    return mlir::failure();

  return mlir::success();
}

// Matches the generic that gathers one gate slice from the collapsed preactivation.
static mlir::LogicalResult
matchCollapsedExtractGeneric(mlir::Operation *op, mlir::Operation *&indexValueOp,
                             mlir::Operation *&collapsedVectorOp,
                             mlir::Operation *&zeroConst,
                             mlir::Operation *&extentConst,
                             mlir::Operation *&gatherEmpty) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic)
    return mlir::failure();

  if (!extractor_utils::hasOperands(op, 2) || !extractor_utils::hasInputs(op, 1))
    return mlir::failure();

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 1);
  if (!outputEmpty)
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return mlir::failure();

  mlir::Block &block = region.front();
  if (block.empty())
    return mlir::failure();

  auto it = block.begin();
  auto e = block.end();
  auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(&*it++);
  auto add =
      (it != e) ? llvm::dyn_cast<mlir::arith::AddIOp>(&*it++)
                : mlir::arith::AddIOp();
  auto select =
      (it != e) ? llvm::dyn_cast<mlir::arith::SelectOp>(&*it++)
                : mlir::arith::SelectOp();
  auto cast = (it != e) ? llvm::dyn_cast<mlir::arith::IndexCastOp>(&*it++)
                        : mlir::arith::IndexCastOp();
  auto extract = (it != e) ? llvm::dyn_cast<mlir::tensor::ExtractOp>(&*it++)
                           : mlir::tensor::ExtractOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!cmp || !add || !select || !cast || !extract || !yield || it != e)
    return mlir::failure();

  if (cmp.getPredicate() != mlir::arith::CmpIPredicate::slt)
    return mlir::failure();

  mlir::Value wrappedIndex = cmp.getLhs();
  mlir::Value zeroValue = cmp.getRhs();
  mlir::Value extentValue;
  if (add.getLhs() == wrappedIndex)
    extentValue = add.getRhs();
  else if (add.getRhs() == wrappedIndex)
    extentValue = add.getLhs();
  else
    return mlir::failure();

  if (select.getCondition() != cmp.getResult() ||
      select.getTrueValue() != add.getResult() ||
      select.getFalseValue() != wrappedIndex || cast.getIn() != select.getResult() ||
      extract.getIndices().size() != 1 ||
      extract.getIndices().front() != cast.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != extract.getResult())
    return mlir::failure();

  zeroConst = zeroValue.getDefiningOp();
  extentConst = extentValue.getDefiningOp();
  collapsedVectorOp = extract.getTensor().getDefiningOp();
  indexValueOp = extractor_utils::defOp(op, 0);
  gatherEmpty = outputEmpty.getOperation();
  if (!llvm::dyn_cast_or_null<ConstantOp>(zeroConst) ||
      !llvm::dyn_cast_or_null<ConstantOp>(extentConst) ||
      !collapsedVectorOp || !indexValueOp)
    return mlir::failure();

  return mlir::success();
}

// Matches the shared zero/range/index-combine scaffold that all four gates reuse.
static mlir::LogicalResult
matchSharedGateIndexScaffold(CollapseShapeOp collapsedIndices,
                             mlir::Operation *expectedZeroConst,
                             mlir::Operation *expectedExtentConst,
                             LSTMIndexingScaffoldMatch &scaffold) {
  if (scaffold.collapsedIndices) {
    return scaffold.collapsedIndices == collapsedIndices.getOperation() &&
                   scaffold.zeroIndexConstant == expectedZeroConst &&
                   scaffold.indexExtentConstant == expectedExtentConst
               ? mlir::success()
               : mlir::failure();
  }

  mlir::Operation *combinedIndices =
      extractor_utils::defOp(collapsedIndices.getSrc());
  if (!combinedIndices || !isAddiGeneric(combinedIndices))
    return mlir::failure();

  if (!extractor_utils::hasOperands(combinedIndices, 3) ||
      !extractor_utils::hasInputs(combinedIndices, 2))
    return mlir::failure();

  auto combinedIndicesEmpty =
      extractor_utils::defOpAs<EmptyOp>(combinedIndices, 2);
  if (!combinedIndicesEmpty)
    return mlir::failure();

  mlir::Operation *baseOffsetGeneric = extractor_utils::defOp(combinedIndices, 0);
  auto rangeExpand = extractor_utils::defOpAs<ExpandShapeOp>(combinedIndices, 1);
  if (!baseOffsetGeneric || !rangeExpand)
    return mlir::failure();

  mlir::Operation *rangeGeneric = extractor_utils::defOp(rangeExpand.getSrc());
  if (!rangeGeneric || mlir::failed(matchIndexRangeGeneric(rangeGeneric)))
    return mlir::failure();

  auto rangeEmpty = extractor_utils::defOpAs<EmptyOp>(rangeGeneric, 0);
  if (!rangeEmpty)
    return mlir::failure();

  if (mlir::failed(
          matchMultiplyByConstantGeneric(baseOffsetGeneric, expectedExtentConst)))
    return mlir::failure();

  auto baseOffsetEmpty = extractor_utils::defOpAs<EmptyOp>(baseOffsetGeneric, 1);
  auto zeroIndexExpand =
      extractor_utils::defOpAs<ExpandShapeOp>(baseOffsetGeneric, 0);
  if (!baseOffsetEmpty || !zeroIndexExpand)
    return mlir::failure();

  mlir::Operation *zeroIndexGeneric =
      extractor_utils::defOp(zeroIndexExpand.getSrc());
  if (!zeroIndexGeneric ||
      mlir::failed(
          matchYieldingConstantGeneric(zeroIndexGeneric, expectedZeroConst)))
    return mlir::failure();

  auto zeroIndexEmpty = extractor_utils::defOpAs<EmptyOp>(zeroIndexGeneric, 0);
  if (!zeroIndexEmpty)
    return mlir::failure();

  scaffold.zeroIndexConstant = expectedZeroConst;
  scaffold.indexExtentConstant = expectedExtentConst;
  scaffold.zeroIndexEmpty = zeroIndexEmpty.getOperation();
  scaffold.zeroIndexGeneric = zeroIndexGeneric;
  scaffold.zeroIndexExpand = zeroIndexExpand.getOperation();
  scaffold.baseOffsetEmpty = baseOffsetEmpty.getOperation();
  scaffold.baseOffsetGeneric = baseOffsetGeneric;
  scaffold.rangeEmpty = rangeEmpty.getOperation();
  scaffold.rangeGeneric = rangeGeneric;
  scaffold.rangeExpand = rangeExpand.getOperation();
  scaffold.combinedIndicesEmpty = combinedIndicesEmpty.getOperation();
  scaffold.combinedIndicesGeneric = combinedIndices;
  scaffold.collapsedIndices = collapsedIndices.getOperation();
  return mlir::success();
}

// Matches one activated gate slice and ties it back to the shared preactivation.
static mlir::LogicalResult
matchGateSlice(mlir::Operation *activationOp,
               mlir::Operation *sharedHiddenOutputEmpty, int64_t expectedOffset,
               bool expectSigmoid, LSTMGateSliceMatch &gate,
               LSTMIndexingScaffoldMatch &scaffold,
               mlir::Operation *&sharedCollapsedPreattivation,
               mlir::Operation *&sharedSigmoidOneConstant) {
  if (expectSigmoid) {
    if (!extractor_utils::isSigmoidGeneric(activationOp))
      return mlir::failure();

    mlir::Operation *sigmoidOneConstant = getSigmoidUnitConstant(activationOp);
    if (!sigmoidOneConstant)
      return mlir::failure();

    if (sharedSigmoidOneConstant && sharedSigmoidOneConstant != sigmoidOneConstant)
      return mlir::failure();
    sharedSigmoidOneConstant = sigmoidOneConstant;
  } else if (!extractor_utils::isTanhGeneric(activationOp)) {
    return mlir::failure();
  }

  if (!extractor_utils::hasOperands(activationOp, 2) ||
      !extractor_utils::hasInputs(activationOp, 1))
    return mlir::failure();

  auto activationOutputEmpty = extractor_utils::defOpAs<EmptyOp>(activationOp, 1);
  if (!activationOutputEmpty ||
      activationOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  auto expand = extractor_utils::defOpAs<ExpandShapeOp>(activationOp, 0);
  if (!expand)
    return mlir::failure();

  mlir::Operation *gather = extractor_utils::defOp(expand.getSrc());
  mlir::Operation *indexValueOp = nullptr;
  mlir::Operation *collapsedVectorOp = nullptr;
  mlir::Operation *zeroConst = nullptr;
  mlir::Operation *extentConst = nullptr;
  mlir::Operation *gatherEmpty = nullptr;
  if (!gather ||
      mlir::failed(matchCollapsedExtractGeneric(gather, indexValueOp,
                                                collapsedVectorOp, zeroConst,
                                                extentConst, gatherEmpty)))
    return mlir::failure();

  if (sharedCollapsedPreattivation &&
      sharedCollapsedPreattivation != collapsedVectorOp)
    return mlir::failure();
  sharedCollapsedPreattivation = collapsedVectorOp;

  if (scaffold.gatherEmpty && scaffold.gatherEmpty != gatherEmpty)
    return mlir::failure();
  scaffold.gatherEmpty = gatherEmpty;

  if (expectedOffset == 0) {
    auto collapsedIndices = llvm::dyn_cast<CollapseShapeOp>(indexValueOp);
    if (!collapsedIndices ||
        mlir::failed(matchSharedGateIndexScaffold(
            collapsedIndices, zeroConst, extentConst, scaffold)))
      return mlir::failure();
  } else {
    mlir::Operation *baseIndicesOp = extractor_utils::defOp(indexValueOp, 0);
    auto collapsedIndices = llvm::dyn_cast_or_null<CollapseShapeOp>(baseIndicesOp);
    if (!collapsedIndices ||
        mlir::failed(matchSharedGateIndexScaffold(
            collapsedIndices, zeroConst, extentConst, scaffold)))
      return mlir::failure();

    mlir::Operation *offsetConstant = nullptr;
    if (mlir::failed(matchOffsetAddGeneric(indexValueOp, scaffold.collapsedIndices,
                                           scaffold.rangeEmpty, expectedOffset,
                                           offsetConstant)))
      return mlir::failure();

    gate.offsetAdd = indexValueOp;
    gate.offsetConstant = offsetConstant;
  }

  gate.gather = gather;
  gate.expand = expand.getOperation();
  gate.activation = activationOp;
  return mlir::success();
}

// Validates the shared transposed-weight matmul core used by one LSTM branch.
static mlir::LogicalResult
matchLSTMMatmulBranchCore(MatmulOp matmul, mlir::Operation *sharedOutputEmpty,
                          LSTMCellMatmulBranchMatch &branch) {
  if (!extractor_utils::hasOperands(matmul.getOperation(), 3) ||
      !extractor_utils::hasInputs(matmul.getOperation(), 2))
    return mlir::failure();

  auto outputInit = extractor_impl::matchFillOutputInit(matmul.getOperation(), 2);
  if (!outputInit || outputInit->outputEmpty != sharedOutputEmpty)
    return mlir::failure();

  auto firstTranspose =
      extractor_utils::defOpAs<TransposeOp>(matmul.getOperation(), 0);
  auto secondTranspose =
      extractor_utils::defOpAs<TransposeOp>(matmul.getOperation(), 1);
  if (static_cast<bool>(firstTranspose) == static_cast<bool>(secondTranspose))
    return mlir::failure();

  auto transpose = firstTranspose ? firstTranspose : secondTranspose;
  auto transposeEmpty =
      extractor_utils::defOpAs<EmptyOp>(transpose.getOperation(), 1);
  auto weightConstant =
      extractor_utils::defOpAs<ConstantOp>(transpose.getOperation(), 0);
  if (!transposeEmpty || !weightConstant)
    return mlir::failure();

  branch.activationInput =
      firstTranspose ? matmul.getOperand(1) : matmul.getOperand(0);
  branch.weightConstant = weightConstant.getOperation();
  branch.transposeEmpty = transposeEmpty.getOperation();
  branch.transpose = transpose.getOperation();
  branch.matmul = matmul.getOperation();
  branch.fill = outputInit->outputFill;
  branch.fillConstant = outputInit->outputFillConstant;
  return mlir::success();
}

// Recognizes a branch where the matmul result is biased before preactivation fusion.
static mlir::LogicalResult
matchLSTMMatmulBranchWithBias(mlir::Operation *branchAdd,
                              mlir::Operation *sharedOutputEmpty,
                              LSTMCellMatmulBranchMatch &branch) {
  if (!extractor_utils::isAddfGeneric(branchAdd) ||
      !extractor_utils::hasOperands(branchAdd, 3) ||
      !extractor_utils::hasInputs(branchAdd, 2))
    return mlir::failure();

  auto branchOutputEmpty = extractor_utils::defOpAs<EmptyOp>(branchAdd, 2);
  if (!branchOutputEmpty ||
      branchOutputEmpty.getOperation() != sharedOutputEmpty ||
      !extractor_utils::inputsAreEither<MatmulOp, ConstantOp>(branchAdd))
    return mlir::failure();

  auto matmul = extractor_utils::defOpAs<MatmulOp>(branchAdd, 0);
  auto biasConstant = extractor_utils::defOpAs<ConstantOp>(branchAdd, 1);
  if (!matmul || !biasConstant) {
    matmul = extractor_utils::defOpAs<MatmulOp>(branchAdd, 1);
    biasConstant = extractor_utils::defOpAs<ConstantOp>(branchAdd, 0);
  }

  if (!matmul || !biasConstant ||
      mlir::failed(
          matchLSTMMatmulBranchCore(matmul, sharedOutputEmpty, branch)))
    return mlir::failure();

  branch.biasAdd = branchAdd;
  branch.biasConstant = biasConstant.getOperation();
  return mlir::success();
}

// Recognizes a bias-free branch represented directly by a matmul.
static mlir::LogicalResult
matchLSTMMatmulBranchWithoutBias(mlir::Operation *branchMatmul,
                                 mlir::Operation *sharedOutputEmpty,
                                 LSTMCellMatmulBranchMatch &branch) {
  auto matmul = llvm::dyn_cast<MatmulOp>(branchMatmul);
  if (!matmul)
    return mlir::failure();

  return matchLSTMMatmulBranchCore(matmul, sharedOutputEmpty, branch);
}

// Assigns the two preactivation branches to x and h based on consumed arguments.
static mlir::LogicalResult
assignLSTMBranches(FuncOp func, const LSTMCellMatmulBranchMatch &firstBranch,
                   const LSTMCellMatmulBranchMatch &secondBranch,
                   LSTMCellMatmulBranchMatch &inputBranch,
                   LSTMCellMatmulBranchMatch &hiddenBranch) {
  if (func.getNumArguments() < 3)
    return mlir::failure();

  mlir::Value inputArg = func.getArgument(0);
  mlir::Value hiddenArg = func.getArgument(1);

  if (firstBranch.activationInput == inputArg &&
      secondBranch.activationInput == hiddenArg) {
    inputBranch = firstBranch;
    hiddenBranch = secondBranch;
    return mlir::success();
  }

  if (firstBranch.activationInput == hiddenArg &&
      secondBranch.activationInput == inputArg) {
    inputBranch = secondBranch;
    hiddenBranch = firstBranch;
    return mlir::success();
  }

  return mlir::failure();
}

// Matches the final hidden update mul and identifies the output gate plus tanh(c_next).
static mlir::LogicalResult
matchFinalHiddenMul(mlir::Operation *op, mlir::Operation *&outputGate,
                    mlir::Operation *&cellTanh,
                    mlir::Operation *&sharedHiddenOutputEmpty) {
  if (!extractor_utils::isMulfGeneric(op) ||
      !extractor_utils::hasOperands(op, 3) ||
      !extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 2);
  if (!outputEmpty)
    return mlir::failure();

  mlir::Operation *firstInput = extractor_utils::defOp(op, 0);
  mlir::Operation *secondInput = extractor_utils::defOp(op, 1);
  if (extractor_utils::isSigmoidGeneric(firstInput) &&
      extractor_utils::isTanhGeneric(secondInput)) {
    outputGate = firstInput;
    cellTanh = secondInput;
  } else if (extractor_utils::isSigmoidGeneric(secondInput) &&
             extractor_utils::isTanhGeneric(firstInput)) {
    outputGate = secondInput;
    cellTanh = firstInput;
  } else {
    return mlir::failure();
  }

  sharedHiddenOutputEmpty = outputEmpty.getOperation();
  return mlir::success();
}

// Matches forget_gate * c_prev and recovers the forget gate activation.
static mlir::LogicalResult
matchForgetCellMul(mlir::Operation *op, mlir::Value cellStateArg,
                   mlir::Operation *sharedHiddenOutputEmpty,
                   mlir::Operation *&forgetGate) {
  if (!extractor_utils::isMulfGeneric(op) ||
      !extractor_utils::hasOperands(op, 3) ||
      !extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 2);
  if (!outputEmpty || outputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  mlir::Operation *firstInput = extractor_utils::defOp(op, 0);
  mlir::Operation *secondInput = extractor_utils::defOp(op, 1);
  if (extractor_utils::isSigmoidGeneric(firstInput) &&
      op->getOperand(1) == cellStateArg) {
    forgetGate = firstInput;
    return mlir::success();
  }

  if (extractor_utils::isSigmoidGeneric(secondInput) &&
      op->getOperand(0) == cellStateArg) {
    forgetGate = secondInput;
    return mlir::success();
  }

  return mlir::failure();
}

// Matches input_gate * candidate_gate and identifies which activation is which.
static mlir::LogicalResult
matchInputCandidateMul(mlir::Operation *op, mlir::Operation *sharedHiddenOutputEmpty,
                       mlir::Operation *&inputGate,
                       mlir::Operation *&candidateGate) {
  if (!extractor_utils::isMulfGeneric(op) ||
      !extractor_utils::hasOperands(op, 3) ||
      !extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 2);
  if (!outputEmpty || outputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  mlir::Operation *firstInput = extractor_utils::defOp(op, 0);
  mlir::Operation *secondInput = extractor_utils::defOp(op, 1);
  if (extractor_utils::isSigmoidGeneric(firstInput) &&
      extractor_utils::isTanhGeneric(secondInput)) {
    inputGate = firstInput;
    candidateGate = secondInput;
    return mlir::success();
  }

  if (extractor_utils::isSigmoidGeneric(secondInput) &&
      extractor_utils::isTanhGeneric(firstInput)) {
    inputGate = secondInput;
    candidateGate = firstInput;
    return mlir::success();
  }

  return mlir::failure();
}

// Collects the matched LSTM cell ops in cloning order.
static void collectMatchedOps(LSTMCellMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.inputBranch.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.inputBranch.transposeEmpty);
  match_utils::appendUniqueOp(match.ops, match.inputBranch.transpose);
  match_utils::appendUniqueOp(match.ops, match.hiddenBranch.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.hiddenBranch.transposeEmpty);
  match_utils::appendUniqueOp(match.ops, match.hiddenBranch.transpose);
  match_utils::appendUniqueOp(match.ops, match.sharedMatmulOutputEmpty);
  match_utils::appendUniqueOp(match.ops, match.sharedFillConstant);
  match_utils::appendUniqueOp(match.ops, match.sharedFillOp);
  match_utils::appendUniqueOp(match.ops, match.inputBranch.matmul);
  match_utils::appendUniqueOp(match.ops, match.inputBranch.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.inputBranch.biasAdd);
  match_utils::appendUniqueOp(match.ops, match.hiddenBranch.matmul);
  match_utils::appendUniqueOp(match.ops, match.hiddenBranch.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.hiddenBranch.biasAdd);
  match_utils::appendUniqueOp(match.ops, match.preActivationAddOp);
  match_utils::appendUniqueOp(match.ops, match.preActivationCollapseOp);
  match_utils::appendUniqueOp(match.ops, match.indexing.zeroIndexConstant);
  match_utils::appendUniqueOp(match.ops, match.indexing.indexExtentConstant);
  match_utils::appendUniqueOp(match.ops, match.indexing.zeroIndexEmpty);
  match_utils::appendUniqueOp(match.ops, match.indexing.zeroIndexGeneric);
  match_utils::appendUniqueOp(match.ops, match.indexing.zeroIndexExpand);
  match_utils::appendUniqueOp(match.ops, match.indexing.baseOffsetEmpty);
  match_utils::appendUniqueOp(match.ops, match.indexing.baseOffsetGeneric);
  match_utils::appendUniqueOp(match.ops, match.indexing.rangeEmpty);
  match_utils::appendUniqueOp(match.ops, match.indexing.rangeGeneric);
  match_utils::appendUniqueOp(match.ops, match.indexing.rangeExpand);
  match_utils::appendUniqueOp(match.ops, match.indexing.combinedIndicesEmpty);
  match_utils::appendUniqueOp(match.ops, match.indexing.combinedIndicesGeneric);
  match_utils::appendUniqueOp(match.ops, match.indexing.collapsedIndices);
  match_utils::appendUniqueOp(match.ops, match.indexing.gatherEmpty);
  match_utils::appendUniqueOp(match.ops, match.sharedHiddenOutputEmpty);
  match_utils::appendUniqueOp(match.ops, match.sigmoidOneConstant);
  match_utils::appendUniqueOp(match.ops, match.inputGate.offsetConstant);
  match_utils::appendUniqueOp(match.ops, match.inputGate.offsetAdd);
  match_utils::appendUniqueOp(match.ops, match.inputGate.gather);
  match_utils::appendUniqueOp(match.ops, match.inputGate.expand);
  match_utils::appendUniqueOp(match.ops, match.inputGate.activation);
  match_utils::appendUniqueOp(match.ops, match.forgetGate.offsetConstant);
  match_utils::appendUniqueOp(match.ops, match.forgetGate.offsetAdd);
  match_utils::appendUniqueOp(match.ops, match.forgetGate.gather);
  match_utils::appendUniqueOp(match.ops, match.forgetGate.expand);
  match_utils::appendUniqueOp(match.ops, match.forgetGate.activation);
  match_utils::appendUniqueOp(match.ops, match.candidateGate.offsetConstant);
  match_utils::appendUniqueOp(match.ops, match.candidateGate.offsetAdd);
  match_utils::appendUniqueOp(match.ops, match.candidateGate.gather);
  match_utils::appendUniqueOp(match.ops, match.candidateGate.expand);
  match_utils::appendUniqueOp(match.ops, match.candidateGate.activation);
  match_utils::appendUniqueOp(match.ops, match.outputGate.offsetConstant);
  match_utils::appendUniqueOp(match.ops, match.outputGate.offsetAdd);
  match_utils::appendUniqueOp(match.ops, match.outputGate.gather);
  match_utils::appendUniqueOp(match.ops, match.outputGate.expand);
  match_utils::appendUniqueOp(match.ops, match.outputGate.activation);
  match_utils::appendUniqueOp(match.ops, match.forgetCellMulOp);
  match_utils::appendUniqueOp(match.ops, match.inputCandidateMulOp);
  match_utils::appendUniqueOp(match.ops, match.cellAddOp);
  match_utils::appendUniqueOp(match.ops, match.cellTanhOp);
  match_utils::appendUniqueOp(match.ops, match.hiddenMulOp);
}

// Computes the full outline boundary once the LSTM structure is proven.
static void finalizeLSTMCellMatch(LSTMCellMatch &match) {
  collectMatchedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Shared whole-cell matcher used by the biased and bias-free entry points.
static std::optional<LSTMCellMatch> matchLSTMCell(mlir::Operation *op,
                                                  bool expectBias) {
  FuncOp func = op ? op->getParentOfType<FuncOp>() : FuncOp();
  if (!func || func.getNumArguments() < 3)
    return std::nullopt;

  mlir::Operation *outputGateActivation = nullptr;
  mlir::Operation *cellTanh = nullptr;
  mlir::Operation *sharedHiddenOutputEmpty = nullptr;
  if (mlir::failed(matchFinalHiddenMul(op, outputGateActivation, cellTanh,
                                       sharedHiddenOutputEmpty)))
    return std::nullopt;

  if (!extractor_utils::hasOperands(cellTanh, 2) ||
      !extractor_utils::hasInputs(cellTanh, 1) ||
      !extractor_utils::isTanhGeneric(cellTanh))
    return std::nullopt;

  auto cellTanhOutputEmpty = extractor_utils::defOpAs<EmptyOp>(cellTanh, 1);
  if (!cellTanhOutputEmpty ||
      cellTanhOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return std::nullopt;

  mlir::Operation *cellAdd = extractor_utils::defOp(cellTanh, 0);
  if (!extractor_utils::isAddfGeneric(cellAdd) ||
      !extractor_utils::hasOperands(cellAdd, 3) ||
      !extractor_utils::hasInputs(cellAdd, 2))
    return std::nullopt;

  auto cellAddOutputEmpty = extractor_utils::defOpAs<EmptyOp>(cellAdd, 2);
  if (!cellAddOutputEmpty ||
      cellAddOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return std::nullopt;

  mlir::Operation *forgetCellMul = nullptr;
  mlir::Operation *inputCandidateMul = nullptr;
  mlir::Operation *firstCellAddInput = extractor_utils::defOp(cellAdd, 0);
  mlir::Operation *secondCellAddInput = extractor_utils::defOp(cellAdd, 1);
  mlir::Operation *forgetGateActivation = nullptr;
  mlir::Operation *inputGateActivation = nullptr;
  mlir::Operation *candidateGateActivation = nullptr;
  if (mlir::succeeded(matchForgetCellMul(firstCellAddInput, func.getArgument(2),
                                         sharedHiddenOutputEmpty,
                                         forgetGateActivation)) &&
      mlir::succeeded(matchInputCandidateMul(secondCellAddInput,
                                             sharedHiddenOutputEmpty,
                                             inputGateActivation,
                                             candidateGateActivation))) {
    forgetCellMul = firstCellAddInput;
    inputCandidateMul = secondCellAddInput;
  } else if (mlir::succeeded(matchForgetCellMul(secondCellAddInput,
                                                func.getArgument(2),
                                                sharedHiddenOutputEmpty,
                                                forgetGateActivation)) &&
             mlir::succeeded(matchInputCandidateMul(firstCellAddInput,
                                                    sharedHiddenOutputEmpty,
                                                    inputGateActivation,
                                                    candidateGateActivation))) {
    forgetCellMul = secondCellAddInput;
    inputCandidateMul = firstCellAddInput;
  } else {
    return std::nullopt;
  }

  LSTMIndexingScaffoldMatch indexing;
  mlir::Operation *collapsedPreattivation = nullptr;
  mlir::Operation *sigmoidOneConstant = nullptr;
  LSTMGateSliceMatch outputGate;
  if (mlir::failed(matchGateSlice(outputGateActivation, sharedHiddenOutputEmpty,
                                  9, true, outputGate, indexing,
                                  collapsedPreattivation,
                                  sigmoidOneConstant)))
    return std::nullopt;

  LSTMGateSliceMatch forgetGate;
  if (mlir::failed(matchGateSlice(forgetGateActivation, sharedHiddenOutputEmpty,
                                  3, true, forgetGate, indexing,
                                  collapsedPreattivation,
                                  sigmoidOneConstant)))
    return std::nullopt;

  LSTMGateSliceMatch inputGate;
  if (mlir::failed(matchGateSlice(inputGateActivation, sharedHiddenOutputEmpty,
                                  0, true, inputGate, indexing,
                                  collapsedPreattivation,
                                  sigmoidOneConstant)))
    return std::nullopt;

  LSTMGateSliceMatch candidateGate;
  if (mlir::failed(matchGateSlice(candidateGateActivation,
                                  sharedHiddenOutputEmpty, 6, false,
                                  candidateGate, indexing,
                                  collapsedPreattivation,
                                  sigmoidOneConstant)))
    return std::nullopt;

  auto collapsedPreattivationOp =
      llvm::dyn_cast_or_null<CollapseShapeOp>(collapsedPreattivation);
  if (!collapsedPreattivationOp)
    return std::nullopt;

  mlir::Operation *preActivationAdd =
      extractor_utils::defOp(collapsedPreattivationOp.getSrc());
  if (!extractor_utils::isAddfGeneric(preActivationAdd) ||
      !extractor_utils::hasOperands(preActivationAdd, 3) ||
      !extractor_utils::hasInputs(preActivationAdd, 2))
    return std::nullopt;

  auto matmulOutputEmpty = extractor_utils::defOpAs<EmptyOp>(preActivationAdd, 2);
  if (!matmulOutputEmpty)
    return std::nullopt;

  LSTMCellMatmulBranchMatch firstBranch;
  LSTMCellMatmulBranchMatch secondBranch;
  if (expectBias) {
    if (mlir::failed(matchLSTMMatmulBranchWithBias(extractor_utils::defOp(preActivationAdd, 0),
                                                   matmulOutputEmpty.getOperation(),
                                                   firstBranch)) ||
        mlir::failed(matchLSTMMatmulBranchWithBias(extractor_utils::defOp(preActivationAdd, 1),
                                                   matmulOutputEmpty.getOperation(),
                                                   secondBranch)))
      return std::nullopt;
  } else {
    if (mlir::failed(matchLSTMMatmulBranchWithoutBias(extractor_utils::defOp(preActivationAdd, 0),
                                                      matmulOutputEmpty.getOperation(),
                                                      firstBranch)) ||
        mlir::failed(matchLSTMMatmulBranchWithoutBias(extractor_utils::defOp(preActivationAdd, 1),
                                                      matmulOutputEmpty.getOperation(),
                                                      secondBranch)))
      return std::nullopt;
  }

  if (firstBranch.fill != secondBranch.fill)
    return std::nullopt;

  LSTMCellMatmulBranchMatch inputBranch;
  LSTMCellMatmulBranchMatch hiddenBranch;
  if (mlir::failed(assignLSTMBranches(func, firstBranch, secondBranch,
                                      inputBranch, hiddenBranch)))
    return std::nullopt;

  LSTMCellMatch match;
  match.sharedMatmulOutputEmpty = matmulOutputEmpty.getOperation();
  match.sharedFillOp = inputBranch.fill;
  match.sharedFillConstant = inputBranch.fillConstant;
  match.preActivationAddOp = preActivationAdd;
  match.preActivationCollapseOp = collapsedPreattivationOp.getOperation();
  match.inputBranch = inputBranch;
  match.hiddenBranch = hiddenBranch;
  match.indexing = indexing;
  match.inputGate = inputGate;
  match.forgetGate = forgetGate;
  match.candidateGate = candidateGate;
  match.outputGate = outputGate;
  match.sharedHiddenOutputEmpty = sharedHiddenOutputEmpty;
  match.sigmoidOneConstant = sigmoidOneConstant;
  match.forgetCellMulOp = forgetCellMul;
  match.inputCandidateMulOp = inputCandidateMul;
  match.cellAddOp = cellAdd;
  match.cellTanhOp = cellTanh;
  match.hiddenMulOp = op;
  match.root = op;
  finalizeLSTMCellMatch(match);
  return match;
}

static std::optional<LSTMCellMatch> matchLSTMCellWithBias(mlir::Operation *op) {
  return matchLSTMCell(op, /*expectBias=*/true);
}

static std::optional<LSTMCellMatch>
matchLSTMCellWithoutBias(mlir::Operation *op) {
  return matchLSTMCell(op, /*expectBias=*/false);
}

// Outlines the matched LSTM cell into a layer function and replaces the root.
static void rewriteLSTMCellExtractor(const LSTMCellMatch &match,
                                     mlir::RewriterBase &rewriter) {
  mlir::StringRef layerType =
      match.inputBranch.biasAdd ? mlir::StringRef("lstm_cell_w_bias")
                                : mlir::StringRef("lstm_cell");
  rewrite_utils::extractToFunction(match.root, match.ops, match.inputs,
                                   match.outputs, rewriter, layerType);
}

// Finds linalg-based LSTM cell bodies and outlines each match.
class LSTMCellExtractor : public mlir::analog::LayerExtractor {
public:
  explicit LSTMCellExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "lstm_cell"; }

  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    extractor_impl::extractAllMatches(func, rewriter, matchLSTMCellWithBias,
                                      rewriteLSTMCellExtractor);
    extractor_impl::extractAllMatches(func, rewriter, matchLSTMCellWithoutBias,
                                      rewriteLSTMCellExtractor);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Adds the LSTM cell extractor to the layer extraction pipeline.
void registerLSTMCellExtractor(LayerExtractors &extractors,
                               MLIRContext *context) {
  extractors.push_back(std::make_unique<LSTMCellExtractor>(context));
}

} // namespace analog
} // namespace mlir
