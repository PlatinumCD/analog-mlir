#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"

#include <memory>
#include <optional>
#include <string>

namespace converter_utils = mlir::analog::converter_utils;
namespace extractor_utils = mlir::analog::extractor_utils;

namespace {

using mlir::arith::AddFOp;
using mlir::arith::AddIOp;
using mlir::arith::CmpIOp;
using mlir::arith::ConstantOp;
using mlir::arith::DivFOp;
using mlir::arith::IndexCastOp;
using mlir::arith::MulFOp;
using mlir::arith::MulIOp;
using mlir::arith::NegFOp;
using mlir::arith::SelectOp;
using mlir::func::ReturnOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::IndexOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::math::ExpOp;
using mlir::math::TanhOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::ExtractOp;

struct SupportedLSTMCellTypes {
  mlir::RankedTensorType inputStateTy;
  mlir::RankedTensorType hiddenStateTy;
  mlir::RankedTensorType cellStateTy;
  mlir::RankedTensorType outputTy;
};

struct SupportedLSTMCellBranch {
  mlir::Value activation;
  mlir::RankedTensorType activationTy;
  ConstantOp weightConstant;
  mlir::RankedTensorType weightConstantTy;
  EmptyOp transposeInit;
  TransposeOp transposeOp;
  MatmulOp matmulOp;
  FillOp sharedFillOp;
  mlir::Value sharedFillValue;
  GenericOp biasAddOp;
  ConstantOp biasConstant;
  mlir::Value bias;
  mlir::RankedTensorType biasTy;
  bool hasBias = false;
};

struct SupportedLSTMIndexingScaffold {
  ConstantOp zeroIndexConstant;
  ConstantOp indexExtentConstant;
  EmptyOp zeroIndexEmpty;
  GenericOp zeroIndexGeneric;
  ExpandShapeOp zeroIndexExpand;
  EmptyOp baseOffsetEmpty;
  GenericOp baseOffsetGeneric;
  EmptyOp rangeEmpty;
  GenericOp rangeGeneric;
  ExpandShapeOp rangeExpand;
  EmptyOp combinedIndicesEmpty;
  GenericOp combinedIndicesGeneric;
  CollapseShapeOp collapsedIndices;
  EmptyOp gatherEmpty;
};

struct SupportedLSTMGate {
  ConstantOp offsetConstant;
  GenericOp offsetAdd;
  GenericOp gather;
  ExpandShapeOp expand;
  GenericOp activation;
};

struct SupportedLSTMCellMatch {
  ReturnOp returnOp;
  EmptyOp sharedMatmulOutputEmpty;
  FillOp sharedFillOp;
  ConstantOp sharedFillConstant;
  GenericOp preActivationAddOp;
  CollapseShapeOp preActivationCollapseOp;
  SupportedLSTMCellBranch inputBranch;
  SupportedLSTMCellBranch hiddenBranch;
  SupportedLSTMIndexingScaffold indexing;
  SupportedLSTMGate inputGate;
  SupportedLSTMGate forgetGate;
  SupportedLSTMGate candidateGate;
  SupportedLSTMGate outputGate;
  EmptyOp sharedHiddenOutputEmpty;
  ConstantOp sigmoidOneConstant;
  GenericOp forgetCellMulOp;
  GenericOp inputCandidateMulOp;
  GenericOp cellAddOp;
  GenericOp cellTanhOp;
  GenericOp hiddenMulOp;
  SupportedLSTMCellTypes types;
  bool hasBias = false;
  llvm::SmallVector<mlir::Operation *> matchedOps;
  llvm::SmallVector<mlir::Operation *> tailOps;
};

struct LSTMCellLoweringState {
  mlir::Location loc;
  mlir::Type elementType;
  mlir::RankedTensorType fusedInputTy;
  mlir::RankedTensorType fusedWeightTy;
  mlir::RankedTensorType fusedBiasTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType outputTy;
  mlir::Value zeroValue;
};

struct PreparedFusedWeight {
  ConstantOp fusedWeightConstant;
  mlir::Value partitionedMatrix;
  mlir::scf::ForOp placementLoop;
  EmptyOp transposeInit;
  TransposeOp transposeOp;
  mlir::Value transposedWeight;
  int64_t matrixId;
};

struct PreparedFusedBias {
  ConstantOp fusedBiasConstant;
  mlir::Value bias;
};

struct PreparedFusedVector {
  mlir::Value fusedInput;
  mlir::Value partitionedVector;
  mlir::scf::ForOp placementLoop;
};

static void appendUniqueOp(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                           mlir::Operation *op) {
  if (!op)
    return;

  if (!llvm::is_contained(ops, op))
    ops.push_back(op);
}

static bool isZeroF32Constant(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return false;

  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(constant.getValue()))
    return floatAttr.getValue().isZero();

  if (auto denseAttr =
          llvm::dyn_cast<mlir::DenseElementsAttr>(constant.getValue())) {
    if (!denseAttr.isSplat())
      return false;

    auto splatValue = llvm::dyn_cast<mlir::FloatAttr>(
        denseAttr.getSplatValue<mlir::Attribute>());
    return splatValue && splatValue.getValue().isZero();
  }

  return false;
}

static bool isResourceBackedF32Constant(ConstantOp constant) {
  return constant &&
         llvm::isa<mlir::DenseF32ResourceElementsAttr>(constant.getValue());
}

static mlir::FailureOr<llvm::SmallVector<float>>
getF32ConstantValues(ConstantOp constant) {
  if (!constant)
    return mlir::failure();

  if (auto denseAttr =
          llvm::dyn_cast<mlir::DenseFPElementsAttr>(constant.getValue())) {
    llvm::SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (const llvm::APFloat &value : denseAttr.getValues<llvm::APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (auto denseResourceAttr =
          llvm::dyn_cast<mlir::DenseF32ResourceElementsAttr>(
              constant.getValue())) {
    std::optional<llvm::ArrayRef<float>> values =
        denseResourceAttr.tryGetAsArrayRef();
    if (!values)
      return mlir::failure();
    return llvm::SmallVector<float>(values->begin(), values->end());
  }

  return mlir::failure();
}

static mlir::Operation *getLaterOp(mlir::Operation *lhs,
                                   mlir::Operation *rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return lhs->isBeforeInBlock(rhs) ? rhs : lhs;
}

static mlir::FailureOr<mlir::RankedTensorType>
getSupportedRank2TensorType(mlir::Value value) {
  auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorTy || !tensorTy.hasStaticShape() || tensorTy.getRank() != 2 ||
      !tensorTy.getElementType().isF32())
    return mlir::failure();

  if (tensorTy.getShape()[0] != 1 || tensorTy.getShape()[1] <= 0)
    return mlir::failure();

  return tensorTy;
}

static bool constantOpHasI64Value(mlir::Operation *op, int64_t expected) {
  auto constant = llvm::dyn_cast_or_null<ConstantOp>(op);
  if (!constant)
    return false;

  auto attr = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return attr && attr.getInt() == expected;
}

static mlir::Operation *getSigmoidUnitConstant(mlir::Operation *op) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
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
  auto neg = llvm::dyn_cast<NegFOp>(&*it++);
  auto exp = (it != e) ? llvm::dyn_cast<ExpOp>(&*it++) : ExpOp();
  auto add = (it != e) ? llvm::dyn_cast<AddFOp>(&*it++) : AddFOp();
  auto div = (it != e) ? llvm::dyn_cast<DivFOp>(&*it++) : DivFOp();
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

static mlir::FailureOr<SupportedLSTMCellTypes>
getSupportedLSTMCellTypes(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 3 || functionType.getNumResults() != 1)
    return mlir::failure();

  auto inputStateTy = getSupportedRank2TensorType(func.getArgument(0));
  auto hiddenStateTy = getSupportedRank2TensorType(func.getArgument(1));
  auto cellStateTy = getSupportedRank2TensorType(func.getArgument(2));
  auto outputTy =
      llvm::dyn_cast<mlir::RankedTensorType>(functionType.getResult(0));
  if (mlir::failed(inputStateTy) || mlir::failed(hiddenStateTy) ||
      mlir::failed(cellStateTy) || !outputTy || !outputTy.hasStaticShape() ||
      outputTy.getRank() != 2 || !outputTy.getElementType().isF32())
    return mlir::failure();

  if (outputTy.getShape()[0] != 1 || outputTy.getShape()[1] <= 0)
    return mlir::failure();

  if (hiddenStateTy->getShape()[1] != outputTy.getShape()[1] ||
      cellStateTy->getShape()[1] != outputTy.getShape()[1])
    return mlir::failure();

  return SupportedLSTMCellTypes{*inputStateTy, *hiddenStateTy, *cellStateTy,
                                outputTy};
}

static mlir::LogicalResult
matchYieldingConstantGeneric(mlir::Operation *op,
                             mlir::Operation *expectedConstant) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
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

static mlir::LogicalResult matchIndexRangeGeneric(mlir::Operation *op) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
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
  auto index = llvm::dyn_cast<IndexOp>(&*it++);
  auto cast = (it != e) ? llvm::dyn_cast<IndexCastOp>(&*it++) : IndexCastOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!index || !cast || !yield || it != e || cast.getIn() != index.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != cast.getResult())
    return mlir::failure();

  return mlir::success();
}

static mlir::LogicalResult
matchMultiplyByConstantGeneric(mlir::Operation *op,
                               mlir::Operation *expectedConstant) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
  if (!generic)
    return mlir::failure();

  if (!extractor_utils::hasOperands(op, 2) || !extractor_utils::hasInputs(op, 1))
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  mlir::Block &block = region.front();
  auto mul = llvm::dyn_cast<MulIOp>(&block.front());
  if (!mul)
    return mlir::failure();

  mlir::Operation *lhsDef = mul.getLhs().getDefiningOp();
  mlir::Operation *rhsDef = mul.getRhs().getDefiningOp();
  if (lhsDef != expectedConstant && rhsDef != expectedConstant)
    return mlir::failure();

  return mlir::success();
}

static mlir::LogicalResult
matchOffsetAddGeneric(mlir::Operation *op, mlir::Operation *expectedBaseIndices,
                      mlir::Operation *expectedOutputEmpty, int64_t expectedOffset,
                      ConstantOp &offsetConstant) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
  if (!generic)
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
  auto add = llvm::dyn_cast<AddIOp>(&block.front());
  if (!add)
    return mlir::failure();

  auto lhsConst = add.getLhs().getDefiningOp<ConstantOp>();
  auto rhsConst = add.getRhs().getDefiningOp<ConstantOp>();
  auto offset = lhsConst ? lhsConst : rhsConst;
  if (!offset || !constantOpHasI64Value(offset.getOperation(), expectedOffset))
    return mlir::failure();

  offsetConstant = offset;
  return mlir::success();
}

static mlir::LogicalResult
matchCollapsedExtractGeneric(mlir::Operation *op, mlir::Operation *&indexValueOp,
                             CollapseShapeOp &collapsedVectorOp,
                             ConstantOp &zeroConst, ConstantOp &extentConst,
                             EmptyOp &gatherEmpty) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
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
  auto cmp = llvm::dyn_cast<CmpIOp>(&*it++);
  auto add = (it != e) ? llvm::dyn_cast<AddIOp>(&*it++) : AddIOp();
  auto select = (it != e) ? llvm::dyn_cast<SelectOp>(&*it++) : SelectOp();
  auto cast = (it != e) ? llvm::dyn_cast<IndexCastOp>(&*it++) : IndexCastOp();
  auto extract = (it != e) ? llvm::dyn_cast<ExtractOp>(&*it++) : ExtractOp();
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

  auto zero = zeroValue.getDefiningOp<ConstantOp>();
  auto extent = extentValue.getDefiningOp<ConstantOp>();
  auto collapsedVector = llvm::dyn_cast_or_null<CollapseShapeOp>(extract.getTensor().getDefiningOp());
  indexValueOp = extractor_utils::defOp(op, 0);
  if (!zero || !extent || !collapsedVector || !indexValueOp)
    return mlir::failure();

  zeroConst = zero;
  extentConst = extent;
  collapsedVectorOp = collapsedVector;
  gatherEmpty = outputEmpty;
  return mlir::success();
}

static mlir::LogicalResult
matchSharedGateIndexScaffold(CollapseShapeOp collapsedIndices,
                             ConstantOp expectedZeroConst,
                             ConstantOp expectedExtentConst,
                             SupportedLSTMIndexingScaffold &scaffold) {
  if (scaffold.collapsedIndices) {
    return scaffold.collapsedIndices == collapsedIndices &&
                   scaffold.zeroIndexConstant == expectedZeroConst &&
                   scaffold.indexExtentConstant == expectedExtentConst
               ? mlir::success()
               : mlir::failure();
  }

  auto combinedIndices =
      extractor_utils::defOpAs<GenericOp>(collapsedIndices.getSrc());
  if (!combinedIndices)
    return mlir::failure();

  if (!extractor_utils::hasOperands(combinedIndices.getOperation(), 3) ||
      !extractor_utils::hasInputs(combinedIndices.getOperation(), 2))
    return mlir::failure();

  auto combinedIndicesEmpty =
      extractor_utils::defOpAs<EmptyOp>(combinedIndices.getOperation(), 2);
  if (!combinedIndicesEmpty)
    return mlir::failure();

  auto baseOffsetGeneric =
      extractor_utils::defOpAs<GenericOp>(combinedIndices.getOperation(), 0);
  auto rangeExpand =
      extractor_utils::defOpAs<ExpandShapeOp>(combinedIndices.getOperation(), 1);
  if (!baseOffsetGeneric || !rangeExpand)
    return mlir::failure();

  auto rangeGeneric = extractor_utils::defOpAs<GenericOp>(rangeExpand.getSrc());
  if (!rangeGeneric ||
      mlir::failed(matchIndexRangeGeneric(rangeGeneric.getOperation())))
    return mlir::failure();

  auto rangeEmpty = extractor_utils::defOpAs<EmptyOp>(rangeGeneric.getOperation(), 0);
  if (!rangeEmpty)
    return mlir::failure();

  if (mlir::failed(matchMultiplyByConstantGeneric(baseOffsetGeneric.getOperation(),
                                                  expectedExtentConst.getOperation())))
    return mlir::failure();

  auto baseOffsetEmpty =
      extractor_utils::defOpAs<EmptyOp>(baseOffsetGeneric.getOperation(), 1);
  auto zeroIndexExpand =
      extractor_utils::defOpAs<ExpandShapeOp>(baseOffsetGeneric.getOperation(), 0);
  if (!baseOffsetEmpty || !zeroIndexExpand)
    return mlir::failure();

  auto zeroIndexGeneric =
      extractor_utils::defOpAs<GenericOp>(zeroIndexExpand.getSrc());
  if (!zeroIndexGeneric ||
      mlir::failed(matchYieldingConstantGeneric(zeroIndexGeneric.getOperation(),
                                                expectedZeroConst.getOperation())))
    return mlir::failure();

  auto zeroIndexEmpty =
      extractor_utils::defOpAs<EmptyOp>(zeroIndexGeneric.getOperation(), 0);
  if (!zeroIndexEmpty)
    return mlir::failure();

  scaffold.zeroIndexConstant = expectedZeroConst;
  scaffold.indexExtentConstant = expectedExtentConst;
  scaffold.zeroIndexEmpty = zeroIndexEmpty;
  scaffold.zeroIndexGeneric = zeroIndexGeneric;
  scaffold.zeroIndexExpand = zeroIndexExpand;
  scaffold.baseOffsetEmpty = baseOffsetEmpty;
  scaffold.baseOffsetGeneric = baseOffsetGeneric;
  scaffold.rangeEmpty = rangeEmpty;
  scaffold.rangeGeneric = rangeGeneric;
  scaffold.rangeExpand = rangeExpand;
  scaffold.combinedIndicesEmpty = combinedIndicesEmpty;
  scaffold.combinedIndicesGeneric = combinedIndices;
  scaffold.collapsedIndices = collapsedIndices;
  return mlir::success();
}

static mlir::LogicalResult
matchGateSlice(mlir::Operation *activationOp, EmptyOp sharedHiddenOutputEmpty,
               int64_t expectedOffset, int64_t expectedExtent,
               mlir::RankedTensorType expectedGateTy, bool expectSigmoid,
               SupportedLSTMGate &gate, SupportedLSTMIndexingScaffold &scaffold,
               CollapseShapeOp &sharedCollapsedPreattivation,
               ConstantOp &sharedSigmoidOneConstant) {
  if (expectSigmoid) {
    if (!extractor_utils::isSigmoidGeneric(activationOp))
      return mlir::failure();

    auto sigmoidOneConstant =
        llvm::dyn_cast_or_null<ConstantOp>(getSigmoidUnitConstant(activationOp));
    if (!sigmoidOneConstant)
      return mlir::failure();

    if (sharedSigmoidOneConstant &&
        sharedSigmoidOneConstant != sigmoidOneConstant)
      return mlir::failure();
    sharedSigmoidOneConstant = sigmoidOneConstant;
  } else if (!extractor_utils::isTanhGeneric(activationOp)) {
    return mlir::failure();
  }

  auto activationGeneric = llvm::dyn_cast_or_null<GenericOp>(activationOp);
  if (!activationGeneric ||
      !extractor_utils::hasOperands(activationOp, 2) ||
      !extractor_utils::hasInputs(activationOp, 1))
    return mlir::failure();

  auto activationResultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(activationGeneric.getResult(0).getType());
  auto activationOutputEmpty =
      extractor_utils::defOpAs<EmptyOp>(activationOp, 1);
  if (!activationResultTy || activationResultTy != expectedGateTy ||
      !activationOutputEmpty || activationOutputEmpty != sharedHiddenOutputEmpty)
    return mlir::failure();

  auto expand = extractor_utils::defOpAs<ExpandShapeOp>(activationOp, 0);
  if (!expand)
    return mlir::failure();

  auto expandedTy =
      llvm::dyn_cast<mlir::RankedTensorType>(expand.getResult().getType());
  if (!expandedTy || expandedTy != expectedGateTy)
    return mlir::failure();

  auto gather = extractor_utils::defOpAs<GenericOp>(expand.getSrc());
  mlir::Operation *indexValueOp = nullptr;
  CollapseShapeOp collapsedVectorOp;
  ConstantOp zeroConst;
  ConstantOp extentConst;
  EmptyOp gatherEmpty;
  if (!gather ||
      mlir::failed(matchCollapsedExtractGeneric(gather.getOperation(), indexValueOp,
                                                collapsedVectorOp, zeroConst,
                                                extentConst, gatherEmpty)))
    return mlir::failure();

  auto gatherResultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(gather.getResult(0).getType());
  if (!gatherResultTy || gatherResultTy.getRank() != 1 ||
      gatherResultTy.getShape()[0] != expectedGateTy.getShape()[1])
    return mlir::failure();

  if (!constantOpHasI64Value(zeroConst.getOperation(), 0) ||
      !constantOpHasI64Value(extentConst.getOperation(), expectedExtent))
    return mlir::failure();

  if (sharedCollapsedPreattivation &&
      sharedCollapsedPreattivation != collapsedVectorOp)
    return mlir::failure();
  sharedCollapsedPreattivation = collapsedVectorOp;

  if (scaffold.gatherEmpty && scaffold.gatherEmpty != gatherEmpty)
    return mlir::failure();
  scaffold.gatherEmpty = gatherEmpty;

  if (expectedOffset == 0) {
    auto collapsedIndices = llvm::dyn_cast_or_null<CollapseShapeOp>(indexValueOp);
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

    ConstantOp offsetConstant;
    if (mlir::failed(matchOffsetAddGeneric(indexValueOp,
                                           scaffold.collapsedIndices.getOperation(),
                                           scaffold.rangeEmpty.getOperation(),
                                           expectedOffset, offsetConstant)))
      return mlir::failure();

    gate.offsetAdd = llvm::cast<GenericOp>(indexValueOp);
    gate.offsetConstant = offsetConstant;
  }

  gate.gather = gather;
  gate.expand = expand;
  gate.activation = activationGeneric;
  return mlir::success();
}

static mlir::FailureOr<SupportedLSTMCellBranch>
matchSupportedLSTMBranchCore(MatmulOp matmul, EmptyOp sharedOutputEmpty,
                             mlir::RankedTensorType preActivationTy) {
  SupportedLSTMCellBranch branch;
  if (!matmul || matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1)
    return mlir::failure();

  auto sharedFillOp = matmul.getOutputs()[0].getDefiningOp<FillOp>();
  auto matmulResultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(matmul.getResult(0).getType());
  if (!sharedFillOp || !matmulResultTy || matmulResultTy != preActivationTy ||
      sharedFillOp.getResult(0) != matmul.getOutputs()[0] ||
      sharedFillOp.getInputs().size() != 1 ||
      !isZeroF32Constant(sharedFillOp.getInputs()[0]) ||
      sharedFillOp.getOutputs()[0].getDefiningOp<EmptyOp>() != sharedOutputEmpty)
    return mlir::failure();

  auto firstTranspose = matmul.getInputs()[0].getDefiningOp<TransposeOp>();
  auto secondTranspose = matmul.getInputs()[1].getDefiningOp<TransposeOp>();
  if (static_cast<bool>(firstTranspose) == static_cast<bool>(secondTranspose))
    return mlir::failure();

  auto transposeOp = firstTranspose ? firstTranspose : secondTranspose;
  mlir::Value activation =
      firstTranspose ? matmul.getInputs()[1] : matmul.getInputs()[0];
  auto activationTy = getSupportedRank2TensorType(activation);
  if (mlir::failed(activationTy))
    return mlir::failure();

  auto weightConstant = transposeOp.getInput().getDefiningOp<ConstantOp>();
  auto transposeInit = transposeOp.getInit().getDefiningOp<EmptyOp>();
  auto weightConstantTy =
      weightConstant
          ? llvm::dyn_cast<mlir::RankedTensorType>(weightConstant.getType())
          : mlir::RankedTensorType();
  auto transposedWeightTy = llvm::dyn_cast<mlir::RankedTensorType>(
      transposeOp.getResult().front().getType());
  if (!weightConstant || !weightConstantTy || !transposedWeightTy ||
      !weightConstantTy.hasStaticShape() || !transposedWeightTy.hasStaticShape() ||
      weightConstantTy.getRank() != 2 || transposedWeightTy.getRank() != 2 ||
      !weightConstantTy.getElementType().isF32() ||
      !transposedWeightTy.getElementType().isF32() || !transposeInit)
    return mlir::failure();

  if (weightConstantTy.getShape()[0] != preActivationTy.getShape()[1] ||
      weightConstantTy.getShape()[1] != activationTy->getShape()[1])
    return mlir::failure();

  if (transposedWeightTy.getShape()[0] != activationTy->getShape()[1] ||
      transposedWeightTy.getShape()[1] != preActivationTy.getShape()[1])
    return mlir::failure();

  branch.activation = activation;
  branch.activationTy = *activationTy;
  branch.weightConstant = weightConstant;
  branch.weightConstantTy = weightConstantTy;
  branch.transposeInit = transposeInit;
  branch.transposeOp = transposeOp;
  branch.matmulOp = matmul;
  branch.sharedFillOp = sharedFillOp;
  branch.sharedFillValue = sharedFillOp.getInputs()[0];
  return branch;
}

static mlir::FailureOr<SupportedLSTMCellBranch>
matchSupportedLSTMBranchWithBias(mlir::Operation *branchAdd,
                                 EmptyOp sharedOutputEmpty,
                                 mlir::RankedTensorType preActivationTy) {
  auto addOp = llvm::dyn_cast_or_null<GenericOp>(branchAdd);
  if (!addOp || !extractor_utils::isAddfGeneric(branchAdd) ||
      !extractor_utils::hasOperands(branchAdd, 3) ||
      !extractor_utils::hasInputs(branchAdd, 2))
    return mlir::failure();

  auto branchOutputEmpty = extractor_utils::defOpAs<EmptyOp>(branchAdd, 2);
  if (!branchOutputEmpty || branchOutputEmpty != sharedOutputEmpty ||
      !extractor_utils::operandDefiningOpsMatchEither<MatmulOp, ConstantOp>(
          branchAdd))
    return mlir::failure();

  auto matmul = extractor_utils::defOpAs<MatmulOp>(branchAdd, 0);
  auto biasConstant = extractor_utils::defOpAs<ConstantOp>(branchAdd, 1);
  if (!matmul || !biasConstant) {
    matmul = extractor_utils::defOpAs<MatmulOp>(branchAdd, 1);
    biasConstant = extractor_utils::defOpAs<ConstantOp>(branchAdd, 0);
  }

  if (!matmul || !biasConstant)
    return mlir::failure();

  auto branch = matchSupportedLSTMBranchCore(matmul, sharedOutputEmpty,
                                             preActivationTy);
  if (mlir::failed(branch))
    return mlir::failure();

  auto biasTy =
      llvm::dyn_cast<mlir::RankedTensorType>(biasConstant.getType());
  if (!biasTy || !biasTy.hasStaticShape() || biasTy.getRank() != 1 ||
      !biasTy.getElementType().isF32() ||
      biasTy.getShape()[0] != preActivationTy.getShape()[1])
    return mlir::failure();

  branch->hasBias = true;
  branch->biasAddOp = addOp;
  branch->biasConstant = biasConstant;
  branch->bias = biasConstant.getResult();
  branch->biasTy = biasTy;
  return branch;
}

static mlir::FailureOr<SupportedLSTMCellBranch>
matchSupportedLSTMBranchWithoutBias(mlir::Operation *branchMatmul,
                                    EmptyOp sharedOutputEmpty,
                                    mlir::RankedTensorType preActivationTy) {
  auto matmul = llvm::dyn_cast_or_null<MatmulOp>(branchMatmul);
  if (!matmul)
    return mlir::failure();

  return matchSupportedLSTMBranchCore(matmul, sharedOutputEmpty,
                                      preActivationTy);
}

static mlir::LogicalResult
assignLSTMBranches(mlir::func::FuncOp func,
                   const SupportedLSTMCellBranch &firstBranch,
                   const SupportedLSTMCellBranch &secondBranch,
                   SupportedLSTMCellBranch &inputBranch,
                   SupportedLSTMCellBranch &hiddenBranch) {
  if (func.getNumArguments() < 3)
    return mlir::failure();

  if (firstBranch.activation == func.getArgument(0) &&
      secondBranch.activation == func.getArgument(1)) {
    inputBranch = firstBranch;
    hiddenBranch = secondBranch;
    return mlir::success();
  }

  if (firstBranch.activation == func.getArgument(1) &&
      secondBranch.activation == func.getArgument(0)) {
    inputBranch = secondBranch;
    hiddenBranch = firstBranch;
    return mlir::success();
  }

  return mlir::failure();
}

static mlir::LogicalResult
matchFinalHiddenMul(mlir::Operation *op, mlir::RankedTensorType outputTy,
                    GenericOp &outputGate, GenericOp &cellTanh,
                    EmptyOp &sharedHiddenOutputEmpty) {
  if (!extractor_utils::isMulfGeneric(op) ||
      !extractor_utils::hasOperands(op, 3) ||
      !extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  auto mulOp = llvm::dyn_cast_or_null<GenericOp>(op);
  auto resultTy =
      mulOp ? llvm::dyn_cast<mlir::RankedTensorType>(mulOp.getResult(0).getType())
            : mlir::RankedTensorType();
  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 2);
  if (!mulOp || !resultTy || resultTy != outputTy || !outputEmpty)
    return mlir::failure();

  auto firstInput = extractor_utils::defOpAs<GenericOp>(op, 0);
  auto secondInput = extractor_utils::defOpAs<GenericOp>(op, 1);
  if (firstInput && secondInput && extractor_utils::isSigmoidGeneric(firstInput) &&
      extractor_utils::isTanhGeneric(secondInput)) {
    outputGate = firstInput;
    cellTanh = secondInput;
  } else if (firstInput && secondInput &&
             extractor_utils::isSigmoidGeneric(secondInput) &&
             extractor_utils::isTanhGeneric(firstInput)) {
    outputGate = secondInput;
    cellTanh = firstInput;
  } else {
    return mlir::failure();
  }

  sharedHiddenOutputEmpty = outputEmpty;
  return mlir::success();
}

static mlir::LogicalResult
matchForgetCellMul(mlir::Operation *op, mlir::Value cellStateArg,
                   EmptyOp sharedHiddenOutputEmpty, GenericOp &forgetGate) {
  if (!extractor_utils::isMulfGeneric(op) ||
      !extractor_utils::hasOperands(op, 3) ||
      !extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  auto mulOp = llvm::dyn_cast_or_null<GenericOp>(op);
  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 2);
  if (!mulOp || !outputEmpty || outputEmpty != sharedHiddenOutputEmpty)
    return mlir::failure();

  auto firstInput = extractor_utils::defOpAs<GenericOp>(op, 0);
  auto secondInput = extractor_utils::defOpAs<GenericOp>(op, 1);
  if (firstInput && extractor_utils::isSigmoidGeneric(firstInput) &&
      op->getOperand(1) == cellStateArg) {
    forgetGate = firstInput;
    return mlir::success();
  }

  if (secondInput && extractor_utils::isSigmoidGeneric(secondInput) &&
      op->getOperand(0) == cellStateArg) {
    forgetGate = secondInput;
    return mlir::success();
  }

  return mlir::failure();
}

static mlir::LogicalResult
matchInputCandidateMul(mlir::Operation *op, EmptyOp sharedHiddenOutputEmpty,
                       GenericOp &inputGate, GenericOp &candidateGate) {
  if (!extractor_utils::isMulfGeneric(op) ||
      !extractor_utils::hasOperands(op, 3) ||
      !extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  auto mulOp = llvm::dyn_cast_or_null<GenericOp>(op);
  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 2);
  if (!mulOp || !outputEmpty || outputEmpty != sharedHiddenOutputEmpty)
    return mlir::failure();

  auto firstInput = extractor_utils::defOpAs<GenericOp>(op, 0);
  auto secondInput = extractor_utils::defOpAs<GenericOp>(op, 1);
  if (firstInput && secondInput && extractor_utils::isSigmoidGeneric(firstInput) &&
      extractor_utils::isTanhGeneric(secondInput)) {
    inputGate = firstInput;
    candidateGate = secondInput;
    return mlir::success();
  }

  if (firstInput && secondInput &&
      extractor_utils::isSigmoidGeneric(secondInput) &&
      extractor_utils::isTanhGeneric(firstInput)) {
    inputGate = secondInput;
    candidateGate = firstInput;
    return mlir::success();
  }

  return mlir::failure();
}

static void collectMatchedOps(SupportedLSTMCellMatch &match) {
  match.matchedOps.clear();

  appendUniqueOp(match.matchedOps, match.inputBranch.weightConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.inputBranch.transposeInit.getOperation());
  appendUniqueOp(match.matchedOps, match.inputBranch.transposeOp.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenBranch.weightConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenBranch.transposeInit.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenBranch.transposeOp.getOperation());
  appendUniqueOp(match.matchedOps, match.sharedMatmulOutputEmpty.getOperation());
  appendUniqueOp(match.matchedOps, match.sharedFillConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.sharedFillOp.getOperation());
  appendUniqueOp(match.matchedOps, match.inputBranch.matmulOp.getOperation());
  appendUniqueOp(match.matchedOps, match.inputBranch.biasConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.inputBranch.biasAddOp.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenBranch.matmulOp.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenBranch.biasConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenBranch.biasAddOp.getOperation());
  appendUniqueOp(match.matchedOps, match.preActivationAddOp.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.preActivationCollapseOp.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.zeroIndexConstant.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.indexExtentConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.indexing.zeroIndexEmpty.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.zeroIndexGeneric.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.zeroIndexExpand.getOperation());
  appendUniqueOp(match.matchedOps, match.indexing.baseOffsetEmpty.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.baseOffsetGeneric.getOperation());
  appendUniqueOp(match.matchedOps, match.indexing.rangeEmpty.getOperation());
  appendUniqueOp(match.matchedOps, match.indexing.rangeGeneric.getOperation());
  appendUniqueOp(match.matchedOps, match.indexing.rangeExpand.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.combinedIndicesEmpty.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.combinedIndicesGeneric.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.indexing.collapsedIndices.getOperation());
  appendUniqueOp(match.matchedOps, match.indexing.gatherEmpty.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.sharedHiddenOutputEmpty.getOperation());
  appendUniqueOp(match.matchedOps, match.sigmoidOneConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.inputGate.offsetConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.inputGate.offsetAdd.getOperation());
  appendUniqueOp(match.matchedOps, match.inputGate.gather.getOperation());
  appendUniqueOp(match.matchedOps, match.inputGate.expand.getOperation());
  appendUniqueOp(match.matchedOps, match.inputGate.activation.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.forgetGate.offsetConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.forgetGate.offsetAdd.getOperation());
  appendUniqueOp(match.matchedOps, match.forgetGate.gather.getOperation());
  appendUniqueOp(match.matchedOps, match.forgetGate.expand.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.forgetGate.activation.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.candidateGate.offsetConstant.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.candidateGate.offsetAdd.getOperation());
  appendUniqueOp(match.matchedOps, match.candidateGate.gather.getOperation());
  appendUniqueOp(match.matchedOps, match.candidateGate.expand.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.candidateGate.activation.getOperation());
  appendUniqueOp(match.matchedOps, match.outputGate.offsetConstant.getOperation());
  appendUniqueOp(match.matchedOps, match.outputGate.offsetAdd.getOperation());
  appendUniqueOp(match.matchedOps, match.outputGate.gather.getOperation());
  appendUniqueOp(match.matchedOps, match.outputGate.expand.getOperation());
  appendUniqueOp(match.matchedOps, match.outputGate.activation.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.forgetCellMulOp.getOperation());
  appendUniqueOp(match.matchedOps,
                 match.inputCandidateMulOp.getOperation());
  appendUniqueOp(match.matchedOps, match.cellAddOp.getOperation());
  appendUniqueOp(match.matchedOps, match.cellTanhOp.getOperation());
  appendUniqueOp(match.matchedOps, match.hiddenMulOp.getOperation());
}

static void collectTailOps(SupportedLSTMCellMatch &match) {
  match.tailOps.clear();

  appendUniqueOp(match.tailOps, match.preActivationCollapseOp.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.zeroIndexConstant.getOperation());
  appendUniqueOp(match.tailOps,
                 match.indexing.indexExtentConstant.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.zeroIndexEmpty.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.zeroIndexGeneric.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.zeroIndexExpand.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.baseOffsetEmpty.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.baseOffsetGeneric.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.rangeEmpty.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.rangeGeneric.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.rangeExpand.getOperation());
  appendUniqueOp(match.tailOps,
                 match.indexing.combinedIndicesEmpty.getOperation());
  appendUniqueOp(match.tailOps,
                 match.indexing.combinedIndicesGeneric.getOperation());
  appendUniqueOp(match.tailOps,
                 match.indexing.collapsedIndices.getOperation());
  appendUniqueOp(match.tailOps, match.indexing.gatherEmpty.getOperation());
  appendUniqueOp(match.tailOps, match.sharedHiddenOutputEmpty.getOperation());
  appendUniqueOp(match.tailOps, match.sigmoidOneConstant.getOperation());
  appendUniqueOp(match.tailOps, match.inputGate.offsetConstant.getOperation());
  appendUniqueOp(match.tailOps, match.inputGate.offsetAdd.getOperation());
  appendUniqueOp(match.tailOps, match.inputGate.gather.getOperation());
  appendUniqueOp(match.tailOps, match.inputGate.expand.getOperation());
  appendUniqueOp(match.tailOps, match.inputGate.activation.getOperation());
  appendUniqueOp(match.tailOps, match.forgetGate.offsetConstant.getOperation());
  appendUniqueOp(match.tailOps, match.forgetGate.offsetAdd.getOperation());
  appendUniqueOp(match.tailOps, match.forgetGate.gather.getOperation());
  appendUniqueOp(match.tailOps, match.forgetGate.expand.getOperation());
  appendUniqueOp(match.tailOps, match.forgetGate.activation.getOperation());
  appendUniqueOp(match.tailOps,
                 match.candidateGate.offsetConstant.getOperation());
  appendUniqueOp(match.tailOps, match.candidateGate.offsetAdd.getOperation());
  appendUniqueOp(match.tailOps, match.candidateGate.gather.getOperation());
  appendUniqueOp(match.tailOps, match.candidateGate.expand.getOperation());
  appendUniqueOp(match.tailOps,
                 match.candidateGate.activation.getOperation());
  appendUniqueOp(match.tailOps, match.outputGate.offsetConstant.getOperation());
  appendUniqueOp(match.tailOps, match.outputGate.offsetAdd.getOperation());
  appendUniqueOp(match.tailOps, match.outputGate.gather.getOperation());
  appendUniqueOp(match.tailOps, match.outputGate.expand.getOperation());
  appendUniqueOp(match.tailOps, match.outputGate.activation.getOperation());
  appendUniqueOp(match.tailOps, match.forgetCellMulOp.getOperation());
  appendUniqueOp(match.tailOps, match.inputCandidateMulOp.getOperation());
  appendUniqueOp(match.tailOps, match.cellAddOp.getOperation());
  appendUniqueOp(match.tailOps, match.cellTanhOp.getOperation());
  appendUniqueOp(match.tailOps, match.hiddenMulOp.getOperation());
}

static mlir::FailureOr<SupportedLSTMCellMatch>
matchSupportedLSTMCell(mlir::func::FuncOp func) {
  auto types = getSupportedLSTMCellTypes(func);
  if (mlir::failed(types) || !func.getBody().hasOneBlock())
    return mlir::failure();

  auto returnOp = llvm::dyn_cast<ReturnOp>(func.front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return mlir::failure();

  auto hiddenMulOp = returnOp.getOperand(0).getDefiningOp<GenericOp>();
  if (!hiddenMulOp)
    return mlir::failure();

  GenericOp outputGateActivation;
  GenericOp cellTanhOp;
  EmptyOp sharedHiddenOutputEmpty;
  if (mlir::failed(matchFinalHiddenMul(hiddenMulOp.getOperation(), types->outputTy,
                                       outputGateActivation, cellTanhOp,
                                       sharedHiddenOutputEmpty)))
    return mlir::failure();

  if (!extractor_utils::hasOperands(cellTanhOp.getOperation(), 2) ||
      !extractor_utils::hasInputs(cellTanhOp.getOperation(), 1) ||
      !extractor_utils::isTanhGeneric(cellTanhOp.getOperation()))
    return mlir::failure();

  auto cellTanhOutputEmpty =
      extractor_utils::defOpAs<EmptyOp>(cellTanhOp.getOperation(), 1);
  if (!cellTanhOutputEmpty || cellTanhOutputEmpty != sharedHiddenOutputEmpty)
    return mlir::failure();

  auto cellAddOp = extractor_utils::defOpAs<GenericOp>(cellTanhOp.getOperation(), 0);
  if (!cellAddOp || !extractor_utils::isAddfGeneric(cellAddOp.getOperation()) ||
      !extractor_utils::hasOperands(cellAddOp.getOperation(), 3) ||
      !extractor_utils::hasInputs(cellAddOp.getOperation(), 2))
    return mlir::failure();

  auto cellAddOutputEmpty =
      extractor_utils::defOpAs<EmptyOp>(cellAddOp.getOperation(), 2);
  if (!cellAddOutputEmpty || cellAddOutputEmpty != sharedHiddenOutputEmpty)
    return mlir::failure();

  GenericOp forgetCellMulOp;
  GenericOp inputCandidateMulOp;
  GenericOp forgetGateActivation;
  GenericOp inputGateActivation;
  GenericOp candidateGateActivation;
  mlir::Operation *firstCellAddInput =
      extractor_utils::defOp(cellAddOp.getOperation(), 0);
  mlir::Operation *secondCellAddInput =
      extractor_utils::defOp(cellAddOp.getOperation(), 1);
  if (mlir::succeeded(matchForgetCellMul(firstCellAddInput, func.getArgument(2),
                                         sharedHiddenOutputEmpty,
                                         forgetGateActivation)) &&
      mlir::succeeded(matchInputCandidateMul(secondCellAddInput,
                                             sharedHiddenOutputEmpty,
                                             inputGateActivation,
                                             candidateGateActivation))) {
    forgetCellMulOp = llvm::cast<GenericOp>(firstCellAddInput);
    inputCandidateMulOp = llvm::cast<GenericOp>(secondCellAddInput);
  } else if (mlir::succeeded(matchForgetCellMul(secondCellAddInput,
                                                func.getArgument(2),
                                                sharedHiddenOutputEmpty,
                                                forgetGateActivation)) &&
             mlir::succeeded(matchInputCandidateMul(firstCellAddInput,
                                                    sharedHiddenOutputEmpty,
                                                    inputGateActivation,
                                                    candidateGateActivation))) {
    forgetCellMulOp = llvm::cast<GenericOp>(secondCellAddInput);
    inputCandidateMulOp = llvm::cast<GenericOp>(firstCellAddInput);
  } else {
    return mlir::failure();
  }

  int64_t hiddenSize = types->outputTy.getShape()[1];
  int64_t gateWidth = hiddenSize * 4;
  mlir::RankedTensorType gateTy = types->outputTy;

  SupportedLSTMIndexingScaffold indexing;
  CollapseShapeOp collapsedPreattivation;
  ConstantOp sigmoidOneConstant;
  SupportedLSTMGate outputGate;
  if (mlir::failed(matchGateSlice(outputGateActivation.getOperation(),
                                  sharedHiddenOutputEmpty, 3 * hiddenSize,
                                  gateWidth, gateTy, /*expectSigmoid=*/true,
                                  outputGate, indexing, collapsedPreattivation,
                                  sigmoidOneConstant)))
    return mlir::failure();

  SupportedLSTMGate forgetGate;
  if (mlir::failed(matchGateSlice(forgetGateActivation.getOperation(),
                                  sharedHiddenOutputEmpty, hiddenSize,
                                  gateWidth, gateTy, /*expectSigmoid=*/true,
                                  forgetGate, indexing, collapsedPreattivation,
                                  sigmoidOneConstant)))
    return mlir::failure();

  SupportedLSTMGate inputGate;
  if (mlir::failed(matchGateSlice(inputGateActivation.getOperation(),
                                  sharedHiddenOutputEmpty, 0, gateWidth, gateTy,
                                  /*expectSigmoid=*/true, inputGate, indexing,
                                  collapsedPreattivation,
                                  sigmoidOneConstant)))
    return mlir::failure();

  SupportedLSTMGate candidateGate;
  if (mlir::failed(matchGateSlice(candidateGateActivation.getOperation(),
                                  sharedHiddenOutputEmpty, 2 * hiddenSize,
                                  gateWidth, gateTy, /*expectSigmoid=*/false,
                                  candidateGate, indexing, collapsedPreattivation,
                                  sigmoidOneConstant)))
    return mlir::failure();

  auto collapsedTy = llvm::dyn_cast<mlir::RankedTensorType>(
      collapsedPreattivation.getResult().getType());
  if (!collapsedTy || !collapsedTy.hasStaticShape() || collapsedTy.getRank() != 1 ||
      collapsedTy.getShape()[0] != gateWidth)
    return mlir::failure();

  auto preActivationAddOp =
      extractor_utils::defOpAs<GenericOp>(collapsedPreattivation.getSrc());
  if (!preActivationAddOp ||
      !extractor_utils::isAddfGeneric(preActivationAddOp.getOperation()) ||
      !extractor_utils::hasOperands(preActivationAddOp.getOperation(), 3) ||
      !extractor_utils::hasInputs(preActivationAddOp.getOperation(), 2))
    return mlir::failure();

  auto preActivationTy = llvm::dyn_cast<mlir::RankedTensorType>(
      preActivationAddOp.getResult(0).getType());
  if (!preActivationTy || !preActivationTy.hasStaticShape() ||
      preActivationTy.getRank() != 2 || !preActivationTy.getElementType().isF32() ||
      preActivationTy.getShape()[0] != 1 || preActivationTy.getShape()[1] != gateWidth)
    return mlir::failure();

  auto matmulOutputEmpty =
      extractor_utils::defOpAs<EmptyOp>(preActivationAddOp.getOperation(), 2);
  if (!matmulOutputEmpty)
    return mlir::failure();

  bool expectBias = converter_utils::hasLayerType(func, "lstm_cell_w_bias");
  bool expectNoBias = converter_utils::hasLayerType(func, "lstm_cell");
  if (expectBias == expectNoBias)
    return mlir::failure();

  auto firstBranchValue = extractor_utils::defOp(preActivationAddOp.getOperation(), 0);
  auto secondBranchValue = extractor_utils::defOp(preActivationAddOp.getOperation(), 1);

  auto firstBranch = expectBias
                         ? matchSupportedLSTMBranchWithBias(firstBranchValue,
                                                            matmulOutputEmpty,
                                                            preActivationTy)
                         : matchSupportedLSTMBranchWithoutBias(firstBranchValue,
                                                               matmulOutputEmpty,
                                                               preActivationTy);
  auto secondBranch = expectBias
                          ? matchSupportedLSTMBranchWithBias(secondBranchValue,
                                                             matmulOutputEmpty,
                                                             preActivationTy)
                          : matchSupportedLSTMBranchWithoutBias(secondBranchValue,
                                                                matmulOutputEmpty,
                                                                preActivationTy);
  if (mlir::failed(firstBranch) || mlir::failed(secondBranch))
    return mlir::failure();

  if (firstBranch->sharedFillOp != secondBranch->sharedFillOp)
    return mlir::failure();

  auto sharedFillConstant =
      firstBranch->sharedFillValue.getDefiningOp<ConstantOp>();
  if (!sharedFillConstant)
    return mlir::failure();

  if (firstBranch->hasBias != secondBranch->hasBias)
    return mlir::failure();

  if (expectBias && !firstBranch->hasBias)
    return mlir::failure();

  if (expectNoBias && firstBranch->hasBias)
    return mlir::failure();

  SupportedLSTMCellBranch inputBranch;
  SupportedLSTMCellBranch hiddenBranch;
  if (mlir::failed(assignLSTMBranches(func, *firstBranch, *secondBranch,
                                      inputBranch, hiddenBranch)))
    return mlir::failure();

  if (inputBranch.weightConstantTy.getShape()[0] != gateWidth ||
      inputBranch.weightConstantTy.getShape()[1] !=
          types->inputStateTy.getShape()[1] ||
      hiddenBranch.weightConstantTy.getShape()[0] != gateWidth ||
      hiddenBranch.weightConstantTy.getShape()[1] !=
          types->hiddenStateTy.getShape()[1])
    return mlir::failure();

  SupportedLSTMCellMatch match{returnOp,
                               matmulOutputEmpty,
                               inputBranch.sharedFillOp,
                               sharedFillConstant,
                               preActivationAddOp,
                               collapsedPreattivation,
                               inputBranch,
                               hiddenBranch,
                               indexing,
                               inputGate,
                               forgetGate,
                               candidateGate,
                               outputGate,
                               sharedHiddenOutputEmpty,
                               sigmoidOneConstant,
                               forgetCellMulOp,
                               inputCandidateMulOp,
                               cellAddOp,
                               cellTanhOp,
                               hiddenMulOp,
                               *types,
                               inputBranch.hasBias};
  collectMatchedOps(match);
  collectTailOps(match);
  return match;
}

static mlir::RankedTensorType
buildFusedWeightType(SupportedLSTMCellMatch &match) {
  return mlir::RankedTensorType::get(
      {match.types.outputTy.getShape()[1] * 4,
       match.types.inputStateTy.getShape()[1] +
           match.types.hiddenStateTy.getShape()[1]},
      match.types.outputTy.getElementType());
}

static mlir::RankedTensorType
buildFusedInputType(SupportedLSTMCellMatch &match) {
  return mlir::RankedTensorType::get(
      {1, match.types.inputStateTy.getShape()[1] +
              match.types.hiddenStateTy.getShape()[1]},
      match.types.outputTy.getElementType());
}

static mlir::RankedTensorType
buildFusedBiasType(SupportedLSTMCellMatch &match) {
  return mlir::RankedTensorType::get(
      {match.types.outputTy.getShape()[1] * 4},
      match.types.outputTy.getElementType());
}

static LSTMCellLoweringState
buildLSTMCellLoweringState(mlir::OpBuilder &builder,
                           SupportedLSTMCellMatch &match) {
  mlir::Location loc = match.returnOp.getLoc();
  mlir::Type elementType = match.types.outputTy.getElementType();
  mlir::RankedTensorType fusedInputTy = buildFusedInputType(match);
  mlir::RankedTensorType fusedWeightTy = buildFusedWeightType(match);
  mlir::RankedTensorType fusedBiasTy = buildFusedBiasType(match);
  mlir::RankedTensorType matmulResultTy = mlir::RankedTensorType::get(
      {1, match.types.outputTy.getShape()[1] * 4}, elementType);

  return LSTMCellLoweringState{
      loc,
      elementType,
      fusedInputTy,
      fusedWeightTy,
      fusedBiasTy,
      matmulResultTy,
      match.types.outputTy,
      builder.create<mlir::arith::ConstantFloatOp>(
          loc, llvm::cast<mlir::FloatType>(elementType), llvm::APFloat(0.0f)),
  };
}

static mlir::TypedAttr buildFusedWeightAttr(SupportedLSTMCellMatch &match,
                                            mlir::RankedTensorType fusedTy) {
  auto maybeInputWeights = getF32ConstantValues(match.inputBranch.weightConstant);
  auto maybeHiddenWeights =
      getF32ConstantValues(match.hiddenBranch.weightConstant);
  if (mlir::failed(maybeInputWeights) || mlir::failed(maybeHiddenWeights))
    return {};

  llvm::SmallVector<float> inputWeights = *maybeInputWeights;
  llvm::SmallVector<float> hiddenWeights = *maybeHiddenWeights;
  int64_t outputSize = fusedTy.getShape()[0];
  int64_t inputWidth = match.inputBranch.weightConstantTy.getShape()[1];
  int64_t hiddenWidth = match.hiddenBranch.weightConstantTy.getShape()[1];
  int64_t fusedWidth = fusedTy.getShape()[1];
  llvm::SmallVector<float> fusedWeights(fusedTy.getNumElements(), 0.0f);

  for (int64_t row = 0; row < outputSize; ++row) {
    int64_t inputOffset = row * inputWidth;
    int64_t hiddenOffset = row * hiddenWidth;
    int64_t fusedOffset = row * fusedWidth;
    for (int64_t col = 0; col < inputWidth; ++col)
      fusedWeights[fusedOffset + col] = inputWeights[inputOffset + col];
    for (int64_t col = 0; col < hiddenWidth; ++col)
      fusedWeights[fusedOffset + inputWidth + col] =
          hiddenWeights[hiddenOffset + col];
  }

  bool useResource = isResourceBackedF32Constant(match.inputBranch.weightConstant) ||
                     isResourceBackedF32Constant(match.hiddenBranch.weightConstant);
  if (useResource) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        "analog_lstm_cell_fused_weight_" + std::to_string(nextResourceId++);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(fusedWeights), /*dataIsMutable=*/false);
    return llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
        fusedTy, resourceName, std::move(blob)));
  }

  return llvm::cast<mlir::TypedAttr>(
      mlir::DenseElementsAttr::get(fusedTy, llvm::ArrayRef<float>(fusedWeights)));
}

static mlir::TypedAttr buildFusedBiasAttr(SupportedLSTMCellMatch &match,
                                          mlir::RankedTensorType fusedTy) {
  auto maybeInputBias = getF32ConstantValues(match.inputBranch.biasConstant);
  auto maybeHiddenBias = getF32ConstantValues(match.hiddenBranch.biasConstant);
  if (mlir::failed(maybeInputBias) || mlir::failed(maybeHiddenBias))
    return {};

  llvm::SmallVector<float> inputBias = *maybeInputBias;
  llvm::SmallVector<float> hiddenBias = *maybeHiddenBias;
  llvm::SmallVector<float> fusedBias(fusedTy.getNumElements(), 0.0f);

  for (int64_t idx = 0; idx < fusedTy.getShape()[0]; ++idx)
    fusedBias[idx] = inputBias[idx] + hiddenBias[idx];

  bool useResource = isResourceBackedF32Constant(match.inputBranch.biasConstant) ||
                     isResourceBackedF32Constant(match.hiddenBranch.biasConstant);
  if (useResource) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        "analog_lstm_cell_fused_bias_" + std::to_string(nextResourceId++);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(fusedBias), /*dataIsMutable=*/false);
    return llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
        fusedTy, resourceName, std::move(blob)));
  }

  return llvm::cast<mlir::TypedAttr>(
      mlir::DenseElementsAttr::get(fusedTy, llvm::ArrayRef<float>(fusedBias)));
}

static mlir::FailureOr<ConstantOp>
createFusedWeightConstant(SupportedLSTMCellMatch &match,
                          mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType fusedTy = buildFusedWeightType(match);
  mlir::TypedAttr fusedAttr = buildFusedWeightAttr(match, fusedTy);
  if (!fusedAttr)
    return mlir::failure();

  mlir::Operation *insertAfter =
      getLaterOp(match.inputBranch.weightConstant.getOperation(),
                 match.hiddenBranch.weightConstant.getOperation());
  rewriter.setInsertionPointAfter(insertAfter);
  return rewriter.create<ConstantOp>(match.returnOp.getLoc(), fusedTy, fusedAttr);
}

static mlir::FailureOr<ConstantOp>
createFusedBiasConstant(SupportedLSTMCellMatch &match,
                        mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType fusedTy = buildFusedBiasType(match);
  mlir::TypedAttr fusedAttr = buildFusedBiasAttr(match, fusedTy);
  if (!fusedAttr)
    return mlir::failure();

  mlir::Operation *insertAfter =
      getLaterOp(match.inputBranch.biasConstant.getOperation(),
                 match.hiddenBranch.biasConstant.getOperation());
  rewriter.setInsertionPointAfter(insertAfter);
  return rewriter.create<ConstantOp>(match.returnOp.getLoc(), fusedTy, fusedAttr);
}

static mlir::Value buildZeroInitializedTensor(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::RankedTensorType tensorTy,
                                              mlir::Value zeroValue) {
  mlir::Value empty = builder.create<EmptyOp>(loc, tensorTy.getShape(),
                                              tensorTy.getElementType());
  return builder.create<FillOp>(loc, mlir::ValueRange{zeroValue},
                                mlir::ValueRange{empty})
      .getResult(0);
}

static mlir::FailureOr<PreparedFusedWeight>
prepareFusedWeight(SupportedLSTMCellMatch &match,
                   const LSTMCellLoweringState &state,
                   mlir::RewriterBase &rewriter, int64_t arrayRows,
                   int64_t arrayCols) {
  auto fusedWeightConstant = createFusedWeightConstant(match, rewriter);
  if (mlir::failed(fusedWeightConstant))
    return mlir::failure();

  auto analogMatrix =
      converter_utils::materializeAnalogMatrix(*fusedWeightConstant, rewriter);
  if (mlir::failed(analogMatrix))
    return mlir::failure();

  auto matrixId = converter_utils::getOrSetMatrixId(*analogMatrix, rewriter);
  if (mlir::failed(matrixId))
    return mlir::failure();

  auto partitionedMatrix = converter_utils::partitionAnalogMatrix(
      *analogMatrix, rewriter, arrayRows, arrayCols);
  if (mlir::failed(partitionedMatrix))
    return mlir::failure();

  auto placementLoop =
      converter_utils::placeAnalogMatrix(*partitionedMatrix, rewriter);
  if (mlir::failed(placementLoop))
    return mlir::failure();

  rewriter.setInsertionPointAfter(placementLoop->getOperation());
  auto transposeInit = rewriter.create<EmptyOp>(
      state.loc,
      llvm::ArrayRef<int64_t>{state.fusedInputTy.getShape()[1],
                              state.fusedWeightTy.getShape()[0]},
      state.elementType);
  auto transposeOp = rewriter.create<TransposeOp>(
      state.loc, fusedWeightConstant->getResult(), transposeInit,
      llvm::ArrayRef<int64_t>{1, 0});

  return PreparedFusedWeight{*fusedWeightConstant,
                             *partitionedMatrix,
                             *placementLoop,
                             transposeInit,
                             transposeOp,
                             transposeOp.getResult().front(),
                             *matrixId};
}

static mlir::FailureOr<PreparedFusedBias>
prepareFusedBias(SupportedLSTMCellMatch &match,
                 mlir::RewriterBase &rewriter) {
  PreparedFusedBias preparedBias;
  if (!match.hasBias)
    return preparedBias;

  auto fusedBiasConstant = createFusedBiasConstant(match, rewriter);
  if (mlir::failed(fusedBiasConstant))
    return mlir::failure();

  preparedBias.fusedBiasConstant = *fusedBiasConstant;
  preparedBias.bias = fusedBiasConstant->getResult();
  return preparedBias;
}

static mlir::FailureOr<PreparedFusedVector>
prepareFusedVector(SupportedLSTMCellMatch &match,
                   PreparedFusedWeight &preparedWeight,
                   const LSTMCellLoweringState &state,
                   mlir::RewriterBase &rewriter, int64_t arrayRows,
                   int64_t arrayCols) {
  rewriter.setInsertionPointAfter(preparedWeight.transposeOp.getOperation());
  auto concatOp = rewriter.create<ConcatOp>(
      state.loc, state.fusedInputTy, /*dim=*/1,
      mlir::ValueRange{match.inputBranch.activation, match.hiddenBranch.activation});

  auto analogVector = converter_utils::materializeAnalogVector(
      concatOp.getResult(), preparedWeight.matrixId, rewriter);
  if (mlir::failed(analogVector))
    return mlir::failure();

  auto partitionedVector = converter_utils::partitionAnalogVector(
      *analogVector, rewriter, arrayRows, arrayCols);
  if (mlir::failed(partitionedVector))
    return mlir::failure();

  auto placementLoop =
      converter_utils::placeAnalogVector(*partitionedVector, rewriter);
  if (mlir::failed(placementLoop))
    return mlir::failure();

  return PreparedFusedVector{concatOp.getResult(), *partitionedVector,
                             *placementLoop};
}

static MatmulOp buildFusedMatmulScaffold(
    mlir::OpBuilder &builder, const LSTMCellLoweringState &state,
    mlir::Value fusedInput, mlir::Value transposedWeight) {
  mlir::Value matmulInit = buildZeroInitializedTensor(
      builder, state.loc, state.matmulResultTy, state.zeroValue);
  return builder.create<MatmulOp>(state.loc, state.matmulResultTy,
                                  mlir::ValueRange{fusedInput, transposedWeight},
                                  mlir::ValueRange{matmulInit});
}

static mlir::Value applyOptionalFusedBias(
    mlir::OpBuilder &builder, const LSTMCellLoweringState &state,
    const PreparedFusedBias &preparedBias, mlir::Value preActivation) {
  if (!preparedBias.bias)
    return preActivation;

  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1}};
  mlir::Value expandedBias = builder.create<ExpandShapeOp>(
      state.loc, state.matmulResultTy, preparedBias.bias, reassociation);
  mlir::Value biasedInit = builder.create<EmptyOp>(
      state.loc, state.matmulResultTy.getShape(), state.elementType);
  return builder
      .create<mlir::linalg::AddOp>(
          state.loc, mlir::ValueRange{preActivation, expandedBias},
          mlir::ValueRange{biasedInit})
      .getResult(0);
}

static mlir::FailureOr<mlir::Value>
cloneTailFromFusedPreActivation(mlir::RewriterBase &rewriter,
                                SupportedLSTMCellMatch &match,
                                mlir::Value fusedPreActivation) {
  mlir::IRMapping mapping;
  mapping.map(match.preActivationAddOp.getResult(0), fusedPreActivation);

  mlir::Operation *clonedHiddenMul = nullptr;
  for (mlir::Operation *op : match.tailOps) {
    mlir::Operation *cloned = rewriter.clone(*op, mapping);
    if (op == match.hiddenMulOp.getOperation())
      clonedHiddenMul = cloned;
  }

  if (!clonedHiddenMul || clonedHiddenMul->getNumResults() != 1)
    return mlir::failure();

  return clonedHiddenMul->getResult(0);
}

static void eraseIfUnused(mlir::Operation *op, mlir::RewriterBase &rewriter) {
  if (op && op->use_empty())
    rewriter.eraseOp(op);
}

static void eraseUnusedFusedMatmulScaffold(MatmulOp matmulOp,
                                           mlir::RewriterBase &rewriter) {
  FillOp fillOp;
  EmptyOp emptyOp;
  ConstantOp zeroConstant;
  if (matmulOp && matmulOp->getNumOperands() >= 3) {
    fillOp = matmulOp->getOperand(2).getDefiningOp<FillOp>();
    if (fillOp && fillOp->getNumOperands() >= 1)
      zeroConstant = fillOp->getOperand(0).getDefiningOp<ConstantOp>();
    if (fillOp && fillOp->getNumOperands() >= 2)
      emptyOp = fillOp->getOperand(1).getDefiningOp<EmptyOp>();
  }

  eraseIfUnused(matmulOp.getOperation(), rewriter);
  eraseIfUnused(fillOp.getOperation(), rewriter);
  eraseIfUnused(emptyOp.getOperation(), rewriter);
  eraseIfUnused(zeroConstant.getOperation(), rewriter);
}

static void eraseUnusedPreparedFusedWeightOps(
    PreparedFusedWeight &preparedWeight, mlir::RewriterBase &rewriter) {
  eraseIfUnused(preparedWeight.transposeOp.getOperation(), rewriter);
  eraseIfUnused(preparedWeight.transposeInit.getOperation(), rewriter);
}

static void eraseUnusedOriginalLSTMCellOps(SupportedLSTMCellMatch &match,
                                           mlir::RewriterBase &rewriter) {
  for (auto it = match.matchedOps.rbegin(); it != match.matchedOps.rend(); ++it)
    eraseIfUnused(*it, rewriter);
}

class LSTMCellConverter : public mlir::analog::LayerConverter {
public:
  mlir::StringRef getName() const override { return "lstm_cell"; }

  void convert(mlir::func::FuncOp func, int64_t arrayRows,
               int64_t arrayCols) const override {
    if (arrayRows <= 0 || arrayCols <= 0)
      return;

    mlir::IRRewriter rewriter(func.getContext());
    auto match = matchSupportedLSTMCell(func);
    if (mlir::failed(match))
      return;

    rewriter.setInsertionPointAfter(
        getLaterOp(match->inputBranch.weightConstant.getOperation(),
                   match->hiddenBranch.weightConstant.getOperation()));
    LSTMCellLoweringState state = buildLSTMCellLoweringState(rewriter, *match);

    auto preparedWeight =
        prepareFusedWeight(*match, state, rewriter, arrayRows, arrayCols);
    if (mlir::failed(preparedWeight))
      return;

    auto preparedBias = prepareFusedBias(*match, rewriter);
    if (mlir::failed(preparedBias))
      return;

    auto preparedVector =
        prepareFusedVector(*match, *preparedWeight, state, rewriter, arrayRows,
                           arrayCols);
    if (mlir::failed(preparedVector))
      return;

    auto executionBuffer = converter_utils::insertArrayExecution(
        preparedWeight->partitionedMatrix, preparedVector->partitionedVector,
        preparedWeight->placementLoop, preparedVector->placementLoop, rewriter);
    if (mlir::failed(executionBuffer))
      return;

    rewriter.setInsertionPoint(match->returnOp);
    MatmulOp fusedMatmul = buildFusedMatmulScaffold(
        rewriter, state, preparedVector->fusedInput,
        preparedWeight->transposedWeight);
    mlir::Value fusedPreActivation =
        applyOptionalFusedBias(rewriter, state, *preparedBias,
                               fusedMatmul.getResult(0));
    auto clonedHiddenResult =
        cloneTailFromFusedPreActivation(rewriter, *match, fusedPreActivation);
    if (mlir::failed(clonedHiddenResult))
      return;

    auto reducedTensor = converter_utils::insertArrayReduction(
        *executionBuffer, preparedWeight->partitionedMatrix, fusedMatmul,
        rewriter);
    if (mlir::failed(reducedTensor))
      return;

    match->hiddenMulOp.getResult(0).replaceAllUsesWith(*clonedHiddenResult);
    eraseUnusedFusedMatmulScaffold(fusedMatmul, rewriter);
    eraseUnusedPreparedFusedWeightOps(*preparedWeight, rewriter);
    eraseUnusedOriginalLSTMCellOps(*match, rewriter);
    func->setAttr("layer_domain", rewriter.getStringAttr("analog"));
  }
};

} // namespace

namespace mlir {
namespace analog {

void registerLSTMCellConverter(LayerConverters &converters,
                               LayerConverterMap &converterMap,
                               MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<LSTMCellConverter>();
  const LayerConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["lstm_cell"] = converterPtr;
  converterMap["lstm_cell_w_bias"] = converterPtr;
}

} // namespace analog
} // namespace mlir
