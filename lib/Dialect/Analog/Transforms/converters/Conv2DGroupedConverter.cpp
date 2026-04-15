#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/LoopUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <optional>
#include <string>

namespace converter_utils = mlir::analog::converter_utils;
namespace loop_utils = mlir::analog::loop_utils;

namespace {

using mlir::arith::ConstantOp;
using mlir::linalg::BroadcastOp;
using mlir::linalg::Conv2DNgchwGfchwOp;
using mlir::linalg::FillOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

struct Conv2DGroupedShapeInfo {
  int64_t g;
  int64_t cTotal;
  int64_t cg;
  int64_t h;
  int64_t w;
  int64_t fTotal;
  int64_t fg;
  int64_t kh;
  int64_t kw;
  int64_t oh;
  int64_t ow;
};

struct SupportedGroupedActivation {
  ExpandShapeOp expandOp;
  mlir::Value sourceActivation;
  mlir::RankedTensorType sourceActivationTy;
  mlir::RankedTensorType groupedActivationTy;
};

struct SupportedGroupedFilter {
  ExpandShapeOp expandOp;
  ConstantOp filterRank4Const;
  mlir::RankedTensorType filterRank4Ty;
  mlir::RankedTensorType groupedFilterTy;
};

struct SupportedGroupedOutputInit {
  FillOp fillOp;
  BroadcastOp broadcastOp;
  ExpandShapeOp outputExpandOp;
  mlir::Value bias;
  mlir::RankedTensorType sourceOutputTy;
  mlir::RankedTensorType biasTy;
  mlir::RankedTensorType biasBroadcastTy;
  bool hasBias = false;
};

struct Conv2DGroupedMatch {
  Conv2DNgchwGfchwOp convOp;
  CollapseShapeOp outputCollapseOp;
  mlir::ArrayAttr collapseReassociation;
  ExpandShapeOp activationExpandOp;
  ExpandShapeOp filterExpandOp;
  ExpandShapeOp outputExpandOp;
  FillOp fillOp;
  BroadcastOp broadcastOp;
  mlir::Value sourceActivation;
  ConstantOp filterRank4Const;
  mlir::Value bias;
  mlir::RankedTensorType sourceActivationTy;
  mlir::RankedTensorType groupedActivationTy;
  mlir::RankedTensorType filterRank4Ty;
  mlir::RankedTensorType groupedFilterTy;
  mlir::RankedTensorType groupedOutputTy;
  mlir::RankedTensorType collapsedOutputTy;
  llvm::SmallVector<int64_t> strides;
  int64_t g;
  int64_t cTotal;
  int64_t cg;
  int64_t h;
  int64_t w;
  int64_t fTotal;
  int64_t fg;
  int64_t kh;
  int64_t kw;
  int64_t oh;
  int64_t ow;
  bool hasBias;
};

struct PreparedGroupedFilter {
  ConstantOp filterRank2Const;
  mlir::Value partitionedMatrix;
  mlir::scf::ForOp placementLoop;
  TransposeOp transposeOp;
  EmptyOp transposeInit;
  mlir::Value transposedFilter;
  int64_t matrixId;
};

struct PreparedGroupedBias {
  mlir::Value bias;
};

struct Conv2DGroupedLoweringState {
  mlir::Location loc;
  mlir::Type elementType;
  int64_t channelPatchWidth;
  mlir::RankedTensorType patchTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType groupedOutputTy;
  mlir::RankedTensorType collapsedOutputTy;
  mlir::Value c0;
  mlir::Value c1;
  mlir::Value ohUpper;
  mlir::Value owUpper;
  mlir::Value cUpper;
  mlir::Value khUpper;
  mlir::Value kwUpper;
  mlir::Value fUpper;
  mlir::Value fgValue;
  mlir::Value channelPatchWidthValue;
  mlir::Value strideH;
  mlir::Value strideW;
  mlir::Value kwValue;
  mlir::Value zeroValue;
};

static bool extractTwoPositiveInts(mlir::DenseIntElementsAttr attr,
                                   llvm::SmallVectorImpl<int64_t> &values) {
  values.clear();
  if (!attr)
    return false;

  for (llvm::APInt value : attr.getValues<llvm::APInt>())
    values.push_back(value.getSExtValue());

  if (values.size() != 2)
    return false;

  return values[0] > 0 && values[1] > 0;
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

static bool hasExpectedGroupedExpandShape(ExpandShapeOp expandOp,
                                          int64_t sourceRank,
                                          int64_t resultRank) {
  if (!expandOp)
    return false;

  auto sourceTy =
      llvm::dyn_cast<mlir::RankedTensorType>(expandOp.getSrc().getType());
  auto resultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(expandOp.getResult().getType());
  if (!sourceTy || !resultTy || !sourceTy.hasStaticShape() ||
      !resultTy.hasStaticShape())
    return false;

  if (sourceTy.getRank() != sourceRank || resultTy.getRank() != resultRank)
    return false;

  return sourceTy.getElementType().isF32() && resultTy.getElementType().isF32();
}

static mlir::FailureOr<SupportedGroupedActivation>
getSupportedGroupedActivation(Conv2DNgchwGfchwOp convOp) {
  auto groupedActivationTy = llvm::dyn_cast<mlir::RankedTensorType>(
      convOp.getInputs()[0].getType());
  auto expandOp = convOp.getInputs()[0].getDefiningOp<ExpandShapeOp>();
  if (!groupedActivationTy || !expandOp ||
      !hasExpectedGroupedExpandShape(expandOp, /*sourceRank=*/4,
                                     /*resultRank=*/5))
    return mlir::failure();

  auto sourceActivationTy = llvm::dyn_cast<mlir::RankedTensorType>(
      expandOp.getSrc().getType());
  if (!sourceActivationTy || !sourceActivationTy.hasStaticShape() ||
      !sourceActivationTy.getElementType().isF32())
    return mlir::failure();

  return SupportedGroupedActivation{expandOp, expandOp.getSrc(),
                                    sourceActivationTy, groupedActivationTy};
}

static mlir::FailureOr<SupportedGroupedFilter>
getSupportedGroupedFilter(Conv2DNgchwGfchwOp convOp) {
  auto groupedFilterTy = llvm::dyn_cast<mlir::RankedTensorType>(
      convOp.getInputs()[1].getType());
  auto expandOp = convOp.getInputs()[1].getDefiningOp<ExpandShapeOp>();
  if (!groupedFilterTy || !expandOp ||
      !hasExpectedGroupedExpandShape(expandOp, /*sourceRank=*/4,
                                     /*resultRank=*/5))
    return mlir::failure();

  auto filterRank4Const = expandOp.getSrc().getDefiningOp<ConstantOp>();
  auto filterRank4Ty =
      filterRank4Const
          ? llvm::dyn_cast<mlir::RankedTensorType>(filterRank4Const.getType())
          : mlir::RankedTensorType();
  if (!filterRank4Const || !filterRank4Ty || !filterRank4Ty.hasStaticShape() ||
      filterRank4Ty.getRank() != 4 || !filterRank4Ty.getElementType().isF32())
    return mlir::failure();

  return SupportedGroupedFilter{expandOp, filterRank4Const, filterRank4Ty,
                                groupedFilterTy};
}

static mlir::FailureOr<SupportedGroupedOutputInit>
getSupportedGroupedZeroFill(mlir::Value outputInit,
                            mlir::RankedTensorType groupedOutputTy) {
  auto fillOp = outputInit.getDefiningOp<FillOp>();
  if (!fillOp)
    return mlir::failure();

  auto fillTy =
      llvm::dyn_cast<mlir::RankedTensorType>(fillOp.getResult(0).getType());
  if (!fillTy || fillTy != groupedOutputTy)
    return mlir::failure();

  if (fillOp.getInputs().size() != 1 || !fillOp.getInputs()[0].getType().isF32())
    return mlir::failure();

  if (!isZeroF32Constant(fillOp.getInputs()[0]))
    return mlir::failure();

  auto outputExpandOp = fillOp->getOperand(1).getDefiningOp<ExpandShapeOp>();
  if (!outputExpandOp ||
      !hasExpectedGroupedExpandShape(outputExpandOp, /*sourceRank=*/4,
                                     /*resultRank=*/5))
    return mlir::failure();

  auto sourceOutputTy = llvm::dyn_cast<mlir::RankedTensorType>(
      outputExpandOp.getSrc().getType());
  if (!sourceOutputTy || !sourceOutputTy.hasStaticShape() ||
      sourceOutputTy.getRank() != 4 ||
      !sourceOutputTy.getElementType().isF32())
    return mlir::failure();

  SupportedGroupedOutputInit output;
  output.fillOp = fillOp;
  output.outputExpandOp = outputExpandOp;
  output.sourceOutputTy = sourceOutputTy;
  return output;
}

static mlir::FailureOr<SupportedGroupedOutputInit>
getSupportedGroupedBiasBroadcast(mlir::Value outputInit,
                                 mlir::RankedTensorType groupedOutputTy) {
  auto outputExpandOp = outputInit.getDefiningOp<ExpandShapeOp>();
  if (!outputExpandOp ||
      !hasExpectedGroupedExpandShape(outputExpandOp, /*sourceRank=*/4,
                                     /*resultRank=*/5))
    return mlir::failure();

  if (llvm::dyn_cast<mlir::RankedTensorType>(outputExpandOp.getResult().getType()) !=
      groupedOutputTy)
    return mlir::failure();

  auto broadcastOp = outputExpandOp.getSrc().getDefiningOp<BroadcastOp>();
  if (!broadcastOp)
    return mlir::failure();

  auto sourceOutputTy = llvm::dyn_cast<mlir::RankedTensorType>(
      outputExpandOp.getSrc().getType());
  auto biasTy =
      llvm::dyn_cast<mlir::RankedTensorType>(broadcastOp.getInput().getType());
  auto biasBroadcastTy = llvm::dyn_cast<mlir::RankedTensorType>(
      broadcastOp.getResult().front().getType());
  if (!sourceOutputTy || !biasTy || !biasBroadcastTy ||
      !sourceOutputTy.hasStaticShape() || !biasTy.hasStaticShape() ||
      !biasBroadcastTy.hasStaticShape())
    return mlir::failure();

  if (sourceOutputTy.getRank() != 4 || biasTy.getRank() != 1 ||
      !sourceOutputTy.getElementType().isF32() ||
      !biasTy.getElementType().isF32() || biasBroadcastTy != sourceOutputTy)
    return mlir::failure();

  auto dims = broadcastOp.getDimensions();
  if (dims.size() != 3 || dims[0] != 0 || dims[1] != 2 || dims[2] != 3)
    return mlir::failure();

  SupportedGroupedOutputInit output;
  output.broadcastOp = broadcastOp;
  output.outputExpandOp = outputExpandOp;
  output.bias = broadcastOp.getInput();
  output.sourceOutputTy = sourceOutputTy;
  output.biasTy = biasTy;
  output.biasBroadcastTy = biasBroadcastTy;
  output.hasBias = true;
  return output;
}

static mlir::FailureOr<llvm::SmallVector<int64_t>>
getSupportedGroupedStrides(Conv2DNgchwGfchwOp convOp) {
  llvm::SmallVector<int64_t> dilations;
  if (!extractTwoPositiveInts(convOp.getDilations(), dilations))
    return mlir::failure();
  if (dilations[0] != 1 || dilations[1] != 1)
    return mlir::failure();

  llvm::SmallVector<int64_t> strides;
  if (!extractTwoPositiveInts(convOp.getStrides(), strides))
    return mlir::failure();

  return strides;
}

static mlir::FailureOr<std::pair<CollapseShapeOp, mlir::RankedTensorType>>
getSupportedGroupedOutputCollapse(Conv2DNgchwGfchwOp convOp) {
  if (!convOp.getResult(0).hasOneUse())
    return mlir::failure();

  auto collapseOp =
      llvm::dyn_cast<CollapseShapeOp>(*convOp.getResult(0).getUsers().begin());
  if (!collapseOp || collapseOp.getSrc() != convOp.getResult(0))
    return mlir::failure();

  auto collapsedOutputTy =
      llvm::dyn_cast<mlir::RankedTensorType>(collapseOp.getResult().getType());
  if (!collapsedOutputTy || !collapsedOutputTy.hasStaticShape() ||
      !collapsedOutputTy.getElementType().isF32())
    return mlir::failure();

  if (collapsedOutputTy.getRank() != 2 && collapsedOutputTy.getRank() != 4)
    return mlir::failure();

  return std::make_pair(collapseOp, collapsedOutputTy);
}

static mlir::FailureOr<Conv2DGroupedShapeInfo>
getValidatedGroupedShapeInfo(mlir::RankedTensorType sourceActivationTy,
                             mlir::RankedTensorType groupedActivationTy,
                             mlir::RankedTensorType filterRank4Ty,
                             mlir::RankedTensorType groupedFilterTy,
                             mlir::RankedTensorType groupedOutputTy,
                             mlir::RankedTensorType sourceOutputTy,
                             mlir::RankedTensorType collapsedOutputTy,
                             const SupportedGroupedOutputInit &outputInit,
                             llvm::ArrayRef<int64_t> strides) {
  auto sourceActivationShape = sourceActivationTy.getShape();
  auto groupedActivationShape = groupedActivationTy.getShape();
  auto filterRank4Shape = filterRank4Ty.getShape();
  auto groupedFilterShape = groupedFilterTy.getShape();
  auto groupedOutputShape = groupedOutputTy.getShape();
  auto sourceOutputShape = sourceOutputTy.getShape();
  auto collapsedOutputShape = collapsedOutputTy.getShape();

  Conv2DGroupedShapeInfo shapeInfo{
      groupedActivationShape[1], sourceActivationShape[1],
      groupedActivationShape[2], groupedActivationShape[3],
      groupedActivationShape[4], filterRank4Shape[0], groupedFilterShape[1],
      filterRank4Shape[2],      filterRank4Shape[3], groupedOutputShape[3],
      groupedOutputShape[4],
  };

  if (sourceActivationShape[0] != 1 || groupedActivationShape[0] != 1 ||
      groupedOutputShape[0] != 1 || sourceOutputShape[0] != 1)
    return mlir::failure();

  if (shapeInfo.g <= 0 || shapeInfo.cg <= 0 || shapeInfo.fg <= 0)
    return mlir::failure();

  if (shapeInfo.cTotal != shapeInfo.g * shapeInfo.cg)
    return mlir::failure();
  if (shapeInfo.fTotal != shapeInfo.g * shapeInfo.fg)
    return mlir::failure();

  if (sourceActivationShape[2] != shapeInfo.h ||
      sourceActivationShape[3] != shapeInfo.w)
    return mlir::failure();

  if (filterRank4Shape[1] != shapeInfo.cg)
    return mlir::failure();

  if (groupedFilterShape[0] != shapeInfo.g ||
      groupedFilterShape[1] != shapeInfo.fg ||
      groupedFilterShape[2] != shapeInfo.cg ||
      groupedFilterShape[3] != shapeInfo.kh ||
      groupedFilterShape[4] != shapeInfo.kw)
    return mlir::failure();

  if (groupedOutputShape[1] != shapeInfo.g ||
      groupedOutputShape[2] != shapeInfo.fg)
    return mlir::failure();

  if (sourceOutputShape[1] != shapeInfo.fTotal ||
      sourceOutputShape[2] != shapeInfo.oh ||
      sourceOutputShape[3] != shapeInfo.ow)
    return mlir::failure();

  if (shapeInfo.kh > shapeInfo.h || shapeInfo.kw > shapeInfo.w)
    return mlir::failure();

  int64_t expectedOh = ((shapeInfo.h - shapeInfo.kh) / strides[0]) + 1;
  int64_t expectedOw = ((shapeInfo.w - shapeInfo.kw) / strides[1]) + 1;
  if (shapeInfo.oh != expectedOh || shapeInfo.ow != expectedOw)
    return mlir::failure();

  if (collapsedOutputTy.getRank() == 4) {
    if (collapsedOutputShape[0] != 1 || collapsedOutputShape[1] != shapeInfo.fTotal ||
        collapsedOutputShape[2] != shapeInfo.oh ||
        collapsedOutputShape[3] != shapeInfo.ow)
      return mlir::failure();
  } else {
    if (collapsedOutputShape[0] != 1 ||
        collapsedOutputShape[1] != shapeInfo.fTotal * shapeInfo.oh * shapeInfo.ow)
      return mlir::failure();
  }

  if (outputInit.hasBias) {
    if (!outputInit.biasTy || !outputInit.biasBroadcastTy)
      return mlir::failure();
    if (outputInit.biasTy.getShape()[0] != shapeInfo.fTotal)
      return mlir::failure();
    if (outputInit.biasBroadcastTy != sourceOutputTy)
      return mlir::failure();
  }

  return shapeInfo;
}

static mlir::FailureOr<Conv2DGroupedMatch>
matchSupportedGroupedConv2D(Conv2DNgchwGfchwOp convOp) {
  if (convOp.getInputs().size() != 2 || convOp.getOutputs().size() != 1)
    return mlir::failure();

  auto groupedOutputTy =
      llvm::dyn_cast<mlir::RankedTensorType>(convOp.getResult(0).getType());
  if (!groupedOutputTy || !groupedOutputTy.hasStaticShape() ||
      groupedOutputTy.getRank() != 5 ||
      !groupedOutputTy.getElementType().isF32())
    return mlir::failure();

  auto activation = getSupportedGroupedActivation(convOp);
  if (failed(activation))
    return mlir::failure();

  auto filter = getSupportedGroupedFilter(convOp);
  if (failed(filter))
    return mlir::failure();

  auto strides = getSupportedGroupedStrides(convOp);
  if (failed(strides))
    return mlir::failure();

  auto collapse = getSupportedGroupedOutputCollapse(convOp);
  if (failed(collapse))
    return mlir::failure();

  auto outputInit = getSupportedGroupedZeroFill(convOp.getOutputs()[0],
                                                groupedOutputTy);
  if (failed(outputInit)) {
    outputInit =
        getSupportedGroupedBiasBroadcast(convOp.getOutputs()[0], groupedOutputTy);
    if (failed(outputInit))
      return mlir::failure();
  }

  auto shapeInfo = getValidatedGroupedShapeInfo(
      activation->sourceActivationTy, activation->groupedActivationTy,
      filter->filterRank4Ty, filter->groupedFilterTy, groupedOutputTy,
      outputInit->sourceOutputTy, collapse->second, *outputInit, *strides);
  if (failed(shapeInfo))
    return mlir::failure();

  return Conv2DGroupedMatch{
      convOp,
      collapse->first,
      collapse->first.getReassociationAttr(),
      activation->expandOp,
      filter->expandOp,
      outputInit->outputExpandOp,
      outputInit->fillOp,
      outputInit->broadcastOp,
      activation->sourceActivation,
      filter->filterRank4Const,
      outputInit->bias,
      activation->sourceActivationTy,
      activation->groupedActivationTy,
      filter->filterRank4Ty,
      filter->groupedFilterTy,
      groupedOutputTy,
      collapse->second,
      *strides,
      shapeInfo->g,
      shapeInfo->cTotal,
      shapeInfo->cg,
      shapeInfo->h,
      shapeInfo->w,
      shapeInfo->fTotal,
      shapeInfo->fg,
      shapeInfo->kh,
      shapeInfo->kw,
      shapeInfo->oh,
      shapeInfo->ow,
      outputInit->hasBias,
  };
}

static mlir::RankedTensorType
buildGroupedFlattenedTensorType(const Conv2DGroupedMatch &match) {
  return mlir::RankedTensorType::get(
      {match.fTotal, match.cTotal * match.kh * match.kw},
      match.filterRank4Ty.getElementType());
}

static mlir::FailureOr<llvm::SmallVector<float>>
getFilterValues(ConstantOp filterConst) {
  if (auto denseAttr =
          llvm::dyn_cast<mlir::DenseFPElementsAttr>(filterConst.getValue())) {
    llvm::SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (const llvm::APFloat &value : denseAttr.getValues<llvm::APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (auto denseResourceAttr =
          llvm::dyn_cast<mlir::DenseF32ResourceElementsAttr>(
              filterConst.getValue())) {
    std::optional<llvm::ArrayRef<float>> values =
        denseResourceAttr.tryGetAsArrayRef();
    if (!values)
      return mlir::failure();
    return llvm::SmallVector<float>(values->begin(), values->end());
  }

  return mlir::failure();
}

static mlir::TypedAttr
buildGroupedBlockDiagonalFilterAttr(ConstantOp filterConst,
                                    const Conv2DGroupedMatch &match,
                                    mlir::RankedTensorType flattenedTy) {
  auto maybeValues = getFilterValues(filterConst);
  if (failed(maybeValues))
    return {};

  llvm::SmallVector<float> sourceValues = *maybeValues;
  llvm::SmallVector<float> flattenedValues(flattenedTy.getNumElements(), 0.0f);
  int64_t flattenedCols = flattenedTy.getShape()[1];

  auto sourceIndex = [&](int64_t f, int64_t cgIdx, int64_t khIdx,
                         int64_t kwIdx) {
    return (((f * match.cg + cgIdx) * match.kh + khIdx) * match.kw + kwIdx);
  };

  auto destIndex = [&](int64_t f, int64_t channel, int64_t khIdx,
                       int64_t kwIdx) {
    int64_t channelOffset = channel * (match.kh * match.kw);
    int64_t khOffset = khIdx * match.kw;
    int64_t flatIndex = channelOffset + khOffset + kwIdx;
    return f * flattenedCols + flatIndex;
  };

  for (int64_t group = 0; group < match.g; ++group) {
    for (int64_t fgIdx = 0; fgIdx < match.fg; ++fgIdx) {
      int64_t f = group * match.fg + fgIdx;
      for (int64_t cgIdx = 0; cgIdx < match.cg; ++cgIdx) {
        int64_t channel = group * match.cg + cgIdx;
        for (int64_t khIdx = 0; khIdx < match.kh; ++khIdx) {
          for (int64_t kwIdx = 0; kwIdx < match.kw; ++kwIdx) {
            flattenedValues[destIndex(f, channel, khIdx, kwIdx)] =
                sourceValues[sourceIndex(f, cgIdx, khIdx, kwIdx)];
          }
        }
      }
    }
  }

  if (llvm::isa<mlir::DenseF32ResourceElementsAttr>(filterConst.getValue())) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        "analog_grouped_conv2d_filter_" + std::to_string(nextResourceId++);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(flattenedValues), /*dataIsMutable=*/false);
    return llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
        flattenedTy, resourceName, std::move(blob)));
  }

  return llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(
      flattenedTy, llvm::ArrayRef<float>(flattenedValues)));
}

static mlir::FailureOr<ConstantOp>
createBlockDiagonalFilter(const Conv2DGroupedMatch &match,
                          mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType flattenedTy = buildGroupedFlattenedTensorType(match);
  mlir::TypedAttr flattenedAttr =
      buildGroupedBlockDiagonalFilterAttr(match.filterRank4Const, match,
                                          flattenedTy);
  if (!flattenedAttr)
    return mlir::failure();

  rewriter.setInsertionPointAfter(match.filterRank4Const);
  return rewriter.create<ConstantOp>(match.sourceActivation.getLoc(),
                                     flattenedTy, flattenedAttr);
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

static Conv2DGroupedLoweringState
buildGroupedLoweringState(mlir::OpBuilder &builder,
                          const Conv2DGroupedMatch &match) {
  mlir::Location loc = match.sourceActivation.getLoc();
  mlir::Type elementType = match.sourceActivationTy.getElementType();
  int64_t channelPatchWidth = match.kh * match.kw;

  return Conv2DGroupedLoweringState{
      loc,
      elementType,
      channelPatchWidth,
      mlir::RankedTensorType::get({1, match.cTotal * channelPatchWidth},
                                  elementType),
      mlir::RankedTensorType::get({1, match.fTotal}, elementType),
      match.groupedOutputTy,
      match.collapsedOutputTy,
      builder.create<mlir::arith::ConstantIndexOp>(loc, 0),
      builder.create<mlir::arith::ConstantIndexOp>(loc, 1),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.oh),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.ow),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.cTotal),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.kh),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.kw),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.fTotal),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.fg),
      builder.create<mlir::arith::ConstantIndexOp>(loc, channelPatchWidth),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.strides[0]),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.strides[1]),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.kw),
      builder.create<mlir::arith::ConstantFloatOp>(
          loc, llvm::cast<mlir::FloatType>(elementType), llvm::APFloat(0.0f)),
  };
}

static mlir::FailureOr<PreparedGroupedFilter>
prepareGroupedFilter(const Conv2DGroupedMatch &match,
                     const Conv2DGroupedLoweringState &state,
                     mlir::RewriterBase &rewriter, int64_t arrayRows,
                     int64_t arrayCols) {
  auto filterRank2Const = createBlockDiagonalFilter(match, rewriter);
  if (failed(filterRank2Const))
    return mlir::failure();

  auto analogMatrix =
      converter_utils::materializeAnalogMatrix(*filterRank2Const, rewriter);
  if (failed(analogMatrix))
    return mlir::failure();

  auto matrixId = converter_utils::getOrSetMatrixId(*analogMatrix, rewriter);
  if (failed(matrixId))
    return mlir::failure();

  auto partitionedMatrix = converter_utils::partitionAnalogMatrix(
      *analogMatrix, rewriter, arrayRows, arrayCols);
  if (failed(partitionedMatrix))
    return mlir::failure();

  auto placementLoop =
      converter_utils::placeAnalogMatrix(*partitionedMatrix, rewriter);
  if (failed(placementLoop))
    return mlir::failure();

  rewriter.setInsertionPointAfter(placementLoop->getOperation());
  auto transposeInit = rewriter.create<EmptyOp>(
      state.loc,
      llvm::ArrayRef<int64_t>{state.patchTy.getShape()[1], match.fTotal},
      state.elementType);
  auto transposeOp = rewriter.create<TransposeOp>(
      state.loc, filterRank2Const->getResult(), transposeInit,
      llvm::ArrayRef<int64_t>{1, 0});

  return PreparedGroupedFilter{*filterRank2Const,
                               *partitionedMatrix,
                               *placementLoop,
                               transposeOp,
                               transposeInit,
                               transposeOp.getResult().front(),
                               *matrixId};
}

static PreparedGroupedBias prepareGroupedBias(const Conv2DGroupedMatch &match) {
  PreparedGroupedBias preparedBias;
  if (!match.hasBias)
    return preparedBias;

  preparedBias.bias = match.bias;
  return preparedBias;
}

static mlir::Value buildGroupedFlattenedPatch(
    mlir::OpBuilder &builder, const Conv2DGroupedMatch &match,
    const Conv2DGroupedLoweringState &state, mlir::Value ohIdx,
    mlir::Value owIdx) {
  mlir::Value patchInit = builder.create<EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);
  auto channelLoop = builder.create<mlir::scf::ForOp>(
      state.loc, state.c0, state.cUpper, state.c1, mlir::ValueRange{patchInit},
      [&](mlir::OpBuilder &channelBuilder, mlir::Location channelLoc,
          mlir::Value cIdx, mlir::ValueRange channelIterArgs) {
        auto khLoop = channelBuilder.create<mlir::scf::ForOp>(
            channelLoc, state.c0, state.khUpper, state.c1, channelIterArgs,
            [&](mlir::OpBuilder &khBuilder, mlir::Location khLoc,
                mlir::Value khIdx, mlir::ValueRange khIterArgs) {
              auto kwLoop = khBuilder.create<mlir::scf::ForOp>(
                  khLoc, state.c0, state.kwUpper, state.c1, khIterArgs,
                  [&](mlir::OpBuilder &kwBuilder, mlir::Location kwLoc,
                      mlir::Value kwIdx, mlir::ValueRange kwIterArgs) {
                    mlir::Value ihBase = kwBuilder.create<mlir::arith::MulIOp>(
                        kwLoc, ohIdx, state.strideH);
                    mlir::Value iwBase = kwBuilder.create<mlir::arith::MulIOp>(
                        kwLoc, owIdx, state.strideW);
                    mlir::Value ih = kwBuilder.create<mlir::arith::AddIOp>(
                        kwLoc, ihBase, khIdx);
                    mlir::Value iw = kwBuilder.create<mlir::arith::AddIOp>(
                        kwLoc, iwBase, kwIdx);
                    mlir::Value inputValue =
                        kwBuilder.create<mlir::tensor::ExtractOp>(
                            kwLoc, match.sourceActivation,
                            mlir::ValueRange{state.c0, cIdx, ih, iw});
                    mlir::Value channelOffset =
                        kwBuilder.create<mlir::arith::MulIOp>(
                            kwLoc, cIdx, state.channelPatchWidthValue);
                    mlir::Value khOffset = kwBuilder.create<mlir::arith::MulIOp>(
                        kwLoc, khIdx, state.kwValue);
                    mlir::Value patchOffset =
                        kwBuilder.create<mlir::arith::AddIOp>(
                            kwLoc, channelOffset, khOffset);
                    mlir::Value flatIndex = kwBuilder.create<mlir::arith::AddIOp>(
                        kwLoc, patchOffset, kwIdx);
                    mlir::Value updatedPatch =
                        kwBuilder.create<mlir::tensor::InsertOp>(
                            kwLoc, inputValue, kwIterArgs[0],
                            mlir::ValueRange{state.c0, flatIndex});
                    kwBuilder.create<mlir::scf::YieldOp>(kwLoc, updatedPatch);
                  });
              khBuilder.create<mlir::scf::YieldOp>(khLoc, kwLoop.getResult(0));
            });
        channelBuilder.create<mlir::scf::YieldOp>(channelLoc,
                                                  khLoop.getResult(0));
      });
  return channelLoop.getResult(0);
}

static MatmulOp buildGroupedPatchMatmul(mlir::OpBuilder &builder,
                                        const Conv2DGroupedLoweringState &state,
                                        mlir::Value patch,
                                        mlir::Value transposedFilter) {
  mlir::Value matmulInit = buildZeroInitializedTensor(
      builder, state.loc, state.matmulResultTy, state.zeroValue);
  return builder.create<MatmulOp>(state.loc, state.matmulResultTy,
                                  mlir::ValueRange{patch, transposedFilter},
                                  mlir::ValueRange{matmulInit});
}

static void eraseUnusedMatmulScaffold(MatmulOp matmulOp) {
  FillOp fillOp;
  EmptyOp emptyOp;
  if (matmulOp->getNumOperands() >= 3) {
    fillOp = matmulOp->getOperand(2).getDefiningOp<FillOp>();
    if (fillOp && fillOp->getNumOperands() >= 2)
      emptyOp = fillOp->getOperand(1).getDefiningOp<EmptyOp>();
  }

  if (matmulOp && matmulOp->use_empty())
    matmulOp->erase();
  if (fillOp && fillOp->use_empty())
    fillOp->erase();
  if (emptyOp && emptyOp->use_empty())
    emptyOp->erase();
}

static mlir::FailureOr<mlir::Value>
executeGroupedPatchOnAnalog(mlir::Value patch,
                            const PreparedGroupedFilter &preparedFilter,
                            const Conv2DGroupedLoweringState &state,
                            mlir::OpBuilder &builder, int64_t arrayRows,
                            int64_t arrayCols) {
  auto analogVector = converter_utils::materializeAnalogVector(
      patch, preparedFilter.matrixId, builder);
  if (failed(analogVector))
    return mlir::failure();

  auto partitionedVector = converter_utils::partitionAnalogVector(
      *analogVector, builder, arrayRows, arrayCols);
  if (failed(partitionedVector))
    return mlir::failure();

  auto placementLoop =
      converter_utils::placeAnalogVector(*partitionedVector, builder);
  if (failed(placementLoop))
    return mlir::failure();

  auto executionBuffer = converter_utils::insertArrayExecution(
      preparedFilter.partitionedMatrix, *partitionedVector,
      preparedFilter.placementLoop, *placementLoop, builder);
  if (failed(executionBuffer))
    return mlir::failure();

  MatmulOp matmulOp = buildGroupedPatchMatmul(
      builder, state, patch, preparedFilter.transposedFilter);
  auto reducedTensor = converter_utils::insertArrayReduction(
      *executionBuffer, preparedFilter.partitionedMatrix, matmulOp, builder);
  if (failed(reducedTensor))
    return mlir::failure();

  eraseUnusedMatmulScaffold(matmulOp);
  return *reducedTensor;
}

static mlir::Value applyOptionalGroupedBias(
    mlir::OpBuilder &builder, const Conv2DGroupedLoweringState &state,
    const PreparedGroupedBias &preparedBias, mlir::Value channelResult) {
  if (!preparedBias.bias)
    return channelResult;

  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1}};
  mlir::Value expandedBias = builder.create<mlir::tensor::ExpandShapeOp>(
      state.loc, state.matmulResultTy, preparedBias.bias, reassociation);
  mlir::Value biasedInit = builder.create<EmptyOp>(
      state.loc, state.matmulResultTy.getShape(), state.elementType);
  return builder
      .create<mlir::linalg::AddOp>(
          state.loc, mlir::ValueRange{channelResult, expandedBias},
          mlir::ValueRange{biasedInit})
      .getResult(0);
}

static void storeGroupedOutputChannels(mlir::OpBuilder &builder,
                                       const Conv2DGroupedLoweringState &state,
                                       mlir::Value channelResult,
                                       mlir::Value outputBuffer,
                                       mlir::Value ohIdx,
                                       mlir::Value owIdx) {
  int64_t numChannels = state.matmulResultTy.getShape()[1];
  for (int64_t channel = 0; channel < numChannels; ++channel) {
    mlir::Value fIdx =
        builder.create<mlir::arith::ConstantIndexOp>(state.loc, channel);
    mlir::Value channelValue = builder.create<mlir::tensor::ExtractOp>(
        state.loc, channelResult, mlir::ValueRange{state.c0, fIdx});
    mlir::Value groupIdx =
        builder.create<mlir::arith::DivUIOp>(state.loc, fIdx, state.fgValue);
    mlir::Value fgIdx =
        builder.create<mlir::arith::RemUIOp>(state.loc, fIdx, state.fgValue);
    builder.create<mlir::memref::StoreOp>(
        state.loc, channelValue, outputBuffer,
        mlir::ValueRange{state.c0, groupIdx, fgIdx, ohIdx, owIdx});
  }
}

static mlir::LogicalResult lowerGroupedOutputPosition(
    mlir::OpBuilder &builder, const Conv2DGroupedMatch &match,
    const PreparedGroupedFilter &preparedFilter,
    const PreparedGroupedBias &preparedBias,
    const Conv2DGroupedLoweringState &state, mlir::Value outputBuffer,
    mlir::Value ohIdx, mlir::Value owIdx, int64_t arrayRows,
    int64_t arrayCols) {
  mlir::Block *outputBody = builder.getBlock();
  mlir::Value patch =
      buildGroupedFlattenedPatch(builder, match, state, ohIdx, owIdx);
  auto channelResult = executeGroupedPatchOnAnalog(
      patch, preparedFilter, state, builder, arrayRows, arrayCols);
  if (failed(channelResult))
    return mlir::failure();

  mlir::OpBuilder outputBuilder(builder.getContext());
  outputBuilder.setInsertionPoint(outputBody->getTerminator());
  mlir::Value biasedResult = applyOptionalGroupedBias(
      outputBuilder, state, preparedBias, *channelResult);
  storeGroupedOutputChannels(outputBuilder, state, biasedResult, outputBuffer,
                             ohIdx, owIdx);
  return mlir::success();
}

static mlir::FailureOr<mlir::Value>
emitGroupedOutputLoops(mlir::RewriterBase &rewriter,
                       const Conv2DGroupedMatch &match,
                       const PreparedGroupedFilter &preparedFilter,
                       const PreparedGroupedBias &preparedBias,
                       const Conv2DGroupedLoweringState &state,
                       int64_t arrayRows, int64_t arrayCols) {
  rewriter.setInsertionPointAfter(match.outputCollapseOp);
  auto outputBufferType = mlir::MemRefType::get(
      state.groupedOutputTy.getShape(), state.groupedOutputTy.getElementType());
  mlir::Value outputBuffer =
      rewriter.create<mlir::memref::AllocOp>(state.loc, outputBufferType);

  bool failedLowering = false;
  loop_utils::build2DIndexLoopNest(
      rewriter, state.loc, match.oh, match.ow,
      [&](mlir::OpBuilder &loopBuilder, mlir::Location loopLoc,
          mlir::Value ohIdx, mlir::Value owIdx) {
        (void)loopLoc;
        if (failedLowering)
          return;
        mlir::OpBuilder bodyBuilder(loopBuilder.getContext());
        bodyBuilder.setInsertionPoint(loopBuilder.getBlock()->getTerminator());
        if (failed(lowerGroupedOutputPosition(bodyBuilder, match,
                                              preparedFilter, preparedBias,
                                              state, outputBuffer, ohIdx, owIdx,
                                              arrayRows, arrayCols)))
          failedLowering = true;
      });

  if (failedLowering)
    return mlir::failure();

  auto toTensor = rewriter.create<mlir::bufferization::ToTensorOp>(
      state.loc, state.groupedOutputTy, outputBuffer);
  toTensor->setAttr("restrict", rewriter.getUnitAttr());

  auto collapsedOutput = rewriter.create<CollapseShapeOp>(
      state.loc, state.collapsedOutputTy, toTensor.getResult(),
      match.collapseReassociation);
  return collapsedOutput.getResult();
}

static void eraseIfUnused(mlir::Operation *op, mlir::RewriterBase &rewriter) {
  if (op && op->use_empty())
    rewriter.eraseOp(op);
}

static void eraseUnusedGroupedPreparedFilterOps(
    PreparedGroupedFilter &preparedFilter, mlir::RewriterBase &rewriter) {
  eraseIfUnused(preparedFilter.transposeOp.getOperation(), rewriter);
  eraseIfUnused(preparedFilter.transposeInit.getOperation(), rewriter);
}

static void eraseUnusedGroupedConv2DOps(Conv2DGroupedMatch &match,
                                        mlir::RewriterBase &rewriter) {
  EmptyOp biasEmpty;
  EmptyOp outputEmpty;
  mlir::Operation *fillInput = nullptr;

  if (match.hasBias) {
    if (match.broadcastOp && match.broadcastOp->getNumOperands() >= 2)
      biasEmpty = match.broadcastOp->getOperand(1).getDefiningOp<EmptyOp>();
  } else {
    if (match.outputExpandOp)
      outputEmpty = match.outputExpandOp.getSrc().getDefiningOp<EmptyOp>();
    if (match.fillOp && match.fillOp->getNumOperands() >= 1)
      fillInput = match.fillOp->getOperand(0).getDefiningOp();
  }

  eraseIfUnused(match.outputCollapseOp.getOperation(), rewriter);
  eraseIfUnused(match.convOp.getOperation(), rewriter);

  if (match.hasBias) {
    eraseIfUnused(match.outputExpandOp.getOperation(), rewriter);
    eraseIfUnused(match.filterExpandOp.getOperation(), rewriter);
    eraseIfUnused(match.activationExpandOp.getOperation(), rewriter);
    eraseIfUnused(match.filterRank4Const.getOperation(), rewriter);
    eraseIfUnused(match.broadcastOp.getOperation(), rewriter);
    eraseIfUnused(biasEmpty.getOperation(), rewriter);
    return;
  }

  eraseIfUnused(match.fillOp.getOperation(), rewriter);
  eraseIfUnused(match.outputExpandOp.getOperation(), rewriter);
  eraseIfUnused(match.filterExpandOp.getOperation(), rewriter);
  eraseIfUnused(match.activationExpandOp.getOperation(), rewriter);
  eraseIfUnused(match.filterRank4Const.getOperation(), rewriter);
  eraseIfUnused(outputEmpty.getOperation(), rewriter);
  eraseIfUnused(fillInput, rewriter);
}

static Conv2DNgchwGfchwOp findFirstGroupedConv2DOp(mlir::func::FuncOp func) {
  Conv2DNgchwGfchwOp convOp;
  func.walk([&](Conv2DNgchwGfchwOp op) {
    if (!convOp)
      convOp = op;
  });
  return convOp;
}

// Converts extracted grouped Conv2D layer bodies into analog array execution.
class Conv2DGroupedConverter : public mlir::analog::LayerConverter {
public:
  mlir::StringRef getName() const override { return "conv2d_grouped"; }

  void convert(mlir::func::FuncOp func, int64_t arrayRows,
               int64_t arrayCols) const override {
    if (arrayRows <= 0 || arrayCols <= 0)
      return;

    Conv2DNgchwGfchwOp convOp = findFirstGroupedConv2DOp(func);
    if (!convOp)
      return;

    mlir::IRRewriter rewriter(func.getContext());
    auto match = matchSupportedGroupedConv2D(convOp);
    if (failed(match))
      return;

    rewriter.setInsertionPointAfter(match->filterExpandOp.getOperation());
    Conv2DGroupedLoweringState state = buildGroupedLoweringState(rewriter, *match);
    auto preparedFilter =
        prepareGroupedFilter(*match, state, rewriter, arrayRows, arrayCols);
    if (failed(preparedFilter))
      return;

    rewriter.setInsertionPointAfter(preparedFilter->transposeOp.getOperation());
    PreparedGroupedBias preparedBias = prepareGroupedBias(*match);

    auto rewrittenOutput = emitGroupedOutputLoops(
        rewriter, *match, *preparedFilter, preparedBias, state, arrayRows,
        arrayCols);
    if (failed(rewrittenOutput))
      return;

    match->outputCollapseOp.getResult().replaceAllUsesWith(*rewrittenOutput);
    eraseUnusedGroupedPreparedFilterOps(*preparedFilter, rewriter);
    eraseUnusedGroupedConv2DOps(*match, rewriter);
    func->setAttr("layer_domain", rewriter.getStringAttr("analog"));
  }
};

} // namespace

namespace mlir {
namespace analog {

// Registers the grouped Conv2D converter for both bias forms outlined by the
// extractor.
void registerConv2DGroupedConverter(LayerConverters &converters,
                                    LayerConverterMap &converterMap,
                                    MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<Conv2DGroupedConverter>();
  const LayerConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["conv2d_grouped"] = converterPtr;
  converterMap["conv2d_grouped_w_bias"] = converterPtr;
}

} // namespace analog
} // namespace mlir
