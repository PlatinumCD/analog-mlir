#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"

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
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <optional>

namespace converter_utils = mlir::analog::converter_utils;

namespace {

using mlir::arith::ConstantOp;
using mlir::linalg::BroadcastOp;
using mlir::linalg::Conv1DNcwFcwOp;
using mlir::linalg::FillOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::EmptyOp;

struct Conv1DShapeInfo {
  int64_t c;
  int64_t w;
  int64_t f;
  int64_t kw;
  int64_t ow;
};

struct Conv1DMatch {
  Conv1DNcwFcwOp convOp;
  FillOp fillOp;
  BroadcastOp broadcastOp;
  mlir::Value activation;
  ConstantOp filterRank3Const;
  ConstantOp filterRank2Const;
  mlir::Value bias;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType filterRank3Ty;
  mlir::RankedTensorType filterRank2Ty;
  mlir::RankedTensorType outputTy;
  int64_t stride;
  int64_t c;
  int64_t w;
  int64_t f;
  int64_t kw;
  int64_t ow;
  bool hasBias;
};

struct PreparedFilter {
  mlir::Value partitionedMatrix;
  mlir::scf::ForOp placementLoop;
  TransposeOp transposeOp;
  EmptyOp transposeInit;
  mlir::Value transposedFilter;
  int64_t matrixId;
};

struct PreparedBias {
  mlir::Value bias;
};

struct Conv1DLoweringState {
  mlir::Location loc;
  mlir::Type elementType;
  int64_t flattenedWidth;
  mlir::RankedTensorType patchTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType outputTy;
  mlir::Value c0;
  mlir::Value c1;
  mlir::Value owUpper;
  mlir::Value cUpper;
  mlir::Value kwUpper;
  mlir::Value fUpper;
  mlir::Value kwValue;
  mlir::Value strideW;
  mlir::Value zeroValue;
};

static bool extractPositiveInts(mlir::DenseIntElementsAttr attr,
                                size_t expectedSize,
                                llvm::SmallVectorImpl<int64_t> &values) {
  values.clear();
  if (!attr)
    return false;

  for (llvm::APInt value : attr.getValues<llvm::APInt>())
    values.push_back(value.getSExtValue());

  if (values.size() != expectedSize)
    return false;

  return llvm::all_of(values, [](int64_t value) { return value > 0; });
}

static mlir::RankedTensorType
buildFlattenedTensorType(mlir::RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  int64_t flattenedCols = shape[1] * shape[2];
  return mlir::RankedTensorType::get({shape[0], flattenedCols},
                                     tensorTy.getElementType());
}

static mlir::TypedAttr buildFlattenedAttr(ConstantOp op,
                                          mlir::RankedTensorType flattenedTy) {
  if (auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue()))
    return denseAttr.reshape(flattenedTy);

  if (auto resourceAttr =
          llvm::dyn_cast<mlir::DenseResourceElementsAttr>(op.getValue())) {
    return mlir::DenseResourceElementsAttr::get(flattenedTy,
                                                resourceAttr.getRawHandle());
  }

  return {};
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

    auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(
        denseAttr.getSplatValue<mlir::Attribute>());
    return floatAttr && floatAttr.getValue().isZero();
  }

  return false;
}

static mlir::FailureOr<std::pair<mlir::RankedTensorType, mlir::RankedTensorType>>
getSupportedInputAndOutputTypes(Conv1DNcwFcwOp convOp,
                                mlir::Value activation) {
  auto inputTy = llvm::dyn_cast<mlir::RankedTensorType>(activation.getType());
  auto outputTy =
      llvm::dyn_cast<mlir::RankedTensorType>(convOp.getResult(0).getType());
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape())
    return mlir::failure();

  if (inputTy.getRank() != 3 || outputTy.getRank() != 3)
    return mlir::failure();

  if (!inputTy.getElementType().isF32() || !outputTy.getElementType().isF32())
    return mlir::failure();

  return std::make_pair(inputTy, outputTy);
}

static mlir::FailureOr<FillOp>
getSupportedZeroFill(mlir::Value outputInit,
                     mlir::RankedTensorType outputTy) {
  auto fillOp = outputInit.getDefiningOp<FillOp>();
  if (!fillOp)
    return mlir::failure();

  auto fillTy =
      llvm::dyn_cast<mlir::RankedTensorType>(fillOp.getResult(0).getType());
  if (!fillTy || fillTy != outputTy)
    return mlir::failure();

  if (fillOp.getInputs().size() != 1 || !fillOp.getInputs()[0].getType().isF32())
    return mlir::failure();

  if (!isZeroF32Constant(fillOp.getInputs()[0]))
    return mlir::failure();

  return fillOp;
}

static mlir::FailureOr<std::pair<BroadcastOp, mlir::RankedTensorType>>
getSupportedBiasBroadcast(mlir::Value outputInit,
                          mlir::RankedTensorType outputTy) {
  auto broadcastOp = outputInit.getDefiningOp<BroadcastOp>();
  if (!broadcastOp)
    return mlir::failure();

  auto biasTy =
      llvm::dyn_cast<mlir::RankedTensorType>(broadcastOp.getInput().getType());
  auto broadcastTy = llvm::dyn_cast<mlir::RankedTensorType>(
      broadcastOp.getResult().front().getType());
  if (!biasTy || !broadcastTy || !biasTy.hasStaticShape() ||
      !broadcastTy.hasStaticShape())
    return mlir::failure();

  if (biasTy.getRank() != 1 || broadcastTy != outputTy)
    return mlir::failure();

  if (!biasTy.getElementType().isF32())
    return mlir::failure();

  auto dims = broadcastOp.getDimensions();
  if (dims.size() != 2 || dims[0] != 0 || dims[1] != 2)
    return mlir::failure();

  return std::make_pair(broadcastOp, biasTy);
}

static mlir::FailureOr<std::pair<ConstantOp, mlir::RankedTensorType>>
getSupportedFilterConstant(mlir::Value filter) {
  auto filterConst = filter.getDefiningOp<ConstantOp>();
  if (!filterConst)
    return mlir::failure();

  auto filterTy =
      llvm::dyn_cast<mlir::RankedTensorType>(filterConst.getType());
  if (!filterTy || !filterTy.hasStaticShape() || filterTy.getRank() != 3)
    return mlir::failure();

  if (!filterTy.getElementType().isF32())
    return mlir::failure();

  return std::make_pair(filterConst, filterTy);
}

static mlir::FailureOr<int64_t> getSupportedStride(Conv1DNcwFcwOp convOp) {
  llvm::SmallVector<int64_t> dilations;
  if (!extractPositiveInts(convOp.getDilations(), 1, dilations))
    return mlir::failure();
  if (dilations[0] != 1)
    return mlir::failure();

  llvm::SmallVector<int64_t> strides;
  if (!extractPositiveInts(convOp.getStrides(), 1, strides))
    return mlir::failure();

  return strides[0];
}

static mlir::FailureOr<Conv1DShapeInfo>
getValidatedShapeInfo(mlir::RankedTensorType inputTy,
                      std::optional<mlir::RankedTensorType> biasTy,
                      mlir::RankedTensorType filterRank3Ty,
                      mlir::RankedTensorType filterRank2Ty,
                      mlir::RankedTensorType outputTy, int64_t stride) {
  auto inputShape = inputTy.getShape();
  auto filterShape = filterRank3Ty.getShape();
  auto filterFlatShape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();

  Conv1DShapeInfo shapeInfo{
      inputShape[1],
      inputShape[2],
      filterShape[0],
      filterShape[2],
      outputShape[2],
  };

  if (inputShape[0] != 1)
    return mlir::failure();

  if (filterShape[1] != shapeInfo.c)
    return mlir::failure();

  if (outputShape[0] != 1 || outputShape[1] != shapeInfo.f)
    return mlir::failure();

  if (filterFlatShape[0] != shapeInfo.f ||
      filterFlatShape[1] != shapeInfo.c * shapeInfo.kw)
    return mlir::failure();

  if (shapeInfo.kw > shapeInfo.w)
    return mlir::failure();

  int64_t expectedOw = ((shapeInfo.w - shapeInfo.kw) / stride) + 1;
  if (shapeInfo.ow != expectedOw)
    return mlir::failure();

  if (biasTy && biasTy->getShape()[0] != shapeInfo.f)
    return mlir::failure();

  return shapeInfo;
}

static mlir::FailureOr<ConstantOp>
createFlattenedFilter(ConstantOp filterConst,
                      mlir::RankedTensorType filterRank3Ty,
                      mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType flattenedTy = buildFlattenedTensorType(filterRank3Ty);
  mlir::TypedAttr flattenedAttr = buildFlattenedAttr(filterConst, flattenedTy);
  if (!flattenedAttr)
    return mlir::failure();

  rewriter.setInsertionPointAfter(filterConst);
  return rewriter.create<ConstantOp>(filterConst.getLoc(), flattenedTy,
                                     flattenedAttr);
}

static mlir::FailureOr<Conv1DMatch>
matchSupportedConv1D(Conv1DNcwFcwOp convOp, mlir::RewriterBase &rewriter) {
  if (convOp.getInputs().size() != 2 || convOp.getOutputs().size() != 1)
    return mlir::failure();

  mlir::Value activation = convOp.getInputs()[0];
  mlir::Value filter = convOp.getInputs()[1];
  mlir::Value outputInit = convOp.getOutputs()[0];

  auto inputOutputTypes = getSupportedInputAndOutputTypes(convOp, activation);
  if (failed(inputOutputTypes))
    return mlir::failure();
  auto [inputTy, outputTy] = *inputOutputTypes;

  auto filterConstant = getSupportedFilterConstant(filter);
  if (failed(filterConstant))
    return mlir::failure();
  auto [filterRank3Const, filterRank3Ty] = *filterConstant;

  auto stride = getSupportedStride(convOp);
  if (failed(stride))
    return mlir::failure();

  FillOp fillOp;
  BroadcastOp broadcastOp;
  mlir::Value bias;
  bool hasBias = false;
  std::optional<mlir::RankedTensorType> biasTy;

  auto maybeFill = getSupportedZeroFill(outputInit, outputTy);
  if (succeeded(maybeFill)) {
    fillOp = *maybeFill;
  } else {
    auto maybeBroadcast = getSupportedBiasBroadcast(outputInit, outputTy);
    if (failed(maybeBroadcast))
      return mlir::failure();

    broadcastOp = maybeBroadcast->first;
    biasTy = maybeBroadcast->second;
    bias = broadcastOp.getInput();
    hasBias = true;
  }

  mlir::RankedTensorType filterRank2Ty = buildFlattenedTensorType(filterRank3Ty);
  auto shapeInfo = getValidatedShapeInfo(inputTy, biasTy, filterRank3Ty,
                                         filterRank2Ty, outputTy, *stride);
  if (failed(shapeInfo))
    return mlir::failure();

  auto filterRank2Const =
      createFlattenedFilter(filterRank3Const, filterRank3Ty, rewriter);
  if (failed(filterRank2Const))
    return mlir::failure();

  return Conv1DMatch{
      convOp, fillOp, broadcastOp, activation, filterRank3Const,
      *filterRank2Const, bias, inputTy, filterRank3Ty, filterRank2Ty,
      outputTy, *stride, shapeInfo->c, shapeInfo->w, shapeInfo->f,
      shapeInfo->kw, shapeInfo->ow, hasBias,
  };
}

static mlir::Value buildZeroInitializedTensor(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::RankedTensorType tensorTy,
                                              mlir::Value zeroValue) {
  mlir::Value empty = builder.create<EmptyOp>(loc, tensorTy.getShape(),
                                              tensorTy.getElementType());
  return builder
      .create<FillOp>(loc, mlir::ValueRange{zeroValue},
                      mlir::ValueRange{empty})
      .getResult(0);
}

static Conv1DLoweringState buildLoweringState(mlir::OpBuilder &builder,
                                              Conv1DMatch &match) {
  mlir::Location loc = match.convOp.getLoc();
  mlir::Type elementType = match.inputTy.getElementType();
  int64_t flattenedWidth = match.c * match.kw;

  return Conv1DLoweringState{
      loc,
      elementType,
      flattenedWidth,
      mlir::RankedTensorType::get({1, flattenedWidth}, elementType),
      mlir::RankedTensorType::get({1, match.f}, elementType),
      match.outputTy,
      builder.create<mlir::arith::ConstantIndexOp>(loc, 0),
      builder.create<mlir::arith::ConstantIndexOp>(loc, 1),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.ow),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.c),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.kw),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.f),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.kw),
      builder.create<mlir::arith::ConstantIndexOp>(loc, match.stride),
      builder.create<mlir::arith::ConstantFloatOp>(
          loc, llvm::cast<mlir::FloatType>(elementType), llvm::APFloat(0.0f)),
  };
}

static mlir::FailureOr<PreparedFilter>
prepareFilter(Conv1DMatch &match, const Conv1DLoweringState &state,
              mlir::RewriterBase &rewriter, int64_t arrayRows,
              int64_t arrayCols) {
  auto analogMatrix =
      converter_utils::materializeAnalogMatrix(match.filterRank2Const, rewriter);
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
      state.loc, llvm::ArrayRef<int64_t>{state.flattenedWidth, match.f},
      state.elementType);
  auto transposeOp = rewriter.create<TransposeOp>(
      state.loc, match.filterRank2Const.getResult(), transposeInit,
      llvm::ArrayRef<int64_t>{1, 0});

  return PreparedFilter{*partitionedMatrix, *placementLoop, transposeOp,
                        transposeInit, transposeOp.getResult().front(),
                        *matrixId};
}

static PreparedBias prepareBias(Conv1DMatch &match,
                                const Conv1DLoweringState &state,
                                mlir::OpBuilder &builder) {
  (void)state;
  (void)builder;

  PreparedBias preparedBias;
  if (!match.hasBias)
    return preparedBias;

  preparedBias.bias = match.bias;
  return preparedBias;
}

static mlir::Value buildFlattenedPatch(mlir::OpBuilder &builder,
                                       const Conv1DMatch &match,
                                       const Conv1DLoweringState &state,
                                       mlir::Value owIdx) {
  mlir::Value patchInit = builder.create<EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);
  auto channelLoop = builder.create<mlir::scf::ForOp>(
      state.loc, state.c0, state.cUpper, state.c1, mlir::ValueRange{patchInit},
      [&](mlir::OpBuilder &channelBuilder, mlir::Location channelLoc,
          mlir::Value cIdx, mlir::ValueRange channelIterArgs) {
        auto kwLoop = channelBuilder.create<mlir::scf::ForOp>(
            channelLoc, state.c0, state.kwUpper, state.c1, channelIterArgs,
            [&](mlir::OpBuilder &kwBuilder, mlir::Location kwLoc,
                mlir::Value kwIdx, mlir::ValueRange kwIterArgs) {
              mlir::Value iwBase = kwBuilder.create<mlir::arith::MulIOp>(
                  kwLoc, owIdx, state.strideW);
              mlir::Value iw =
                  kwBuilder.create<mlir::arith::AddIOp>(kwLoc, iwBase, kwIdx);
              mlir::Value inputValue = kwBuilder.create<mlir::tensor::ExtractOp>(
                  kwLoc, match.activation,
                  mlir::ValueRange{state.c0, cIdx, iw});
              mlir::Value channelOffset = kwBuilder.create<mlir::arith::MulIOp>(
                  kwLoc, cIdx, state.kwValue);
              mlir::Value flatIndex = kwBuilder.create<mlir::arith::AddIOp>(
                  kwLoc, channelOffset, kwIdx);
              mlir::Value updatedPatch = kwBuilder.create<mlir::tensor::InsertOp>(
                  kwLoc, inputValue, kwIterArgs[0],
                  mlir::ValueRange{state.c0, flatIndex});
              kwBuilder.create<mlir::scf::YieldOp>(kwLoc, updatedPatch);
            });
        channelBuilder.create<mlir::scf::YieldOp>(channelLoc,
                                                  kwLoop.getResult(0));
      });
  return channelLoop.getResult(0);
}

static MatmulOp buildPatchMatmul(mlir::OpBuilder &builder,
                                 const Conv1DLoweringState &state,
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
executePatchOnAnalog(mlir::Value patch, const PreparedFilter &preparedFilter,
                     const Conv1DLoweringState &state,
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

  MatmulOp matmulOp =
      buildPatchMatmul(builder, state, patch, preparedFilter.transposedFilter);
  auto reducedTensor = converter_utils::insertArrayReduction(
      *executionBuffer, preparedFilter.partitionedMatrix, matmulOp, builder);
  if (failed(reducedTensor))
    return mlir::failure();

  eraseUnusedMatmulScaffold(matmulOp);
  return *reducedTensor;
}

static mlir::Value applyOptionalBias(mlir::OpBuilder &builder,
                                     const Conv1DLoweringState &state,
                                     const PreparedBias &preparedBias,
                                     mlir::Value channelResult) {
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

static void storeOutputChannels(mlir::OpBuilder &builder,
                                const Conv1DLoweringState &state,
                                mlir::Value channelResult,
                                mlir::Value outputBuffer, mlir::Value owIdx) {
  int64_t numChannels = state.outputTy.getShape()[1];
  for (int64_t channel = 0; channel < numChannels; ++channel) {
    mlir::Value fIdx =
        builder.create<mlir::arith::ConstantIndexOp>(state.loc, channel);
    mlir::Value channelValue = builder.create<mlir::tensor::ExtractOp>(
        state.loc, channelResult, mlir::ValueRange{state.c0, fIdx});
    builder.create<mlir::memref::StoreOp>(
        state.loc, channelValue, outputBuffer,
        mlir::ValueRange{state.c0, fIdx, owIdx});
  }
}

static mlir::LogicalResult
lowerOutputPosition(mlir::OpBuilder &builder, const Conv1DMatch &match,
                    const PreparedFilter &preparedFilter,
                    const PreparedBias &preparedBias,
                    const Conv1DLoweringState &state,
                    mlir::Value outputBuffer, mlir::Value owIdx,
                    int64_t arrayRows, int64_t arrayCols) {
  mlir::Block *outputBody = builder.getBlock();
  mlir::Value patch = buildFlattenedPatch(builder, match, state, owIdx);
  auto channelResult = executePatchOnAnalog(
      patch, preparedFilter, state, builder, arrayRows, arrayCols);
  if (failed(channelResult))
    return mlir::failure();

  mlir::OpBuilder outputBuilder(builder.getContext());
  outputBuilder.setInsertionPoint(outputBody->getTerminator());
  mlir::Value biasedResult = applyOptionalBias(outputBuilder, state,
                                               preparedBias, *channelResult);
  storeOutputChannels(outputBuilder, state, biasedResult, outputBuffer, owIdx);
  return mlir::success();
}

static mlir::FailureOr<mlir::Value>
emitOutputLoops(mlir::RewriterBase &rewriter, const Conv1DMatch &match,
                const PreparedFilter &preparedFilter,
                const PreparedBias &preparedBias,
                const Conv1DLoweringState &state, int64_t arrayRows,
                int64_t arrayCols) {
  rewriter.setInsertionPointAfter(match.convOp);
  auto outputBufferType = mlir::MemRefType::get(state.outputTy.getShape(),
                                                state.outputTy.getElementType());
  mlir::Value outputBuffer =
      rewriter.create<mlir::memref::AllocOp>(state.loc, outputBufferType);

  auto outputLoop =
      rewriter.create<mlir::scf::ForOp>(state.loc, state.c0, state.owUpper,
                                        state.c1);
  mlir::OpBuilder loopBuilder(rewriter.getContext());
  loopBuilder.setInsertionPoint(outputLoop.getBody()->getTerminator());
  if (failed(lowerOutputPosition(loopBuilder, match, preparedFilter,
                                 preparedBias, state, outputBuffer,
                                 outputLoop.getInductionVar(), arrayRows,
                                 arrayCols)))
    return mlir::failure();

  rewriter.setInsertionPointAfter(outputLoop);
  auto toTensor = rewriter.create<mlir::bufferization::ToTensorOp>(
      state.loc, state.outputTy, outputBuffer);
  toTensor->setAttr("restrict", rewriter.getUnitAttr());
  return toTensor.getResult();
}

static void eraseIfUnused(mlir::Operation *op, mlir::RewriterBase &rewriter) {
  if (op && op->use_empty())
    rewriter.eraseOp(op);
}

static void eraseUnusedPreparedFilterOps(PreparedFilter &preparedFilter,
                                         mlir::RewriterBase &rewriter) {
  eraseIfUnused(preparedFilter.transposeOp.getOperation(), rewriter);
  eraseIfUnused(preparedFilter.transposeInit.getOperation(), rewriter);
}

static void eraseUnusedConv1DOps(Conv1DMatch &match,
                                 mlir::RewriterBase &rewriter) {
  eraseIfUnused(match.convOp.getOperation(), rewriter);
  eraseIfUnused(match.filterRank3Const.getOperation(), rewriter);

  if (match.hasBias) {
    EmptyOp biasEmpty;
    mlir::Operation *biasInput = nullptr;
    if (match.broadcastOp) {
      if (match.broadcastOp->getNumOperands() >= 2)
        biasEmpty = match.broadcastOp->getOperand(1).getDefiningOp<EmptyOp>();
      if (match.broadcastOp->getNumOperands() >= 1)
        biasInput = match.broadcastOp->getOperand(0).getDefiningOp();
    }
    eraseIfUnused(match.broadcastOp.getOperation(), rewriter);
    eraseIfUnused(biasEmpty.getOperation(), rewriter);
    eraseIfUnused(biasInput, rewriter);
    return;
  }

  EmptyOp fillEmpty;
  mlir::Operation *fillInput = nullptr;
  if (match.fillOp) {
    if (match.fillOp->getNumOperands() >= 2)
      fillEmpty = match.fillOp->getOperand(1).getDefiningOp<EmptyOp>();
    if (match.fillOp->getNumOperands() >= 1)
      fillInput = match.fillOp->getOperand(0).getDefiningOp();
  }
  eraseIfUnused(match.fillOp.getOperation(), rewriter);
  eraseIfUnused(fillEmpty.getOperation(), rewriter);
  eraseIfUnused(fillInput, rewriter);
}

static Conv1DNcwFcwOp findFirstConv1DOp(mlir::func::FuncOp func) {
  Conv1DNcwFcwOp convOp;
  func.walk([&](Conv1DNcwFcwOp op) {
    if (!convOp)
      convOp = op;
  });
  return convOp;
}

// Converts extracted conv1d layer bodies into analog array execution.
class Conv1DConverter : public mlir::analog::LayerConverter {
public:
  mlir::StringRef getName() const override { return "conv1d"; }

  void convert(mlir::func::FuncOp func, int64_t arrayRows,
               int64_t arrayCols) const override {
    if (arrayRows <= 0 || arrayCols <= 0)
      return;

    Conv1DNcwFcwOp convOp = findFirstConv1DOp(func);
    if (!convOp)
      return;

    mlir::IRRewriter rewriter(func.getContext());
    auto match = matchSupportedConv1D(convOp, rewriter);
    if (failed(match))
      return;

    Conv1DLoweringState state = buildLoweringState(rewriter, *match);
    auto preparedFilter =
        prepareFilter(*match, state, rewriter, arrayRows, arrayCols);
    if (failed(preparedFilter))
      return;

    rewriter.setInsertionPointAfter(preparedFilter->transposeOp.getOperation());
    PreparedBias preparedBias = prepareBias(*match, state, rewriter);

    auto rewrittenOutput = emitOutputLoops(rewriter, *match, *preparedFilter,
                                           preparedBias, state, arrayRows,
                                           arrayCols);
    if (failed(rewrittenOutput))
      return;

    match->convOp.getResult(0).replaceAllUsesWith(*rewrittenOutput);
    eraseUnusedPreparedFilterOps(*preparedFilter, rewriter);
    eraseUnusedConv1DOps(*match, rewriter);
    func->setAttr("layer_domain", rewriter.getStringAttr("analog"));
  }
};

} // namespace

namespace mlir {
namespace analog {

// Registers the Conv1D converter for both biased and bias-free layer slices.
void registerConv1DConverter(LayerConverters &converters,
                             LayerConverterMap &converterMap,
                             MLIRContext *context) {
  (void)context;

  auto converter = std::make_unique<Conv1DConverter>();
  const LayerConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));

  converterMap["conv1d"] = converterPtr;
  converterMap["conv1d_w_bias"] = converterPtr;
}

} // namespace analog
} // namespace mlir
