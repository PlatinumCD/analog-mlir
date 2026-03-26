#include "analog-mlir/Dialect/Analog/Transforms/RewriteGroupedConv2DToMatmul.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

static constexpr llvm::StringLiteral kMatrixSourceIdAttr =
    "analog.matrix_source_id";
static constexpr llvm::StringLiteral kDeleteInFuturePassAttr =
    "analog.delete_in_future_pass";
static constexpr llvm::StringLiteral kSlidingWindowMatmulAttr =
    "analog.sliding_window_matmul";
static constexpr llvm::StringLiteral kSlidingWindowBiasAddAttr =
    "analog.sliding_window_bias_add";
static constexpr llvm::StringLiteral kOutputChannelAssemblyAttr =
    "analog.output_channel_assembly";
static constexpr llvm::StringLiteral kSlidingWindowPatchAttr =
    "analog.sliding_window_patch";
static constexpr llvm::StringLiteral kRewrittenGroupedConv2DOutputAttr =
    "analog.rewritten_grouped_conv2d_output";

struct MatchedGroupedConv2D {
  linalg::Conv2DNgchwGfchwOp convOp;
  tensor::ExpandShapeOp activationExpandOp;
  tensor::ExpandShapeOp filterExpandOp;
  tensor::ExpandShapeOp outputExpandOp;
  linalg::BroadcastOp broadcastOp;
  Value sourceActivation;
  Value groupedActivation;
  arith::ConstantOp filterRank4Const;
  arith::ConstantOp filterRank2Const;
  Value bias;
  RankedTensorType sourceActivationTy;
  RankedTensorType groupedActivationTy;
  RankedTensorType filterRank4Ty;
  RankedTensorType groupedFilterTy;
  RankedTensorType filterRank2Ty;
  RankedTensorType biasBroadcastTy;
  RankedTensorType groupedOutputTy;
  SmallVector<int64_t> strides;
  int64_t n;
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

struct SlidingWindowLoweringState {
  Location loc;
  Type elementType;
  int64_t patchWidth;
  RankedTensorType patchTy;
  RankedTensorType matmulResultTy;
  RankedTensorType outputTy;
  Value c0;
  Value c1;
  Value ohUpper;
  Value owUpper;
  Value gUpper;
  Value cgUpper;
  Value khUpper;
  Value kwUpper;
  Value fUpper;
  Value fgUpper;
  Value strideH;
  Value strideW;
  Value channelPatchWidthValue;
  Value cgValue;
  Value kwValue;
  Value fgValue;
  Value zeroValue;
  Value transposedFilter;
  Value expandedBias;
};

static arith::ConstantOp findPreparedFlattenedFilter(
    arith::ConstantOp filterConst) {
  if (!filterConst || !filterConst->hasAttr(kDeleteInFuturePassAttr))
    return {};

  Operation *next = filterConst->getNextNode();
  auto flattenedConst = dyn_cast_or_null<arith::ConstantOp>(next);
  if (!flattenedConst)
    return {};

  auto flattenedTy = dyn_cast<RankedTensorType>(flattenedConst.getType());
  if (!flattenedTy || flattenedTy.getRank() != 2)
    return {};

  return flattenedConst;
}

static RankedTensorType buildFlattenedTensorType(int64_t patchWidth,
                                                 int64_t fTotal,
                                                 Type elementType) {
  return RankedTensorType::get({fTotal, patchWidth}, elementType);
}

static FailureOr<SmallVector<float>> getFilterValues(arith::ConstantOp op) {
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(op.getValue())) {
    SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (const APFloat &value : denseAttr.getValues<APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (isa<DenseResourceElementsAttr>(op.getValue())) {
    auto typedResourceAttr = dyn_cast<DenseF32ResourceElementsAttr>(op.getValue());
    if (!typedResourceAttr)
      return failure();
    std::optional<ArrayRef<float>> values = typedResourceAttr.tryGetAsArrayRef();
    if (!values)
      return failure();
    return SmallVector<float>(values->begin(), values->end());
  }

  return failure();
}

static TypedAttr buildBlockDiagonalFilterAttr(arith::ConstantOp filterConst,
                                              RankedTensorType flattenedTy,
                                              int64_t g, int64_t fg,
                                              int64_t cTotal, int64_t cg,
                                              int64_t kh, int64_t kw) {
  auto maybeValues = getFilterValues(filterConst);
  if (failed(maybeValues))
    return {};

  SmallVector<float> sourceValues = *maybeValues;
  SmallVector<float> flattenedValues(flattenedTy.getNumElements(), 0.0f);
  int64_t totalPatchWidth = cTotal * kh * kw;

  auto sourceIndex = [&](int64_t f, int64_t localC, int64_t khIdx,
                         int64_t kwIdx) {
    return (((f * cg + localC) * kh + khIdx) * kw + kwIdx);
  };

  auto destIndex = [&](int64_t f, int64_t channel, int64_t khIdx,
                       int64_t kwIdx) {
    return f * totalPatchWidth + ((channel * kh + khIdx) * kw + kwIdx);
  };

  for (int64_t group = 0; group < g; ++group) {
    for (int64_t fgIdx = 0; fgIdx < fg; ++fgIdx) {
      int64_t f = group * fg + fgIdx;
      for (int64_t cgIdx = 0; cgIdx < cg; ++cgIdx) {
        int64_t channel = group * cg + cgIdx;
        for (int64_t khIdx = 0; khIdx < kh; ++khIdx) {
          for (int64_t kwIdx = 0; kwIdx < kw; ++kwIdx) {
            flattenedValues[destIndex(f, channel, khIdx, kwIdx)] =
                sourceValues[sourceIndex(f, cgIdx, khIdx, kwIdx)];
          }
        }
      }
    }
  }

  auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
      ArrayRef<float>(flattenedValues), /*dataIsMutable=*/false);
  return llvm::cast<TypedAttr>(DenseF32ResourceElementsAttr::get(
      flattenedTy, "analog_grouped_conv2d_filter", std::move(blob)));
}

static FailureOr<arith::ConstantOp> getOrCreateFlattenedFilter(
    arith::ConstantOp filterConst, RankedTensorType filterRank4Ty, int64_t g,
    int64_t fg, int64_t cTotal, int64_t cg, int64_t kh, int64_t kw) {
  if (auto flattenedConst = findPreparedFlattenedFilter(filterConst))
    return flattenedConst;

  RankedTensorType flattenedTy =
      buildFlattenedTensorType(cTotal * kh * kw, g * fg,
                               filterRank4Ty.getElementType());
  TypedAttr flattenedAttr = buildBlockDiagonalFilterAttr(
      filterConst, flattenedTy, g, fg, cTotal, cg, kh, kw);
  if (!flattenedAttr)
    return failure();

  OpBuilder builder(filterConst);
  builder.setInsertionPointAfter(filterConst);
  auto flattenedConst =
      builder.create<arith::ConstantOp>(filterConst.getLoc(), flattenedTy,
                                        flattenedAttr);
  filterConst->setAttr(kDeleteInFuturePassAttr, builder.getUnitAttr());
  return flattenedConst;
}

static bool extractTwoPositiveInts(DenseIntElementsAttr attr,
                                   SmallVectorImpl<int64_t> &values) {
  values.clear();
  if (!attr)
    return false;
  for (APInt value : attr.getValues<APInt>())
    values.push_back(value.getSExtValue());
  return values.size() == 2 && values[0] > 0 && values[1] > 0;
}

static bool hasExpectedGroupedExpandShape(tensor::ExpandShapeOp expandOp,
                                          int64_t sourceRank,
                                          int64_t resultRank) {
  auto sourceTy = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
  auto resultTy = dyn_cast<RankedTensorType>(expandOp.getResult().getType());
  return sourceTy && resultTy && sourceTy.getRank() == sourceRank &&
         resultTy.getRank() == resultRank && sourceTy.hasStaticShape() &&
         resultTy.hasStaticShape();
}

static FailureOr<MatchedGroupedConv2D>
matchSupportedGroupedConv2D(linalg::Conv2DNgchwGfchwOp convOp) {
  if (convOp.getInputs().size() != 2 || convOp.getOutputs().size() != 1)
    return failure();

  auto activationExpandOp =
      convOp.getInputs()[0].getDefiningOp<tensor::ExpandShapeOp>();
  auto filterExpandOp =
      convOp.getInputs()[1].getDefiningOp<tensor::ExpandShapeOp>();
  auto outputExpandOp =
      convOp.getOutputs()[0].getDefiningOp<tensor::ExpandShapeOp>();
  if (!activationExpandOp || !filterExpandOp || !outputExpandOp)
    return failure();

  if (!hasExpectedGroupedExpandShape(activationExpandOp, 4, 5) ||
      !hasExpectedGroupedExpandShape(filterExpandOp, 4, 5) ||
      !hasExpectedGroupedExpandShape(outputExpandOp, 4, 5)) {
    return failure();
  }

  Value sourceActivation = activationExpandOp.getSrc();
  Value groupedActivation = activationExpandOp.getResult();
  auto sourceActivationTy = dyn_cast<RankedTensorType>(sourceActivation.getType());
  auto groupedActivationTy =
      dyn_cast<RankedTensorType>(activationExpandOp.getResult().getType());
  auto groupedOutputTy =
      dyn_cast<RankedTensorType>(convOp.getResult(0).getType());
  if (!sourceActivationTy || !groupedActivationTy || !groupedOutputTy ||
      !sourceActivationTy.getElementType().isF32() ||
      !groupedActivationTy.getElementType().isF32() ||
      !groupedOutputTy.getElementType().isF32()) {
    return failure();
  }

  auto filterRank4Const = filterExpandOp.getSrc().getDefiningOp<arith::ConstantOp>();
  if (!filterRank4Const)
    return failure();

  auto filterRank4Ty = dyn_cast<RankedTensorType>(filterRank4Const.getType());
  auto groupedFilterTy =
      dyn_cast<RankedTensorType>(filterExpandOp.getResult().getType());
  if (!filterRank4Ty || !groupedFilterTy || !filterRank4Ty.hasStaticShape() ||
      !groupedFilterTy.hasStaticShape() || filterRank4Ty.getRank() != 4 ||
      groupedFilterTy.getRank() != 5 ||
      !filterRank4Ty.getElementType().isF32() ||
      !groupedFilterTy.getElementType().isF32()) {
    return failure();
  }

  auto broadcastOp = outputExpandOp.getSrc().getDefiningOp<linalg::BroadcastOp>();
  if (!broadcastOp)
    return failure();
  auto biasTy = dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
  auto biasBroadcastTy =
      dyn_cast<RankedTensorType>(broadcastOp.getResult().front().getType());
  if (!biasTy || !biasBroadcastTy || !biasTy.hasStaticShape() ||
      !biasBroadcastTy.hasStaticShape() || biasTy.getRank() != 1 ||
      biasBroadcastTy.getRank() != 4 || !biasTy.getElementType().isF32() ||
      !biasBroadcastTy.getElementType().isF32()) {
    return failure();
  }

  auto biasDims = broadcastOp.getDimensions();
  if (biasDims.size() != 3 || biasDims[0] != 0 || biasDims[1] != 2 ||
      biasDims[2] != 3) {
    return failure();
  }

  SmallVector<int64_t> strides;
  if (!extractTwoPositiveInts(convOp.getStrides(), strides))
    return failure();
  SmallVector<int64_t> dilations;
  if (!extractTwoPositiveInts(convOp.getDilations(), dilations))
    return failure();
  if (dilations[0] != 1 || dilations[1] != 1)
    return failure();

  auto inShape = sourceActivationTy.getShape();
  auto groupedInShape = groupedActivationTy.getShape();
  auto filterShape = filterRank4Ty.getShape();
  auto groupedFilterShape = groupedFilterTy.getShape();
  auto biasBroadcastShape = biasBroadcastTy.getShape();
  auto groupedOutShape = groupedOutputTy.getShape();
  auto biasShape = biasTy.getShape();

  int64_t n = inShape[0];
  int64_t cTotal = inShape[1];
  int64_t h = inShape[2];
  int64_t w = inShape[3];
  int64_t g = groupedInShape[1];
  int64_t cg = groupedInShape[2];
  int64_t fTotal = filterShape[0];
  int64_t fg = groupedFilterShape[1];
  int64_t kh = filterShape[2];
  int64_t kw = filterShape[3];
  int64_t oh = groupedOutShape[3];
  int64_t ow = groupedOutShape[4];

  if (n != 1 || groupedInShape[0] != n || groupedInShape[3] != h ||
      groupedInShape[4] != w)
    return failure();
  if (cTotal != g * cg)
    return failure();
  if (groupedFilterShape[0] != g || groupedFilterShape[2] != cg ||
      groupedFilterShape[3] != kh || groupedFilterShape[4] != kw)
    return failure();
  if (fTotal != g * fg)
    return failure();
  if (filterShape[1] != cg)
    return failure();
  if (biasShape[0] != fTotal)
    return failure();
  if (biasBroadcastShape[0] != n || biasBroadcastShape[1] != fTotal ||
      biasBroadcastShape[2] != oh || biasBroadcastShape[3] != ow)
    return failure();
  if (groupedOutShape[0] != n || groupedOutShape[1] != g ||
      groupedOutShape[2] != fg)
    return failure();

  auto filterRank2Const = getOrCreateFlattenedFilter(filterRank4Const,
                                                     filterRank4Ty, g, fg,
                                                     cTotal, cg, kh, kw);
  if (failed(filterRank2Const))
    return failure();

  auto filterRank2Ty = dyn_cast<RankedTensorType>((*filterRank2Const).getType());
  if (!filterRank2Ty || filterRank2Ty.getShape()[0] != fTotal ||
      filterRank2Ty.getShape()[1] != cTotal * kh * kw)
    return failure();

  return MatchedGroupedConv2D{
      convOp, activationExpandOp, filterExpandOp, outputExpandOp,
      broadcastOp, sourceActivation, groupedActivation, filterRank4Const,
      *filterRank2Const, broadcastOp.getInput(), sourceActivationTy,
      groupedActivationTy, filterRank4Ty,
      groupedFilterTy, filterRank2Ty, biasBroadcastTy, groupedOutputTy, strides,
      n, g, cTotal, cg, h, w, fTotal, fg, kh, kw, oh, ow};
}

static Value buildZeroInitializedTensor(OpBuilder &builder, Location loc,
                                        RankedTensorType tensorTy,
                                        Value zeroValue) {
  Value empty = builder.create<tensor::EmptyOp>(loc, tensorTy.getShape(),
                                                tensorTy.getElementType());
  return builder.create<linalg::FillOp>(loc, ValueRange{zeroValue},
                                        ValueRange{empty})
      .getResult(0);
}

static Value buildTransposedFilter(OpBuilder &builder, MatchedGroupedConv2D &match,
                                   const SlidingWindowLoweringState &state) {
  Value init = builder.create<tensor::EmptyOp>(
      state.loc,
      ArrayRef<int64_t>{match.cTotal * match.kh * match.kw, match.fTotal},
      state.elementType);
  return builder
      .create<linalg::TransposeOp>(state.loc, match.filterRank2Const.getResult(),
                                   init, ArrayRef<int64_t>{1, 0})
      .getResult()
      .front();
}

static Value buildExpandedBias(OpBuilder &builder, MatchedGroupedConv2D &match,
                               const SlidingWindowLoweringState &state) {
  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}};
  return builder.create<tensor::ExpandShapeOp>(state.loc, state.matmulResultTy,
                                               match.bias, reassociation);
}

static SlidingWindowLoweringState buildSlidingWindowState(
    OpBuilder &builder, MatchedGroupedConv2D &match) {
  Location loc = match.convOp.getLoc();
  Type elementType = match.groupedOutputTy.getElementType();
  int64_t patchWidth = match.cTotal * match.kh * match.kw;
  auto patchTy = RankedTensorType::get({1, patchWidth}, elementType);
  auto matmulResultTy =
      RankedTensorType::get({1, match.fTotal}, elementType);

  SlidingWindowLoweringState state{
      loc,
      elementType,
      patchWidth,
      patchTy,
      matmulResultTy,
      match.groupedOutputTy,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      builder.create<arith::ConstantIndexOp>(loc, 1),
      builder.create<arith::ConstantIndexOp>(loc, match.oh),
      builder.create<arith::ConstantIndexOp>(loc, match.ow),
      builder.create<arith::ConstantIndexOp>(loc, match.g),
      builder.create<arith::ConstantIndexOp>(loc, match.cg),
      builder.create<arith::ConstantIndexOp>(loc, match.kh),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.fTotal),
      builder.create<arith::ConstantIndexOp>(loc, match.fg),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[0]),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[1]),
      builder.create<arith::ConstantIndexOp>(loc, match.kh * match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.cg),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.fg),
      builder.create<arith::ConstantFloatOp>(
          loc, llvm::cast<FloatType>(elementType), llvm::APFloat(0.0f)),
      Value{},
      Value{},
  };
  state.transposedFilter = buildTransposedFilter(builder, match, state);
  state.expandedBias = buildExpandedBias(builder, match, state);
  return state;
}

static Value buildFlattenedPatch(OpBuilder &builder, MatchedGroupedConv2D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value ohIdx, Value owIdx) {
  Value patchInit = builder.create<tensor::EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);
  auto groupLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.gUpper, state.c1, ValueRange{patchInit},
      [&](OpBuilder &groupBuilder, Location groupLoc, Value gIdx,
          ValueRange groupIterArgs) {
        auto channelLoop = groupBuilder.create<scf::ForOp>(
            groupLoc, state.c0, state.cgUpper, state.c1, groupIterArgs,
            [&](OpBuilder &channelBuilder, Location channelLoc, Value cgIdx,
                ValueRange channelIterArgs) {
              auto khLoop = channelBuilder.create<scf::ForOp>(
                  channelLoc, state.c0, state.khUpper, state.c1, channelIterArgs,
                  [&](OpBuilder &khBuilder, Location khLoc, Value khIdx,
                      ValueRange khIterArgs) {
                    auto kwLoop = khBuilder.create<scf::ForOp>(
                        khLoc, state.c0, state.kwUpper, state.c1, khIterArgs,
                        [&](OpBuilder &kwBuilder, Location kwLoc, Value kwIdx,
                            ValueRange kwIterArgs) {
                          Value ihBase = kwBuilder.create<arith::MulIOp>(
                              kwLoc, ohIdx, state.strideH);
                          Value iwBase = kwBuilder.create<arith::MulIOp>(
                              kwLoc, owIdx, state.strideW);
                          Value ih = kwBuilder.create<arith::AddIOp>(
                              kwLoc, ihBase, khIdx);
                          Value iw = kwBuilder.create<arith::AddIOp>(
                              kwLoc, iwBase, kwIdx);
                          Value inputValue = kwBuilder.create<tensor::ExtractOp>(
                              kwLoc, match.groupedActivation,
                              ValueRange{state.c0, gIdx, cgIdx, ih, iw});
                          Value groupBase = kwBuilder.create<arith::MulIOp>(
                              kwLoc, gIdx, state.cgValue);
                          Value flatChannel = kwBuilder.create<arith::AddIOp>(
                              kwLoc, groupBase, cgIdx);
                          Value channelOffset = kwBuilder.create<arith::MulIOp>(
                              kwLoc, flatChannel, state.channelPatchWidthValue);
                          Value khOffset = kwBuilder.create<arith::MulIOp>(
                              kwLoc, khIdx, state.kwValue);
                          Value patchOffset = kwBuilder.create<arith::AddIOp>(
                              kwLoc, channelOffset, khOffset);
                          Value flatIndex = kwBuilder.create<arith::AddIOp>(
                              kwLoc, patchOffset, kwIdx);
                          Value updatedPatch = kwBuilder.create<tensor::InsertOp>(
                              kwLoc, inputValue, kwIterArgs[0],
                              ValueRange{state.c0, flatIndex});
                          kwBuilder.create<scf::YieldOp>(kwLoc, updatedPatch);
                        });
                    khBuilder.create<scf::YieldOp>(khLoc, kwLoop.getResult(0));
                  });
              channelBuilder.create<scf::YieldOp>(channelLoc, khLoop.getResult(0));
            });
        groupBuilder.create<scf::YieldOp>(groupLoc, channelLoop.getResult(0));
      });
  groupLoop->setAttr(kSlidingWindowPatchAttr, builder.getUnitAttr());
  return groupLoop.getResult(0);
}

static Value buildPatchMatmul(OpBuilder &builder, MatchedGroupedConv2D &match,
                              const SlidingWindowLoweringState &state,
                              Value patch) {
  Value init = buildZeroInitializedTensor(builder, state.loc,
                                          state.matmulResultTy, state.zeroValue);
  auto matmulOp = builder.create<linalg::MatmulOp>(
      state.loc, state.matmulResultTy, ValueRange{patch, state.transposedFilter},
      ValueRange{init});
  matmulOp->setAttr(kSlidingWindowMatmulAttr, builder.getUnitAttr());
  if (auto matrixSourceId = match.filterRank2Const->getAttr(kMatrixSourceIdAttr))
    matmulOp->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  return matmulOp.getResult(0);
}

static Value addBiasToChannelResult(OpBuilder &builder,
                                    const SlidingWindowLoweringState &state,
                                    Value channelResult) {
  Value biasedInitEmpty = builder.create<tensor::EmptyOp>(
      state.loc, state.matmulResultTy.getShape(), state.elementType);
  auto biasedResult = builder.create<linalg::AddOp>(
      state.loc, ValueRange{channelResult, state.expandedBias},
      ValueRange{biasedInitEmpty});
  biasedResult->setAttr(kSlidingWindowBiasAddAttr, builder.getUnitAttr());
  return biasedResult.getResultTensors().front();
}

static Value assembleOutputChannels(OpBuilder &builder,
                                    MatchedGroupedConv2D &match,
                                    const SlidingWindowLoweringState &state,
                                    Value biasedResult, Value ohIdx, Value owIdx,
                                    Value currentOutput) {
  auto channelAssembleLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.fUpper, state.c1, ValueRange{currentOutput},
      [&](OpBuilder &fBuilder, Location fLoc, Value fIdx, ValueRange fIterArgs) {
        Value channelValue = fBuilder.create<tensor::ExtractOp>(
            fLoc, biasedResult, ValueRange{state.c0, fIdx});
        Value groupIdx = fBuilder.create<arith::DivUIOp>(fLoc, fIdx, state.fgValue);
        Value fgIdx = fBuilder.create<arith::RemUIOp>(fLoc, fIdx, state.fgValue);
        Value updatedOutput = fBuilder.create<tensor::InsertOp>(
            fLoc, channelValue, fIterArgs[0],
            ValueRange{state.c0, groupIdx, fgIdx, ohIdx, owIdx});
        fBuilder.create<scf::YieldOp>(fLoc, updatedOutput);
      });
  channelAssembleLoop->setAttr(kOutputChannelAssemblyAttr, builder.getUnitAttr());
  return channelAssembleLoop.getResult(0);
}

static Value emitSlidingWindowIR(MatchedGroupedConv2D &match) {
  Operation *insertionPoint = match.convOp.getOperation();
  OpBuilder builder(insertionPoint->getContext());
  builder.setInsertionPointAfter(insertionPoint);
  SlidingWindowLoweringState state = buildSlidingWindowState(builder, match);
  Value rewrittenOutputInit = buildZeroInitializedTensor(
      builder, state.loc, state.outputTy, state.zeroValue);

  auto ohLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.ohUpper, state.c1,
      ValueRange{rewrittenOutputInit},
      [&](OpBuilder &ohBuilder, Location ohLoc, Value ohIdx,
          ValueRange ohIterArgs) {
        auto owLoop = ohBuilder.create<scf::ForOp>(
            ohLoc, state.c0, state.owUpper, state.c1, ohIterArgs,
            [&](OpBuilder &owBuilder, Location owLoc, Value owIdx,
                ValueRange owIterArgs) {
              Value patch = buildFlattenedPatch(owBuilder, match, state, ohIdx,
                                               owIdx);
              Value matmulResult =
                  buildPatchMatmul(owBuilder, match, state, patch);
              Value biasedResult =
                  addBiasToChannelResult(owBuilder, state, matmulResult);
              Value updatedOutput = assembleOutputChannels(
                  owBuilder, match, state, biasedResult, ohIdx, owIdx,
                  owIterArgs[0]);
              owBuilder.create<scf::YieldOp>(owLoc, updatedOutput);
            });
        ohBuilder.create<scf::YieldOp>(ohLoc, owLoop.getResult(0));
      });
  ohLoop->setAttr(kRewrittenGroupedConv2DOutputAttr, builder.getUnitAttr());
  return ohLoop.getResult(0);
}

} // namespace

llvm::StringRef RewriteGroupedConv2DToMatmulPass::getArgument() const {
  return "analog-rewrite-grouped-conv2d-to-matmul";
}

llvm::StringRef RewriteGroupedConv2DToMatmulPass::getDescription() const {
  return "Rewrite supported grouped conv2d ops into a matmul-oriented form";
}

void RewriteGroupedConv2DToMatmulPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}

void RewriteGroupedConv2DToMatmulPass::runOnOperation() {
  auto func = getOperation();
  int64_t nextMatrixSourceId = 0;

  func.walk([&](arith::ConstantOp op) {
    if (auto matrixSourceId = op->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr))
      nextMatrixSourceId = std::max(nextMatrixSourceId, matrixSourceId.getInt() + 1);
  });

  func.walk([&](linalg::Conv2DNgchwGfchwOp convOp) {
    auto maybeMatch = matchSupportedGroupedConv2D(convOp);
    if (failed(maybeMatch))
      return;

    MatchedGroupedConv2D match = *maybeMatch;
    IntegerAttr matrixSourceId =
        match.filterRank2Const->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId) {
      matrixSourceId = IntegerAttr::get(
          IntegerType::get(func.getContext(), 64), nextMatrixSourceId++);
      match.filterRank2Const->setAttr(kMatrixSourceIdAttr, matrixSourceId);
    }

    Value rewrittenOutput = emitSlidingWindowIR(match);
    match.convOp.getResult(0).replaceAllUsesWith(rewrittenOutput);

    auto eraseIfUnused = [](Operation *op) {
      if (op && op->use_empty())
        op->erase();
    };
    eraseIfUnused(match.convOp.getOperation());
    eraseIfUnused(match.outputExpandOp.getOperation());
    eraseIfUnused(match.broadcastOp.getOperation());
    eraseIfUnused(match.filterExpandOp.getOperation());
    eraseIfUnused(match.activationExpandOp.getOperation());
    if (match.filterRank4Const->hasAttr(kDeleteInFuturePassAttr) &&
        match.filterRank4Const->use_empty()) {
      match.filterRank4Const->erase();
    }
  });
}

std::unique_ptr<mlir::Pass> createRewriteGroupedConv2DToMatmulPass() {
  return std::make_unique<RewriteGroupedConv2DToMatmulPass>();
}

} // namespace analog
} // namespace mlir
