#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv3DToMatmul.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
static constexpr llvm::StringLiteral kRewrittenConvOutputAttr =
    "analog.rewritten_conv3d_output";

struct MatchedConv3D {
  linalg::Conv3DNcdhwFcdhwOp convOp;
  linalg::BroadcastOp broadcastOp;
  Value activation;
  arith::ConstantOp filterRank5Const;
  arith::ConstantOp filterRank2Const;
  Value bias;
  Value outputInit;
  RankedTensorType inputTy;
  RankedTensorType filterRank5Ty;
  RankedTensorType filterRank2Ty;
  RankedTensorType outputTy;
  SmallVector<int64_t> strides;
  int64_t n;
  int64_t c;
  int64_t d;
  int64_t h;
  int64_t w;
  int64_t f;
  int64_t kd;
  int64_t kh;
  int64_t kw;
  int64_t od;
  int64_t oh;
  int64_t ow;
};

struct SlidingWindowLoweringState {
  Location loc;
  Type elementType;
  int64_t patchVolume;
  RankedTensorType patchTy;
  RankedTensorType matmulResultTy;
  RankedTensorType outputTy;
  Value c0;
  Value c1;
  Value odUpper;
  Value ohUpper;
  Value owUpper;
  Value cUpper;
  Value kdUpper;
  Value khUpper;
  Value kwUpper;
  Value fUpper;
  Value patchVolumeValue;
  Value strideD;
  Value strideH;
  Value strideW;
  Value khkwValue;
  Value kwValue;
  Value zeroValue;
  Value transposedFilter;
  Value expandedBias;
};

struct ConvTensorShapeInfo {
  int64_t n;
  int64_t c;
  int64_t d;
  int64_t h;
  int64_t w;
  int64_t f;
  int64_t kd;
  int64_t kh;
  int64_t kw;
  int64_t od;
  int64_t oh;
  int64_t ow;
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

static RankedTensorType buildFlattenedTensorType(RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  int64_t flattenedCols = shape[1] * shape[2] * shape[3] * shape[4];
  return RankedTensorType::get({shape[0], flattenedCols},
                               tensorTy.getElementType());
}

static TypedAttr buildFlattenedAttr(arith::ConstantOp op,
                                    RankedTensorType flattenedTy) {
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue()))
    return denseAttr.reshape(flattenedTy);

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(op.getValue()))
    return DenseResourceElementsAttr::get(flattenedTy,
                                          resourceAttr.getRawHandle());

  return {};
}

static FailureOr<arith::ConstantOp> getOrCreateFlattenedFilter(
    arith::ConstantOp filterConst, RankedTensorType filterRank5Ty) {
  if (auto flattenedConst = findPreparedFlattenedFilter(filterConst))
    return flattenedConst;

  RankedTensorType flattenedTy = buildFlattenedTensorType(filterRank5Ty);
  TypedAttr flattenedAttr = buildFlattenedAttr(filterConst, flattenedTy);
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

static bool extractPositiveInts(DenseIntElementsAttr attr, size_t expectedSize,
                                SmallVectorImpl<int64_t> &values) {
  values.clear();
  if (!attr)
    return false;

  for (APInt value : attr.getValues<APInt>())
    values.push_back(value.getSExtValue());

  if (values.size() != expectedSize)
    return false;

  return llvm::all_of(values, [](int64_t value) { return value > 0; });
}

static FailureOr<std::pair<RankedTensorType, RankedTensorType>>
getSupportedInputAndOutputTypes(linalg::Conv3DNcdhwFcdhwOp convOp,
                                Value activation) {
  auto inputTy = dyn_cast<RankedTensorType>(activation.getType());
  auto outputTy = dyn_cast<RankedTensorType>(convOp.getResult(0).getType());
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape()) {
    return failure();
  }
  if (inputTy.getRank() != 5 || outputTy.getRank() != 5)
    return failure();
  if (!inputTy.getElementType().isF32() || !outputTy.getElementType().isF32())
    return failure();

  return std::make_pair(inputTy, outputTy);
}

static FailureOr<std::pair<linalg::BroadcastOp, RankedTensorType>>
getSupportedBiasBroadcast(Value outputInit, RankedTensorType outputTy) {
  auto broadcastOp = outputInit.getDefiningOp<linalg::BroadcastOp>();
  if (!broadcastOp)
    return failure();

  auto biasTy = dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
  auto broadcastInitTy =
      dyn_cast<RankedTensorType>(broadcastOp.getInit().getType());
  if (!biasTy || !broadcastInitTy || !biasTy.hasStaticShape() ||
      !broadcastInitTy.hasStaticShape()) {
    return failure();
  }
  if (biasTy.getRank() != 1 || broadcastInitTy.getRank() != 5)
    return failure();
  if (broadcastInitTy != outputTy)
    return failure();

  auto dims = broadcastOp.getDimensions();
  if (dims.size() != 4 || dims[0] != 0 || dims[1] != 2 || dims[2] != 3 ||
      dims[3] != 4) {
    return failure();
  }

  return std::make_pair(broadcastOp, biasTy);
}

static FailureOr<std::tuple<arith::ConstantOp, RankedTensorType,
                            arith::ConstantOp, RankedTensorType>>
getSupportedFilterConstants(Value filter) {
  auto filterRank5Const = filter.getDefiningOp<arith::ConstantOp>();
  if (!filterRank5Const)
    return failure();

  auto filterRank5Ty = dyn_cast<RankedTensorType>(filterRank5Const.getType());
  if (!filterRank5Ty || !filterRank5Ty.hasStaticShape() ||
      filterRank5Ty.getRank() != 5) {
    return failure();
  }
  if (!filterRank5Ty.getElementType().isF32())
    return failure();

  auto filterRank2Const = getOrCreateFlattenedFilter(filterRank5Const,
                                                     filterRank5Ty);
  if (failed(filterRank2Const))
    return failure();

  auto filterRank2Ty = dyn_cast<RankedTensorType>((*filterRank2Const).getType());
  if (!filterRank2Ty || !filterRank2Ty.hasStaticShape() ||
      filterRank2Ty.getRank() != 2) {
    return failure();
  }
  if (!filterRank2Ty.getElementType().isF32())
    return failure();

  return std::make_tuple(filterRank5Const, filterRank5Ty, *filterRank2Const,
                         filterRank2Ty);
}

static FailureOr<SmallVector<int64_t>> getSupportedStrides(
    linalg::Conv3DNcdhwFcdhwOp convOp) {
  SmallVector<int64_t> dilations;
  if (!extractPositiveInts(convOp.getDilations(), 3, dilations))
    return failure();
  if (dilations[0] != 1 || dilations[1] != 1 || dilations[2] != 1)
    return failure();

  SmallVector<int64_t> strides;
  if (!extractPositiveInts(convOp.getStrides(), 3, strides))
    return failure();

  return strides;
}

static FailureOr<ConvTensorShapeInfo> getValidatedShapeInfo(
    RankedTensorType inputTy, RankedTensorType biasTy,
    RankedTensorType filterRank5Ty, RankedTensorType filterRank2Ty,
    RankedTensorType outputTy) {
  auto inputShape = inputTy.getShape();
  auto filterShape = filterRank5Ty.getShape();
  auto filterFlatShape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();
  auto biasShape = biasTy.getShape();

  ConvTensorShapeInfo shapeInfo{
      inputShape[0],  inputShape[1],  inputShape[2],  inputShape[3],
      inputShape[4],  filterShape[0], filterShape[2], filterShape[3],
      filterShape[4], outputShape[2], outputShape[3], outputShape[4],
  };
  int64_t filterChannels = filterShape[1];
  int64_t outN = outputShape[0];
  int64_t outF = outputShape[1];

  if (shapeInfo.n != 1)
    return failure();
  if (filterChannels != shapeInfo.c)
    return failure();
  if (filterFlatShape[0] != shapeInfo.f ||
      filterFlatShape[1] !=
          shapeInfo.c * shapeInfo.kd * shapeInfo.kh * shapeInfo.kw) {
    return failure();
  }
  if (biasShape[0] != shapeInfo.f)
    return failure();
  if (outN != shapeInfo.n || outF != shapeInfo.f)
    return failure();

  return shapeInfo;
}

static FailureOr<MatchedConv3D>
matchSupportedConv3D(linalg::Conv3DNcdhwFcdhwOp convOp) {
  if (convOp.getInputs().size() != 2 || convOp.getOutputs().size() != 1)
    return failure();

  Value activation = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value outputInit = convOp.getOutputs()[0];

  auto inputOutputTypes = getSupportedInputAndOutputTypes(convOp, activation);
  if (failed(inputOutputTypes))
    return failure();
  auto [inputTy, outputTy] = *inputOutputTypes;

  auto biasBroadcast = getSupportedBiasBroadcast(outputInit, outputTy);
  if (failed(biasBroadcast))
    return failure();
  auto [broadcastOp, biasTy] = *biasBroadcast;

  auto filterConstants = getSupportedFilterConstants(filter);
  if (failed(filterConstants))
    return failure();
  auto [filterRank5Const, filterRank5Ty, filterRank2Const, filterRank2Ty] =
      *filterConstants;

  auto strides = getSupportedStrides(convOp);
  if (failed(strides))
    return failure();

  auto shapeInfo = getValidatedShapeInfo(inputTy, biasTy, filterRank5Ty,
                                         filterRank2Ty, outputTy);
  if (failed(shapeInfo))
    return failure();

  return MatchedConv3D{
      convOp,
      broadcastOp,
      activation,
      filterRank5Const,
      filterRank2Const,
      broadcastOp.getInput(),
      outputInit,
      inputTy,
      filterRank5Ty,
      filterRank2Ty,
      outputTy,
      *strides,
      shapeInfo->n,
      shapeInfo->c,
      shapeInfo->d,
      shapeInfo->h,
      shapeInfo->w,
      shapeInfo->f,
      shapeInfo->kd,
      shapeInfo->kh,
      shapeInfo->kw,
      shapeInfo->od,
      shapeInfo->oh,
      shapeInfo->ow,
  };
}

static Value buildTransposedFilter(OpBuilder &builder, MatchedConv3D &match,
                                   const SlidingWindowLoweringState &state) {
  Value transposedFilterInit = builder.create<tensor::EmptyOp>(
      state.loc, ArrayRef<int64_t>{match.c * state.patchVolume, match.f},
      state.elementType);
  return builder
      .create<linalg::TransposeOp>(state.loc, match.filterRank2Const.getResult(),
                                   transposedFilterInit,
                                   ArrayRef<int64_t>{1, 0})
      .getResult()
      .front();
}

static Value buildExpandedBias(OpBuilder &builder, MatchedConv3D &match,
                               const SlidingWindowLoweringState &state) {
  SmallVector<ReassociationIndices, 2> biasExpandReassociation = {{0, 1}};
  return builder.create<tensor::ExpandShapeOp>(state.loc, state.matmulResultTy,
                                               match.bias,
                                               biasExpandReassociation);
}

static Value buildZeroInitializedTensor(OpBuilder &builder, Location loc,
                                        RankedTensorType tensorTy,
                                        Value zeroValue) {
  Value empty = builder.create<tensor::EmptyOp>(
      loc, tensorTy.getShape(), tensorTy.getElementType());
  return builder.create<linalg::FillOp>(loc, ValueRange{zeroValue},
                                        ValueRange{empty})
      .getResult(0);
}

static SlidingWindowLoweringState buildSlidingWindowState(
    OpBuilder &builder, MatchedConv3D &match) {
  Location loc = match.convOp.getLoc();
  Type elementType = match.inputTy.getElementType();
  int64_t patchVolume = match.kd * match.kh * match.kw;
  auto patchTy =
      RankedTensorType::get({1, match.c * patchVolume}, elementType);
  auto matmulResultTy = RankedTensorType::get({1, match.f}, elementType);

  SlidingWindowLoweringState state{
      loc,
      elementType,
      patchVolume,
      patchTy,
      matmulResultTy,
      match.outputTy,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      builder.create<arith::ConstantIndexOp>(loc, 1),
      builder.create<arith::ConstantIndexOp>(loc, match.od),
      builder.create<arith::ConstantIndexOp>(loc, match.oh),
      builder.create<arith::ConstantIndexOp>(loc, match.ow),
      builder.create<arith::ConstantIndexOp>(loc, match.c),
      builder.create<arith::ConstantIndexOp>(loc, match.kd),
      builder.create<arith::ConstantIndexOp>(loc, match.kh),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.f),
      builder.create<arith::ConstantIndexOp>(loc, patchVolume),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[0]),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[1]),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[2]),
      builder.create<arith::ConstantIndexOp>(loc, match.kh * match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(elementType), llvm::APFloat(0.0f)),
      Value{},
      Value{},
  };
  state.transposedFilter = buildTransposedFilter(builder, match, state);
  state.expandedBias = buildExpandedBias(builder, match, state);
  return state;
}

static Value buildFlattenedPatch(OpBuilder &builder, MatchedConv3D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value odIdx, Value ohIdx, Value owIdx) {
  Value patchInit = builder.create<tensor::EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);
  auto channelLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.cUpper, state.c1, ValueRange{patchInit},
      [&](OpBuilder &channelBuilder, Location channelLoc, Value cIdx,
          ValueRange channelIterArgs) {
        auto kdLoop = channelBuilder.create<scf::ForOp>(
            channelLoc, state.c0, state.kdUpper, state.c1, channelIterArgs,
            [&](OpBuilder &kdBuilder, Location kdLoc, Value kdIdx,
                ValueRange kdIterArgs) {
              auto khLoop = kdBuilder.create<scf::ForOp>(
                  kdLoc, state.c0, state.khUpper, state.c1, kdIterArgs,
                  [&](OpBuilder &khBuilder, Location khLoc, Value khIdx,
                      ValueRange khIterArgs) {
                    auto kwLoop = khBuilder.create<scf::ForOp>(
                        khLoc, state.c0, state.kwUpper, state.c1, khIterArgs,
                        [&](OpBuilder &kwBuilder, Location kwLoc, Value kwIdx,
                            ValueRange kwIterArgs) {
                          Value idBase = kwBuilder.create<arith::MulIOp>(
                              kwLoc, odIdx, state.strideD);
                          Value ihBase = kwBuilder.create<arith::MulIOp>(
                              kwLoc, ohIdx, state.strideH);
                          Value iwBase = kwBuilder.create<arith::MulIOp>(
                              kwLoc, owIdx, state.strideW);
                          Value id =
                              kwBuilder.create<arith::AddIOp>(kwLoc, idBase, kdIdx);
                          Value ih =
                              kwBuilder.create<arith::AddIOp>(kwLoc, ihBase, khIdx);
                          Value iw =
                              kwBuilder.create<arith::AddIOp>(kwLoc, iwBase, kwIdx);
                          Value inputValue = kwBuilder.create<tensor::ExtractOp>(
                              kwLoc, match.activation,
                              ValueRange{state.c0, cIdx, id, ih, iw});
                          Value channelOffset = kwBuilder.create<arith::MulIOp>(
                              kwLoc, cIdx, state.patchVolumeValue);
                          Value kdOffset = kwBuilder.create<arith::MulIOp>(
                              kwLoc, kdIdx, state.khkwValue);
                          Value khOffset = kwBuilder.create<arith::MulIOp>(
                              kwLoc, khIdx, state.kwValue);
                          Value depthOffset = kwBuilder.create<arith::AddIOp>(
                              kwLoc, channelOffset, kdOffset);
                          Value patchOffset = kwBuilder.create<arith::AddIOp>(
                              kwLoc, depthOffset, khOffset);
                          Value flatIndex = kwBuilder.create<arith::AddIOp>(
                              kwLoc, patchOffset, kwIdx);
                          Value updatedPatch = kwBuilder.create<tensor::InsertOp>(
                              kwLoc, inputValue, kwIterArgs[0],
                              ValueRange{state.c0, flatIndex});
                          kwBuilder.create<scf::YieldOp>(kwLoc, updatedPatch);
                        });
                    khBuilder.create<scf::YieldOp>(khLoc, kwLoop.getResult(0));
                  });
              kdBuilder.create<scf::YieldOp>(kdLoc, khLoop.getResult(0));
            });
        channelBuilder.create<scf::YieldOp>(channelLoc, kdLoop.getResult(0));
      });
  channelLoop->setAttr(kSlidingWindowPatchAttr, builder.getUnitAttr());
  return channelLoop.getResult(0);
}

static Value buildPatchMatmul(OpBuilder &builder, MatchedConv3D &match,
                              const SlidingWindowLoweringState &state,
                              Value patch) {
  Value matmulInit = buildZeroInitializedTensor(builder, state.loc,
                                                state.matmulResultTy,
                                                state.zeroValue);
  auto matmulOp = builder.create<linalg::MatmulOp>(
      state.loc, state.matmulResultTy, ValueRange{patch, state.transposedFilter},
      ValueRange{matmulInit});
  matmulOp->setAttr(kSlidingWindowMatmulAttr, builder.getUnitAttr());
  if (auto matrixSourceId =
          match.filterRank2Const->getAttr(kMatrixSourceIdAttr)) {
    matmulOp->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  }
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
                                    const SlidingWindowLoweringState &state,
                                    Value biasedResult, Value odIdx, Value ohIdx,
                                    Value owIdx, Value currentOutput) {
  auto channelAssembleLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.fUpper, state.c1, ValueRange{currentOutput},
      [&](OpBuilder &fBuilder, Location fLoc, Value fIdx, ValueRange fIterArgs) {
        Value channelValue = fBuilder.create<tensor::ExtractOp>(
            fLoc, biasedResult, ValueRange{state.c0, fIdx});
        Value updatedOutput = fBuilder.create<tensor::InsertOp>(
            fLoc, channelValue, fIterArgs[0],
            ValueRange{state.c0, fIdx, odIdx, ohIdx, owIdx});
        fBuilder.create<scf::YieldOp>(fLoc, updatedOutput);
      });
  channelAssembleLoop->setAttr(kOutputChannelAssemblyAttr, builder.getUnitAttr());
  return channelAssembleLoop.getResult(0);
}

static Value lowerOutputPosition(OpBuilder &builder, MatchedConv3D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value odIdx, Value ohIdx, Value owIdx,
                                 Value currentOutput) {
  Value patch = buildFlattenedPatch(builder, match, state, odIdx, ohIdx, owIdx);
  Value matmulResult = buildPatchMatmul(builder, match, state, patch);
  Value biasedResult = addBiasToChannelResult(builder, state, matmulResult);
  return assembleOutputChannels(builder, state, biasedResult, odIdx, ohIdx,
                                owIdx, currentOutput);
}

static Value emitSlidingWindowIR(MatchedConv3D &match) {
  Operation *insertionPoint = match.convOp.getOperation();
  OpBuilder builder(insertionPoint->getContext());
  builder.setInsertionPointAfter(insertionPoint);
  SlidingWindowLoweringState state = buildSlidingWindowState(builder, match);
  Value rewrittenOutputInit =
      buildZeroInitializedTensor(builder, state.loc, state.outputTy,
                                 state.zeroValue);

  auto odLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.odUpper, state.c1,
      ValueRange{rewrittenOutputInit},
      [&](OpBuilder &odBuilder, Location odLoc, Value odIdx,
          ValueRange odIterArgs) {
        auto ohLoop = odBuilder.create<scf::ForOp>(
            odLoc, state.c0, state.ohUpper, state.c1, odIterArgs,
            [&](OpBuilder &ohBuilder, Location ohLoc, Value ohIdx,
                ValueRange ohIterArgs) {
              auto owLoop = ohBuilder.create<scf::ForOp>(
                  ohLoc, state.c0, state.owUpper, state.c1, ohIterArgs,
                  [&](OpBuilder &owBuilder, Location owLoc, Value owIdx,
                      ValueRange owIterArgs) {
                    Value updatedOutput = lowerOutputPosition(
                        owBuilder, match, state, odIdx, ohIdx, owIdx,
                        owIterArgs[0]);
                    owBuilder.create<scf::YieldOp>(owLoc, updatedOutput);
                  });
              ohBuilder.create<scf::YieldOp>(ohLoc, owLoop.getResult(0));
            });
        odBuilder.create<scf::YieldOp>(odLoc, ohLoop.getResult(0));
      });
  odLoop->setAttr(kRewrittenConvOutputAttr, builder.getUnitAttr());
  return odLoop.getResult(0);
}

} // namespace

llvm::StringRef RewriteConv3DToMatmulPass::getArgument() const {
  return "analog-rewrite-conv3d-to-matmul";
}

llvm::StringRef RewriteConv3DToMatmulPass::getDescription() const {
  return "Rewrite supported conv3d ops into a matmul-oriented form";
}

void RewriteConv3DToMatmulPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}

void RewriteConv3DToMatmulPass::runOnOperation() {
  auto func = getOperation();
  int64_t nextMatrixSourceId = 0;

  func.walk([&](linalg::Conv3DNcdhwFcdhwOp convOp) {
    FailureOr<MatchedConv3D> maybeMatch = matchSupportedConv3D(convOp);
    if (failed(maybeMatch))
      return;

    MatchedConv3D match = *maybeMatch;

    IntegerAttr matrixSourceId =
        match.filterRank2Const->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId) {
      matrixSourceId = IntegerAttr::get(
          IntegerType::get(func.getContext(), 64), nextMatrixSourceId++);
      match.filterRank2Const->setAttr(kMatrixSourceIdAttr, matrixSourceId);
    }

    Value rewrittenOutput = emitSlidingWindowIR(match);
    match.convOp.getResult(0).replaceAllUsesWith(rewrittenOutput);

    Operation *conv = match.convOp.getOperation();
    Operation *broadcast = match.broadcastOp.getOperation();
    Operation *filterRank5 = match.filterRank5Const.getOperation();
    if (conv->use_empty())
      conv->erase();
    if (broadcast->use_empty())
      broadcast->erase();
    if (filterRank5->hasAttr(kDeleteInFuturePassAttr) &&
        filterRank5->use_empty()) {
      filterRank5->erase();
    }
  });
}

std::unique_ptr<mlir::Pass> createRewriteConv3DToMatmulPass() {
  return std::make_unique<RewriteConv3DToMatmulPass>();
}

} // namespace analog
} // namespace mlir
