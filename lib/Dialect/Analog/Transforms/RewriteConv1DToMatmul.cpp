#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv1DToMatmul.h"

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
    "analog.rewritten_conv2d_output";

struct MatchedConv1D {
  linalg::Conv1DNcwFcwOp convOp;
  linalg::BroadcastOp broadcastOp;
  Value activation;
  arith::ConstantOp filterRank3Const;
  arith::ConstantOp filterRank2Const;
  Value bias;
  Value outputInit;
  RankedTensorType inputTy;
  RankedTensorType filterRank3Ty;
  RankedTensorType filterRank2Ty;
  RankedTensorType outputTy;
  int64_t stride;
  int64_t n;
  int64_t c;
  int64_t w;
  int64_t f;
  int64_t kw;
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
  Value owUpper;
  Value cUpper;
  Value kwUpper;
  Value fUpper;
  Value patchWidthValue;
  Value strideW;
  Value zeroValue;
  Value transposedFilter;
  Value expandedBias;
};

struct ConvTensorShapeInfo {
  int64_t n;
  int64_t c;
  int64_t w;
  int64_t f;
  int64_t kw;
  int64_t ow;
};


// Finds the flattened rank-2 filter constant when one has already been
// materialized next to the original rank-3 filter constant.

static arith::ConstantOp findPreparedFlattenedFilter(
    arith::ConstantOp filterConst) {
  if (!filterConst)
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


// Computes the rank-2 tensor type produced by flattening `[F, C, KW]`
// filter constants into `[F, C * KW]`.

static RankedTensorType buildFlattenedTensorType(RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  int64_t flattenedCols = shape[1] * shape[2];
  return RankedTensorType::get({shape[0], flattenedCols},
                               tensorTy.getElementType());
}


// Rebuilds the filter constant payload with the flattened type while
// preserving dense or resource-backed storage.

static TypedAttr buildFlattenedAttr(arith::ConstantOp op,
                                    RankedTensorType flattenedTy) {
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue()))
    return denseAttr.reshape(flattenedTy);

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(op.getValue())) {
    return DenseResourceElementsAttr::get(flattenedTy,
                                          resourceAttr.getRawHandle());
  }

  return {};
}


// Creates the flattened rank-2 filter constant on demand when no
// prepare pass has already materialized it.

static FailureOr<arith::ConstantOp> getOrCreateFlattenedFilter(
    arith::ConstantOp filterConst, RankedTensorType filterRank3Ty) {
  if (auto flattenedConst = findPreparedFlattenedFilter(filterConst))
    return flattenedConst;

  RankedTensorType flattenedTy = buildFlattenedTensorType(filterRank3Ty);
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


// Reads a fixed-size integer attribute into a vector and rejects
// missing, malformed, or non-positive values.

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


// Validates the activation and output tensors expected by the conv
// rewrite and returns their ranked tensor types.

static FailureOr<std::pair<RankedTensorType, RankedTensorType>>
getSupportedInputAndOutputTypes(linalg::Conv1DNcwFcwOp convOp,
                                Value activation) {
  auto inputTy = dyn_cast<RankedTensorType>(activation.getType());
  auto outputTy = dyn_cast<RankedTensorType>(convOp.getResult(0).getType());
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape()) {
    return failure();
  }
  if (inputTy.getRank() != 3 || outputTy.getRank() != 3)
    return failure();
  if (!inputTy.getElementType().isF32() || !outputTy.getElementType().isF32())
    return failure();

  return std::make_pair(inputTy, outputTy);
}


// Validates the broadcasted bias initializer and returns the broadcast
// op together with the 1-D bias tensor type.

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
  if (biasTy.getRank() != 1 || broadcastInitTy.getRank() != 3)
    return failure();
  if (broadcastInitTy != outputTy)
    return failure();

  auto dims = broadcastOp.getDimensions();
  if (dims.size() != 2 || dims[0] != 0 || dims[1] != 2)
    return failure();

  return std::make_pair(broadcastOp, biasTy);
}


// Validates the original and flattened filter constants and returns
// both constants together with their tensor types.

static FailureOr<std::tuple<arith::ConstantOp, RankedTensorType,
                            arith::ConstantOp, RankedTensorType>>
getSupportedFilterConstants(Value filter) {
  auto filterRank3Const = filter.getDefiningOp<arith::ConstantOp>();
  if (!filterRank3Const)
    return failure();

  auto filterRank3Ty = dyn_cast<RankedTensorType>(filterRank3Const.getType());
  if (!filterRank3Ty || !filterRank3Ty.hasStaticShape() ||
      filterRank3Ty.getRank() != 3) {
    return failure();
  }
  if (!filterRank3Ty.getElementType().isF32())
    return failure();

  auto filterRank2Const = getOrCreateFlattenedFilter(filterRank3Const,
                                                     filterRank3Ty);
  if (failed(filterRank2Const))
    return failure();

  auto filterRank2Ty = dyn_cast<RankedTensorType>((*filterRank2Const).getType());
  if (!filterRank2Ty || !filterRank2Ty.hasStaticShape() ||
      filterRank2Ty.getRank() != 2) {
    return failure();
  }
  if (!filterRank2Ty.getElementType().isF32())
    return failure();

  return std::make_tuple(filterRank3Const, filterRank3Ty, *filterRank2Const,
                         filterRank2Ty);
}


// Extracts the supported stride while enforcing unit dilation for the
// conv pattern this lowering handles.

static FailureOr<int64_t> getSupportedStride(linalg::Conv1DNcwFcwOp convOp) {
  SmallVector<int64_t> dilations;
  if (!extractPositiveInts(convOp.getDilations(), 1, dilations))
    return failure();
  if (dilations[0] != 1)
    return failure();

  SmallVector<int64_t> strides;
  if (!extractPositiveInts(convOp.getStrides(), 1, strides))
    return failure();

  return strides[0];
}


// Derives the shape metadata used by the rewrite and verifies the
// matched tensors agree with one another.

static FailureOr<ConvTensorShapeInfo> getValidatedShapeInfo(
    RankedTensorType inputTy, RankedTensorType biasTy,
    RankedTensorType filterRank3Ty, RankedTensorType filterRank2Ty,
    RankedTensorType outputTy) {
  auto inputShape = inputTy.getShape();
  auto filterShape = filterRank3Ty.getShape();
  auto filterFlatShape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();
  auto biasShape = biasTy.getShape();

  ConvTensorShapeInfo shapeInfo{
      inputShape[0], inputShape[1], inputShape[2],
      filterShape[0], filterShape[2], outputShape[2],
  };
  int64_t filterChannels = filterShape[1];
  int64_t outN = outputShape[0];
  int64_t outF = outputShape[1];

  if (shapeInfo.n != 1)
    return failure();
  if (filterChannels != shapeInfo.c)
    return failure();
  if (filterFlatShape[0] != shapeInfo.f ||
      filterFlatShape[1] != shapeInfo.c * shapeInfo.kw) {
    return failure();
  }
  if (biasShape[0] != shapeInfo.f)
    return failure();
  if (outN != shapeInfo.n || outF != shapeInfo.f)
    return failure();

  return shapeInfo;
}


// Matches the restricted conv1d pattern this lowering supports and
// collects the operands, types, and shape metadata it needs.

static FailureOr<MatchedConv1D>
matchSupportedConv1D(linalg::Conv1DNcwFcwOp convOp) {
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
  auto [filterRank3Const, filterRank3Ty, filterRank2Const, filterRank2Ty] =
      *filterConstants;

  auto stride = getSupportedStride(convOp);
  if (failed(stride))
    return failure();

  auto shapeInfo = getValidatedShapeInfo(inputTy, biasTy, filterRank3Ty,
                                         filterRank2Ty, outputTy);
  if (failed(shapeInfo))
    return failure();

  return MatchedConv1D{
      convOp,
      broadcastOp,
      activation,
      filterRank3Const,
      filterRank2Const,
      broadcastOp.getInput(),
      outputInit,
      inputTy,
      filterRank3Ty,
      filterRank2Ty,
      outputTy,
      *stride,
      shapeInfo->n,
      shapeInfo->c,
      shapeInfo->w,
      shapeInfo->f,
      shapeInfo->kw,
      shapeInfo->ow,
  };
}


// Builds a one-time transposed filter so each flattened patch matmul sees
// the weight matrix in `[C * KW, F]` layout.

static Value buildTransposedFilter(OpBuilder &builder, MatchedConv1D &match,
                                   const SlidingWindowLoweringState &state) {
  Value transposedFilterInit = builder.create<tensor::EmptyOp>(
      state.loc, ArrayRef<int64_t>{match.c * state.patchWidth, match.f},
      state.elementType);
  return builder
      .create<linalg::TransposeOp>(state.loc, match.filterRank2Const.getResult(),
                                   transposedFilterInit,
                                   ArrayRef<int64_t>{1, 0})
      .getResult()
      .front();
}


// Expands the bias vector into the same shape used by each per-window
// matmul result.

static Value buildExpandedBias(OpBuilder &builder, MatchedConv1D &match,
                               const SlidingWindowLoweringState &state) {
  SmallVector<ReassociationIndices, 2> biasExpandReassociation = {{0, 1}};
  return builder.create<tensor::ExpandShapeOp>(state.loc, state.matmulResultTy,
                                               match.bias,
                                               biasExpandReassociation);
}


// Allocates a tensor and fills it with zeros so later structured ops
// can treat it as their destination tensor.

static Value buildZeroInitializedTensor(OpBuilder &builder, Location loc,
                                        RankedTensorType tensorTy,
                                        Value zeroValue) {
  Value empty = builder.create<tensor::EmptyOp>(
      loc, tensorTy.getShape(), tensorTy.getElementType());
  return builder.create<linalg::FillOp>(loc, ValueRange{zeroValue},
                                        ValueRange{empty})
      .getResult(0);
}


// Precomputes the common types, loop bounds, and constants reused
// across the sliding-window rewrite.

static SlidingWindowLoweringState buildSlidingWindowState(
    OpBuilder &builder, MatchedConv1D &match) {
  Location loc = match.convOp.getLoc();
  Type elementType = match.inputTy.getElementType();
  int64_t patchWidth = match.kw;
  auto patchTy = RankedTensorType::get({1, match.c * patchWidth}, elementType);
  auto matmulResultTy = RankedTensorType::get({1, match.f}, elementType);

  SlidingWindowLoweringState state{
      loc,
      elementType,
      patchWidth,
      patchTy,
      matmulResultTy,
      match.outputTy,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      builder.create<arith::ConstantIndexOp>(loc, 1),
      builder.create<arith::ConstantIndexOp>(loc, match.ow),
      builder.create<arith::ConstantIndexOp>(loc, match.c),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.f),
      builder.create<arith::ConstantIndexOp>(loc, patchWidth),
      builder.create<arith::ConstantIndexOp>(loc, match.stride),
      builder.create<arith::ConstantFloatOp>(
          loc, cast<FloatType>(elementType), llvm::APFloat(0.0f)),
      Value{},
      Value{},
  };
  state.transposedFilter = buildTransposedFilter(builder, match, state);
  state.expandedBias = buildExpandedBias(builder, match, state);
  return state;
}


// Materializes the fully flattened input patch for one output position
// across all input channels in `[C * KW]` order.

static Value buildFlattenedPatch(OpBuilder &builder, MatchedConv1D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value owIdx) {
  Value patchInit = builder.create<tensor::EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);
  auto channelLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.cUpper, state.c1, ValueRange{patchInit},
      [&](OpBuilder &channelBuilder, Location channelLoc, Value cIdx,
          ValueRange channelIterArgs) {
        auto kwLoop = channelBuilder.create<scf::ForOp>(
            channelLoc, state.c0, state.kwUpper, state.c1, channelIterArgs,
            [&](OpBuilder &kwBuilder, Location kwLoc, Value kwIdx,
                ValueRange kwIterArgs) {
              Value iwBase =
                  kwBuilder.create<arith::MulIOp>(kwLoc, owIdx, state.strideW);
              Value iw =
                  kwBuilder.create<arith::AddIOp>(kwLoc, iwBase, kwIdx);
              Value inputValue = kwBuilder.create<tensor::ExtractOp>(
                  kwLoc, match.activation, ValueRange{state.c0, cIdx, iw});
              Value channelOffset = kwBuilder.create<arith::MulIOp>(
                  kwLoc, cIdx, state.patchWidthValue);
              Value flatIndex = kwBuilder.create<arith::AddIOp>(
                  kwLoc, channelOffset, kwIdx);
              Value updatedPatch = kwBuilder.create<tensor::InsertOp>(
                  kwLoc, inputValue, kwIterArgs[0],
                  ValueRange{state.c0, flatIndex});
              kwBuilder.create<scf::YieldOp>(kwLoc, updatedPatch);
            });
        channelBuilder.create<scf::YieldOp>(channelLoc, kwLoop.getResult(0));
      });
  channelLoop->setAttr(kSlidingWindowPatchAttr, builder.getUnitAttr());
  return channelLoop.getResult(0);
}


// Runs the full flattened patch matmul for one output position.

static Value buildPatchMatmul(OpBuilder &builder, MatchedConv1D &match,
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


// Adds the broadcasted bias to one output-position matmul result.

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


// Writes the computed output-channel values back into the final
// `[N, F, OW]` result tensor.

static Value assembleOutputChannels(OpBuilder &builder,
                                    const SlidingWindowLoweringState &state,
                                    Value biasedResult, Value owIdx,
                                    Value currentOutput) {
  auto channelAssembleLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.fUpper, state.c1, ValueRange{currentOutput},
      [&](OpBuilder &fBuilder, Location fLoc, Value fIdx,
          ValueRange fIterArgs) {
        Value channelValue = fBuilder.create<tensor::ExtractOp>(
            fLoc, biasedResult, ValueRange{state.c0, fIdx});
        Value updatedOutput = fBuilder.create<tensor::InsertOp>(
            fLoc, channelValue, fIterArgs[0],
            ValueRange{state.c0, fIdx, owIdx});
        fBuilder.create<scf::YieldOp>(fLoc, updatedOutput);
      });
  channelAssembleLoop->setAttr(kOutputChannelAssemblyAttr,
                               builder.getUnitAttr());
  return channelAssembleLoop.getResult(0);
}


// Lowers a single output position by building one flattened patch, running
// the full MVM, adding bias, and stitching the result back into the output tensor.

static Value lowerOutputPosition(OpBuilder &builder, MatchedConv1D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value owIdx, Value currentOutput) {
  Value patch = buildFlattenedPatch(builder, match, state, owIdx);
  Value matmulResult = buildPatchMatmul(builder, match, state, patch);
  Value biasedResult = addBiasToChannelResult(builder, state, matmulResult);
  return assembleOutputChannels(builder, state, biasedResult, owIdx,
                                currentOutput);
}


// Emits the loop IR that lowers one supported conv1d into a sequence of
// patch extraction, matmul, bias add, and output assembly ops.

static Value emitSlidingWindowIR(MatchedConv1D &match) {
  Operation *insertionPoint = match.convOp.getOperation();
  OpBuilder builder(insertionPoint->getContext());
  builder.setInsertionPointAfter(insertionPoint);
  SlidingWindowLoweringState state = buildSlidingWindowState(builder, match);
  Value rewrittenOutputInit =
      buildZeroInitializedTensor(builder, state.loc, state.outputTy,
                                 state.zeroValue);

  auto owLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.owUpper, state.c1,
      ValueRange{rewrittenOutputInit},
      [&](OpBuilder &owBuilder, Location owLoc, Value owIdx,
          ValueRange owIterArgs) {
        Value updatedOutput =
            lowerOutputPosition(owBuilder, match, state, owIdx, owIterArgs[0]);
        owBuilder.create<scf::YieldOp>(owLoc, updatedOutput);
      });
  owLoop->setAttr(kRewrittenConvOutputAttr, builder.getUnitAttr());
  return owLoop.getResult(0);
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef RewriteConv1DToMatmulPass::getArgument() const {
  return "analog-rewrite-conv1d-to-matmul";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef RewriteConv1DToMatmulPass::getDescription() const {
  return "Rewrite supported conv1d ops into a matmul-oriented form";
}


// Declares the dialects this pass may create while building the
// replacement tensor and loop IR.

void RewriteConv1DToMatmulPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}


// Rewrites each supported conv1d into an explicit sliding-window
// matmul form and removes now-dead source ops.

void RewriteConv1DToMatmulPass::runOnOperation() {
  auto func = getOperation();
  int64_t nextMatrixSourceId = 0;

  func.walk([&](linalg::Conv1DNcwFcwOp convOp) {
    FailureOr<MatchedConv1D> maybeMatch = matchSupportedConv1D(convOp);
    if (failed(maybeMatch))
      return;

    MatchedConv1D match = *maybeMatch;

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
    Operation *filterRank3 = match.filterRank3Const.getOperation();
    if (conv->use_empty())
      conv->erase();
    if (broadcast->use_empty())
      broadcast->erase();
    if (filterRank3->hasAttr(kDeleteInFuturePassAttr) &&
        filterRank3->use_empty()) {
      filterRank3->erase();
    }
  });
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createRewriteConv1DToMatmulPass() {
  return std::make_unique<RewriteConv1DToMatmulPass>();
}

} // namespace analog
} // namespace mlir
