#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv2DToMatmul.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

static constexpr llvm::StringLiteral kMatrixSourceIdAttr = "analog.matrix_source_id";
static constexpr llvm::StringLiteral kDeleteInFuturePassAttr = "analog.delete_in_future_pass";
static constexpr llvm::StringLiteral kSlidingWindowMatmulAttr = "analog.sliding_window_matmul";
static constexpr llvm::StringLiteral kSlidingWindowBiasAddAttr = "analog.sliding_window_bias_add";
static constexpr llvm::StringLiteral kOutputChannelAssemblyAttr = "analog.output_channel_assembly";
static constexpr llvm::StringLiteral kSlidingWindowPatchAttr = "analog.sliding_window_patch";
static constexpr llvm::StringLiteral kRewrittenConv2DOutputAttr = "analog.rewritten_conv2d_output";

struct MatchedConv2D {
  linalg::Conv2DNchwFchwOp convOp;
  linalg::BroadcastOp broadcastOp;
  Value activation;
  arith::ConstantOp filterRank4Const;
  arith::ConstantOp filterRank2Const;
  Value bias;
  Value outputInit;
  RankedTensorType inputTy;
  RankedTensorType filterRank4Ty;
  RankedTensorType filterRank2Ty;
  RankedTensorType outputTy;
  SmallVector<int64_t> strides;
  int64_t n;
  int64_t c;
  int64_t h;
  int64_t w;
  int64_t f;
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
  Value cUpper;
  Value khUpper;
  Value kwUpper;
  Value fUpper;
  Value patchWidthValue;
  Value strideH;
  Value strideW;
  Value kwValue;
  Value zeroValue;
  Value transposedFilter;
  Value expandedBias;
};

struct ConvTensorShapeInfo {
  int64_t n;
  int64_t c;
  int64_t h;
  int64_t w;
  int64_t f;
  int64_t kh;
  int64_t kw;
  int64_t oh;
  int64_t ow;
};


// Finds the rank-2 filter constant emitted by the prepare pass next to
// the original rank-4 filter constant.

static arith::ConstantOp findPreparedFlattenedFilter(arith::ConstantOp filterConst) {
  if (!filterConst || !filterConst->hasAttr(kDeleteInFuturePassAttr)) {
    return {};
  }

  // PrepareConv2DToMatmul emits the flattened weight constant immediately
  // after the original rank-4 filter. This helper relies on that adjacency.
  Operation *next = filterConst->getNextNode();
  auto flattenedConst = dyn_cast_or_null<arith::ConstantOp>(next);
  if (!flattenedConst) {
    return {};
  }

  auto flattenedTy = dyn_cast<RankedTensorType>(flattenedConst.getType());
  if (!flattenedTy || flattenedTy.getRank() != 2) {
    return {};
  }

  return flattenedConst;
}


// Reads a two-element integer attribute into a vector and rejects
// missing, malformed, or non-positive values.

static bool extractTwoPositiveInts(DenseIntElementsAttr attr,
                                   SmallVectorImpl<int64_t> &values) {
  values.clear();
  if (!attr) {
    return false;
  }

  for (APInt value : attr.getValues<APInt>()) {
    values.push_back(value.getSExtValue());
  }

  if (values.size() != 2) {
    return false;
  }

  return values[0] > 0 && values[1] > 0;
}


// Validates the activation and output tensors expected by the conv
// rewrite and returns their ranked tensor types.

static FailureOr<std::pair<RankedTensorType, RankedTensorType>>
getSupportedInputAndOutputTypes(linalg::Conv2DNchwFchwOp convOp, Value activation) {
  auto inputTy = dyn_cast<RankedTensorType>(activation.getType());
  auto outputTy = dyn_cast<RankedTensorType>(convOp.getResult(0).getType());
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape()) {
    return failure();
  }
  if (inputTy.getRank() != 4 || outputTy.getRank() != 4) {
    return failure();
  }
  if (!inputTy.getElementType().isF32() || !outputTy.getElementType().isF32()) {
    return failure();
  }

  return std::make_pair(inputTy, outputTy);
}


// Validates the broadcasted bias initializer and returns the broadcast
// op together with the 1-D bias tensor type.

static FailureOr<std::pair<linalg::BroadcastOp, RankedTensorType>>
getSupportedBiasBroadcast(Value outputInit, RankedTensorType outputTy) {
  auto broadcastOp = outputInit.getDefiningOp<linalg::BroadcastOp>();
  if (!broadcastOp) {
    return failure();
  }

  auto biasTy = dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
  auto broadcastInitTy =
      dyn_cast<RankedTensorType>(broadcastOp.getInit().getType());
  if (!biasTy || !broadcastInitTy || !biasTy.hasStaticShape() ||
      !broadcastInitTy.hasStaticShape()) {
    return failure();
  }
  if (biasTy.getRank() != 1 || broadcastInitTy.getRank() != 4) {
    return failure();
  }
  if (broadcastInitTy != outputTy) {
    return failure();
  }

  auto dims = broadcastOp.getDimensions();
  if (dims.size() != 3 || dims[0] != 0 || dims[1] != 2 || dims[2] != 3) {
    return failure();
  }

  return std::make_pair(broadcastOp, biasTy);
}


// Validates the original and prepared filter constants and returns
// both constants together with their tensor types.

static FailureOr<std::tuple<arith::ConstantOp, RankedTensorType,
                            arith::ConstantOp, RankedTensorType>>
getSupportedFilterConstants(Value filter) {
  auto filterRank4Const = filter.getDefiningOp<arith::ConstantOp>();
  if (!filterRank4Const) {
    return failure();
  }

  auto filterRank4Ty = dyn_cast<RankedTensorType>(filterRank4Const.getType());
  if (!filterRank4Ty || !filterRank4Ty.hasStaticShape() ||
      filterRank4Ty.getRank() != 4) {
    return failure();
  }
  if (!filterRank4Ty.getElementType().isF32()) {
    return failure();
  }

  auto filterRank2Const = findPreparedFlattenedFilter(filterRank4Const);
  if (!filterRank2Const) {
    return failure();
  }

  auto filterRank2Ty = dyn_cast<RankedTensorType>(filterRank2Const.getType());
  if (!filterRank2Ty || !filterRank2Ty.hasStaticShape() ||
      filterRank2Ty.getRank() != 2) {
    return failure();
  }
  if (!filterRank2Ty.getElementType().isF32()) {
    return failure();
  }

  return std::make_tuple(filterRank4Const, filterRank4Ty, filterRank2Const,
                         filterRank2Ty);
}


// Extracts the supported stride values while enforcing unit dilation
// for the conv pattern this lowering handles.

static FailureOr<SmallVector<int64_t>> getSupportedStrides(
    linalg::Conv2DNchwFchwOp convOp) {
  SmallVector<int64_t> dilations;
  if (!extractTwoPositiveInts(convOp.getDilations(), dilations)) {
    return failure();
  }
  if (dilations[0] != 1 || dilations[1] != 1) {
    return failure();
  }

  SmallVector<int64_t> strides;
  if (!extractTwoPositiveInts(convOp.getStrides(), strides)) {
    return failure();
  }

  return strides;
}


// Derives the shape metadata used by the rewrite and verifies the
// matched tensors agree with one another.

static FailureOr<ConvTensorShapeInfo> getValidatedShapeInfo(
    RankedTensorType inputTy, RankedTensorType biasTy,
    RankedTensorType filterRank4Ty, RankedTensorType filterRank2Ty,
    RankedTensorType outputTy) {
  auto inputShape = inputTy.getShape();
  auto filterShape = filterRank4Ty.getShape();
  auto filterFlatShape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();
  auto biasShape = biasTy.getShape();

  ConvTensorShapeInfo shapeInfo{
      inputShape[0],  inputShape[1],  inputShape[2],  inputShape[3],
      filterShape[0], filterShape[2], filterShape[3], outputShape[2],
      outputShape[3],
  };
  int64_t filterChannels = filterShape[1];
  int64_t outN = outputShape[0];
  int64_t outF = outputShape[1];

  if (shapeInfo.n != 1) {
    return failure();
  }
  if (filterChannels != shapeInfo.c) {
    return failure();
  }
  if (filterFlatShape[0] != shapeInfo.f ||
      filterFlatShape[1] != shapeInfo.c * shapeInfo.kh * shapeInfo.kw) {
    return failure();
  }
  if (biasShape[0] != shapeInfo.f) {
    return failure();
  }
  if (outN != shapeInfo.n || outF != shapeInfo.f) {
    return failure();
  }

  return shapeInfo;
}


// Matches the restricted conv2d pattern this lowering supports and
// collects all of the operands, types, and shape metadata it needs.

static FailureOr<MatchedConv2D> matchSupportedConv2D(linalg::Conv2DNchwFchwOp convOp) {
  // This rewrite is intentionally narrow: static-shape NCHW/FCHW f32 conv,
  // batch size 1, dilation 1, and bias materialized through linalg.broadcast.
  if (convOp.getInputs().size() != 2 || convOp.getOutputs().size() != 1) {
    return failure();
  }

  Value activation = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value outputInit = convOp.getOutputs()[0];

  auto inputOutputTypes = getSupportedInputAndOutputTypes(convOp, activation);
  if (failed(inputOutputTypes)) {
    return failure();
  }
  auto [inputTy, outputTy] = *inputOutputTypes;

  auto biasBroadcast = getSupportedBiasBroadcast(outputInit, outputTy);
  if (failed(biasBroadcast)) {
    return failure();
  }
  auto [broadcastOp, biasTy] = *biasBroadcast;

  auto filterConstants = getSupportedFilterConstants(filter);
  if (failed(filterConstants)) {
    return failure();
  }
  auto [filterRank4Const, filterRank4Ty, filterRank2Const, filterRank2Ty] =
      *filterConstants;

  auto strides = getSupportedStrides(convOp);
  if (failed(strides)) {
    return failure();
  }

  auto shapeInfo = getValidatedShapeInfo(inputTy, biasTy, filterRank4Ty,
                                         filterRank2Ty, outputTy);
  if (failed(shapeInfo)) {
    return failure();
  }

  return MatchedConv2D{
      convOp,
      broadcastOp,
      activation,
      filterRank4Const,
      filterRank2Const,
      broadcastOp.getInput(),
      outputInit,
      inputTy,
      filterRank4Ty,
      filterRank2Ty,
      outputTy,
      *strides,
      shapeInfo->n,
      shapeInfo->c,
      shapeInfo->h,
      shapeInfo->w,
      shapeInfo->f,
      shapeInfo->kh,
      shapeInfo->kw,
      shapeInfo->oh,
      shapeInfo->ow,
  };
}


// Builds a one-time transposed filter so each channel slice can be
// extracted in the layout expected by the generated matmuls.

static Value buildTransposedFilter(OpBuilder &builder, MatchedConv2D &match,
                                   const SlidingWindowLoweringState &state) {
  Value transposedFilterInit = builder.create<tensor::EmptyOp>(
      state.loc,
      ArrayRef<int64_t>{match.c * state.patchWidth, match.f},
      state.elementType);
  // The generated matmuls consume one output-channel slice at a time, so the
  // flattened filter is transposed once up front into [patch, out_channel].
  return builder
      .create<linalg::TransposeOp>(state.loc, match.filterRank2Const.getResult(),
                                   transposedFilterInit, ArrayRef<int64_t>{1, 0})
      .getResult()
      .front();
}


// Expands the bias vector into the same shape used by each per-window
// matmul result.

static Value buildExpandedBias(OpBuilder &builder, MatchedConv2D &match,
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
  Value empty =
      builder.create<tensor::EmptyOp>(loc, tensorTy.getShape(), tensorTy.getElementType());
  return builder.create<linalg::FillOp>(loc, ValueRange{zeroValue},
                                        ValueRange{empty})
      .getResult(0);
}


// Precomputes the common types, loop bounds, and constants reused
// across the sliding-window rewrite.

static SlidingWindowLoweringState buildSlidingWindowState(OpBuilder &builder,
                                                          MatchedConv2D &match) {
  Location loc = match.convOp.getLoc();
  Type elementType = match.inputTy.getElementType();
  int64_t patchWidth = match.kh * match.kw;
  auto patchTy =
      RankedTensorType::get({1, match.c * patchWidth}, elementType);
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
      builder.create<arith::ConstantIndexOp>(loc, match.oh),
      builder.create<arith::ConstantIndexOp>(loc, match.ow),
      builder.create<arith::ConstantIndexOp>(loc, match.c),
      builder.create<arith::ConstantIndexOp>(loc, match.kh),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantIndexOp>(loc, match.f),
      builder.create<arith::ConstantIndexOp>(loc, patchWidth),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[0]),
      builder.create<arith::ConstantIndexOp>(loc, match.strides[1]),
      builder.create<arith::ConstantIndexOp>(loc, match.kw),
      builder.create<arith::ConstantFloatOp>(
          loc, llvm::cast<FloatType>(elementType), llvm::APFloat(0.0f)),
      Value{},
      Value{},
  };
  state.transposedFilter = buildTransposedFilter(builder, match, state);
  state.expandedBias = buildExpandedBias(builder, match, state);
  return state;
}


// Materializes the fully flattened input patch for one output position
// across all input channels in `[C * KH * KW]` order.

static Value buildFlattenedPatch(OpBuilder &builder, MatchedConv2D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value ohIdx, Value owIdx) {
  Value patchInit =
      builder.create<tensor::EmptyOp>(state.loc, state.patchTy.getShape(),
                                      state.elementType);
  auto channelLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.cUpper, state.c1, ValueRange{patchInit},
      [&](OpBuilder &channelBuilder, Location channelLoc, Value cIdx,
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
                    Value ih =
                        kwBuilder.create<arith::AddIOp>(kwLoc, ihBase, khIdx);
                    Value iw =
                        kwBuilder.create<arith::AddIOp>(kwLoc, iwBase, kwIdx);
                    Value inputValue = kwBuilder.create<tensor::ExtractOp>(
                        kwLoc, match.activation,
                        ValueRange{state.c0, cIdx, ih, iw});
                    Value channelOffset = kwBuilder.create<arith::MulIOp>(
                        kwLoc, cIdx, state.patchWidthValue);
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
  channelLoop->setAttr(kSlidingWindowPatchAttr, builder.getUnitAttr());
  return channelLoop.getResult(0);
}


// Runs the full flattened patch matmul for one output position.

static Value buildPatchMatmul(OpBuilder &builder, MatchedConv2D &match,
                              const SlidingWindowLoweringState &state,
                              Value patch) {
  Value matmulInit = buildZeroInitializedTensor(builder, state.loc,
                                                state.matmulResultTy,
                                                state.zeroValue);
  auto matmulOp = builder.create<linalg::MatmulOp>(
      state.loc, state.matmulResultTy, ValueRange{patch, state.transposedFilter},
      ValueRange{matmulInit});
  matmulOp->setAttr(kSlidingWindowMatmulAttr, builder.getUnitAttr());
  if (auto matrixSourceId = match.filterRank2Const->getAttr(kMatrixSourceIdAttr)) {
    matmulOp->setAttr(kMatrixSourceIdAttr, matrixSourceId);
  }
  return matmulOp.getResult(0);
}


// Adds the broadcasted bias once all channel contributions have been
// accumulated for an output position.

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
// `[N, F, OH, OW]` result tensor.

static Value assembleOutputChannels(OpBuilder &builder,
                                    const SlidingWindowLoweringState &state,
                                    Value biasedResult, Value ohIdx, Value owIdx,
                                    Value currentOutput) {
  auto channelAssembleLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.fUpper, state.c1, ValueRange{currentOutput},
      [&](OpBuilder &fBuilder, Location fLoc, Value fIdx, ValueRange fIterArgs) {
        Value channelValue = fBuilder.create<tensor::ExtractOp>(
            fLoc, biasedResult, ValueRange{state.c0, fIdx});
        Value updatedOutput = fBuilder.create<tensor::InsertOp>(
            fLoc, channelValue, fIterArgs[0],
            ValueRange{state.c0, fIdx, ohIdx, owIdx});
        fBuilder.create<scf::YieldOp>(fLoc, updatedOutput);
      });
  channelAssembleLoop->setAttr(kOutputChannelAssemblyAttr, builder.getUnitAttr());
  return channelAssembleLoop.getResult(0);
}


// Lowers a single output position by building one flattened patch, running
// the full MVM, adding bias, and stitching the result back into the output tensor.
// bias, and stitching the result back into the output tensor.

static Value lowerOutputPosition(OpBuilder &builder, MatchedConv2D &match,
                                 const SlidingWindowLoweringState &state,
                                 Value ohIdx, Value owIdx,
                                 Value currentOutput) {
  Value patch = buildFlattenedPatch(builder, match, state, ohIdx, owIdx);
  Value matmulResult = buildPatchMatmul(builder, match, state, patch);
  Value biasedResult = addBiasToChannelResult(builder, state, matmulResult);
  return assembleOutputChannels(builder, state, biasedResult, ohIdx, owIdx,
                                currentOutput);
}


// Emits the nested loop IR that lowers one supported conv2d into a
// sequence of patch extraction, matmul, and output assembly ops.

static Value emitSlidingWindowIR(MatchedConv2D &match) {
  Operation *insertionPoint = match.convOp.getOperation();
  OpBuilder builder(insertionPoint->getContext());
  builder.setInsertionPointAfter(insertionPoint);
  SlidingWindowLoweringState state = buildSlidingWindowState(builder, match);
  Value rewrittenOutputInit =
      buildZeroInitializedTensor(builder, state.loc, state.outputTy,
                                 state.zeroValue);

  auto ohLoop = builder.create<scf::ForOp>(
      state.loc, state.c0, state.ohUpper, state.c1, ValueRange{rewrittenOutputInit},
      [&](OpBuilder &ohBuilder, Location ohLoc, Value ohIdx, ValueRange ohIterArgs) {
        auto owLoop = ohBuilder.create<scf::ForOp>(
            ohLoc, state.c0, state.owUpper, state.c1, ohIterArgs,
            [&](OpBuilder &owBuilder, Location owLoc, Value owIdx, ValueRange owIterArgs) {
              Value updatedOutput = lowerOutputPosition(
                  owBuilder, match, state, ohIdx, owIdx, owIterArgs[0]);
              owBuilder.create<scf::YieldOp>(owLoc, updatedOutput);
            });

        ohBuilder.create<scf::YieldOp>(ohLoc, owLoop.getResult(0));
      });
  ohLoop->setAttr(kRewrittenConv2DOutputAttr, builder.getUnitAttr());
  return ohLoop.getResult(0);
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef RewriteConv2DToMatmulPass::getArgument() const {
  return "analog-rewrite-conv2d-to-matmul";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef RewriteConv2DToMatmulPass::getDescription() const {
  return "Rewrite supported conv2d ops into a matmul-oriented form";
}


// Declares the dialects this pass may create while building the
// replacement tensor and loop IR.

void RewriteConv2DToMatmulPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}


// Rewrites each supported conv2d into an explicit sliding-window
// matmul form and removes now-dead source ops.

void RewriteConv2DToMatmulPass::runOnOperation() {
  auto func = getOperation();
  int64_t nextMatrixSourceId = 0;

  func.walk([&](linalg::Conv2DNchwFchwOp convOp) {
    FailureOr<MatchedConv2D> maybeMatch = matchSupportedConv2D(convOp);
    if (failed(maybeMatch)) {
      return;
    }

    MatchedConv2D match = *maybeMatch;

    // The later analog passes need an explicit link between the original
    // weight tensor and each generated matmul, so the rewrite seeds that id.
    IntegerAttr matrixSourceId =
        match.filterRank2Const->getAttrOfType<IntegerAttr>(kMatrixSourceIdAttr);
    if (!matrixSourceId) {
      matrixSourceId = IntegerAttr::get(
          IntegerType::get(func.getContext(), 64), nextMatrixSourceId++);
      match.filterRank2Const->setAttr(kMatrixSourceIdAttr, matrixSourceId);
    }

    /*
    llvm::errs() << "[analog-rewrite-conv2d-to-matmul] matched candidate\n";
    llvm::errs() << "  conv: ";
    match.convOp->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  activation: ";
    match.activation.print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  filter-rank4: ";
    match.filterRank4Const->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  filter-rank2: ";
    match.filterRank2Const->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  broadcast: ";
    match.broadcastOp->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  bias: ";
    match.bias.print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  output-init: ";
    match.outputInit.print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  input-shape: [" << match.n << ", " << match.c << ", "
                 << match.h << ", " << match.w << "]\n";
    llvm::errs() << "  filter-shape: [" << match.f << ", " << match.c << ", "
                 << match.kh << ", " << match.kw << "]\n";
    llvm::errs() << "  flattened-filter-shape: [" << match.filterRank2Ty.getShape()[0]
                 << ", " << match.filterRank2Ty.getShape()[1] << "]\n";
    llvm::errs() << "  output-shape: [" << match.outputTy.getShape()[0] << ", "
                 << match.outputTy.getShape()[1] << ", " << match.oh << ", "
                 << match.ow << "]\n";
    llvm::errs() << "  strides: [" << match.strides[0] << ", " << match.strides[1]
                 << "]\n";
    llvm::errs() << "  dilations: [1, 1]\n";
    llvm::errs() << "\n";
    */

    Value rewrittenOutput = emitSlidingWindowIR(match);
    match.convOp.getResult(0).replaceAllUsesWith(rewrittenOutput);

    Operation *conv = match.convOp.getOperation();
    Operation *broadcast = match.broadcastOp.getOperation();
    Operation *filterRank4 = match.filterRank4Const.getOperation();
    if (conv->use_empty()) {
      conv->erase();
    }
    if (broadcast->use_empty()) {
      broadcast->erase();
    }
    if (filterRank4->hasAttr(kDeleteInFuturePassAttr) &&
        filterRank4->use_empty()) {
      filterRank4->erase();
    }
  });
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createRewriteConv2DToMatmulPass() {
  return std::make_unique<RewriteConv2DToMatmulPass>();
}

} // namespace analog
} // namespace mlir
