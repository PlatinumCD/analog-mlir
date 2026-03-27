#include "analog-mlir/Dialect/Analog/Transforms/PrepareRNNForAnalog.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

constexpr StringLiteral kRecurrentPatternAttr = "analog.recurrent_pattern";
constexpr StringLiteral kPreparedForAnalogAttr = "analog.prepared_for_analog";

struct MatchedRNNSequence {
  scf::ExecuteRegionOp executeRegion;
  Value sequenceInput;
  Value initialHidden;
  arith::ConstantOp inputWeightConst;
  arith::ConstantOp recurrentWeightConst;
  arith::ConstantOp inputBiasConst;
  arith::ConstantOp recurrentBiasConst;
  RankedTensorType sequenceInputTy;
  RankedTensorType initialHiddenTy;
  RankedTensorType outputTy;
  int64_t sequenceLength;
  int64_t inputSize;
  int64_t hiddenSize;
};

//===----------------------------------------------------------------------===//
// Generic Helpers
//===----------------------------------------------------------------------===//

// Returns whether the generic is a simple elementwise floating-point add.
static bool isPointwiseAddGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1)
    return false;

  Block &body = generic.getRegion().front();
  auto it = body.begin();
  auto addOp = dyn_cast<arith::AddFOp>(*it++);
  if (!addOp)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(*it++);
  if (!yieldOp || it != body.end())
    return false;
  return yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == addOp.getResult();
}

// Extracts floating-point payloads from dense or resource-backed constants.
static FailureOr<SmallVector<float>> extractFloatElements(arith::ConstantOp op) {
  if (!op)
    return failure();

  TypedAttr attr = op.getValue();
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (APFloat value : denseAttr.getValues<APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (auto resourceAttr = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    if (std::optional<ArrayRef<float>> values = resourceAttr.tryGetAsArrayRef())
      return SmallVector<float>(values->begin(), values->end());
  }

  return failure();
}

// Builds a rank-2 + rank-1 pointwise add used for the final bias add.
static Value buildPointwiseAdd(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs, RankedTensorType resultTy) {
  auto empty = builder.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                               resultTy.getElementType());
  SmallVector<AffineMap> maps = {
      builder.getMultiDimIdentityMap(resultTy.getRank()),
      AffineMap::get(resultTy.getRank(), 0, builder.getAffineDimExpr(1),
                     builder.getContext()),
      builder.getMultiDimIdentityMap(resultTy.getRank())};
  SmallVector<utils::IteratorType> iterators(
      resultTy.getRank(), utils::IteratorType::parallel);

  auto add = builder.create<linalg::GenericOp>(
      loc, resultTy, ValueRange{lhs, rhs}, ValueRange{empty}, maps, iterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      });
  return add.getResult(0);
}

// Builds the final tanh over the fused affine result.
static Value buildTanh(OpBuilder &builder, Location loc, Value input,
                       RankedTensorType resultTy) {
  auto empty = builder.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                               resultTy.getElementType());
  SmallVector<AffineMap> maps = {builder.getMultiDimIdentityMap(resultTy.getRank()),
                                 builder.getMultiDimIdentityMap(resultTy.getRank())};
  SmallVector<utils::IteratorType> iterators(
      resultTy.getRank(), utils::IteratorType::parallel);

  auto tanh = builder.create<linalg::GenericOp>(
      loc, resultTy, ValueRange{input}, ValueRange{empty}, maps, iterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value value = nestedBuilder.create<math::TanhOp>(nestedLoc, args[0]);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, value);
      });
  return tanh.getResult(0);
}

static void eraseIfDead(arith::ConstantOp op) {
  if (op && op->use_empty())
    op.erase();
}

static bool hasShape(Value value, ArrayRef<int64_t> shape) {
  auto ty = dyn_cast<RankedTensorType>(value.getType());
  return ty && ty.hasStaticShape() && ty.getShape() == shape;
}

//===----------------------------------------------------------------------===//
// RNN Sequence Matching
//===----------------------------------------------------------------------===//

// Finds the input-side weight constant used by the batched input projection.
static arith::ConstantOp findInputWeightConstant(Operation *scope, int64_t seqLen,
                                                 int64_t inputSize,
                                                 int64_t hiddenSize) {
  arith::ConstantOp match;
  scope->walk([&](linalg::MatmulOp matmul) {
    if (match)
      return;
    if (!hasShape(matmul.getInputs()[0], {seqLen, inputSize}))
      return;

    auto transpose = matmul.getInputs()[1].getDefiningOp<linalg::TransposeOp>();
    if (!transpose)
      return;

    auto permutation = transpose.getPermutation();
    if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
      return;

    auto constant = transpose.getInput().getDefiningOp<arith::ConstantOp>();
    if (!constant || !hasShape(constant.getResult(), {hiddenSize, inputSize}))
      return;
    match = constant;
  });
  return match;
}

// Finds the recurrent weight constant used by per-step hidden-state matmuls.
static arith::ConstantOp findRecurrentWeightConstant(Operation *scope,
                                                     int64_t hiddenSize) {
  arith::ConstantOp match;
  scope->walk([&](linalg::MatmulOp matmul) {
    if (match)
      return;
    if (!hasShape(matmul.getInputs()[0], {1, hiddenSize}))
      return;

    auto transpose = matmul.getInputs()[1].getDefiningOp<linalg::TransposeOp>();
    if (!transpose)
      return;

    auto permutation = transpose.getPermutation();
    if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
      return;

    auto constant = transpose.getInput().getDefiningOp<arith::ConstantOp>();
    if (!constant || !hasShape(constant.getResult(), {hiddenSize, hiddenSize}))
      return;
    match = constant;
  });
  return match;
}

// Finds the rank-1 input bias used by the batched input projection.
static arith::ConstantOp findInputBiasConstant(Operation *scope, int64_t seqLen,
                                               int64_t hiddenSize) {
  arith::ConstantOp match;
  scope->walk([&](linalg::GenericOp generic) {
    if (match || !isPointwiseAddGeneric(generic))
      return;
    if (!hasShape(generic.getResult(0), {seqLen, hiddenSize}))
      return;

    for (Value operand : generic.getDpsInputs()) {
      auto constant = operand.getDefiningOp<arith::ConstantOp>();
      if (!constant || !hasShape(constant.getResult(), {hiddenSize}))
        continue;
      match = constant;
      return;
    }
  });
  return match;
}

// Finds the rank-1 recurrent bias used by each per-step recurrent update.
static arith::ConstantOp findRecurrentBiasConstant(Operation *scope,
                                                   int64_t hiddenSize) {
  arith::ConstantOp match;
  scope->walk([&](linalg::GenericOp generic) {
    if (match || !isPointwiseAddGeneric(generic))
      return;
    if (!hasShape(generic.getResult(0), {1, hiddenSize}))
      return;

    for (Value operand : generic.getDpsInputs()) {
      auto constant = operand.getDefiningOp<arith::ConstantOp>();
      if (!constant || !hasShape(constant.getResult(), {hiddenSize}))
        continue;
      match = constant;
      return;
    }
  });
  return match;
}

// Matches the isolated unrolled RNN sequence boundary emitted by
// IdentifyRecurrentPatternsPass and extracts the source parameters needed for
// loop-based analog preparation.
static FailureOr<MatchedRNNSequence>
matchRNNSequenceBlock(func::FuncOp func, scf::ExecuteRegionOp executeRegion) {
  if (!executeRegion->hasAttr(kRecurrentPatternAttr) ||
      executeRegion->getAttrOfType<StringAttr>(kRecurrentPatternAttr).getValue() !=
          "rnn" ||
      executeRegion->hasAttr(kPreparedForAnalogAttr))
    return failure();

  if (func.getNumArguments() != 2)
    return failure();

  auto sequenceInputTy =
      dyn_cast<RankedTensorType>(func.getArgument(0).getType());
  auto initialHiddenTy =
      dyn_cast<RankedTensorType>(func.getArgument(1).getType());
  auto outputTy = dyn_cast<RankedTensorType>(executeRegion.getResult(0).getType());
  if (!sequenceInputTy || !initialHiddenTy || !outputTy ||
      !sequenceInputTy.hasStaticShape() || !initialHiddenTy.hasStaticShape() ||
      !outputTy.hasStaticShape())
    return failure();

  if (sequenceInputTy.getRank() != 3 || initialHiddenTy.getRank() != 3 ||
      outputTy.getRank() != 2)
    return failure();
  if (!sequenceInputTy.getElementType().isF32() ||
      !initialHiddenTy.getElementType().isF32() ||
      !outputTy.getElementType().isF32())
    return failure();

  int64_t batch = sequenceInputTy.getShape()[0];
  int64_t seqLen = sequenceInputTy.getShape()[1];
  int64_t inputSize = sequenceInputTy.getShape()[2];
  int64_t hiddenBatch = initialHiddenTy.getShape()[0];
  int64_t hiddenSteps = initialHiddenTy.getShape()[1];
  int64_t hiddenSize = initialHiddenTy.getShape()[2];
  if (batch != 1 || hiddenBatch != 1 || hiddenSteps != 1)
    return failure();
  if (outputTy.getShape()[0] != 1 || outputTy.getShape()[1] != hiddenSize)
    return failure();

  arith::ConstantOp inputWeightConst =
      findInputWeightConstant(executeRegion, seqLen, inputSize, hiddenSize);
  arith::ConstantOp recurrentWeightConst =
      findRecurrentWeightConstant(executeRegion, hiddenSize);
  arith::ConstantOp inputBiasConst =
      findInputBiasConstant(executeRegion, seqLen, hiddenSize);
  arith::ConstantOp recurrentBiasConst =
      findRecurrentBiasConstant(executeRegion, hiddenSize);
  if (!inputWeightConst || !recurrentWeightConst || !inputBiasConst ||
      !recurrentBiasConst)
    return failure();

  MatchedRNNSequence match;
  match.executeRegion = executeRegion;
  match.sequenceInput = func.getArgument(0);
  match.initialHidden = func.getArgument(1);
  match.inputWeightConst = inputWeightConst;
  match.recurrentWeightConst = recurrentWeightConst;
  match.inputBiasConst = inputBiasConst;
  match.recurrentBiasConst = recurrentBiasConst;
  match.sequenceInputTy = sequenceInputTy;
  match.initialHiddenTy = initialHiddenTy;
  match.outputTy = outputTy;
  match.sequenceLength = seqLen;
  match.inputSize = inputSize;
  match.hiddenSize = hiddenSize;
  return match;
}

//===----------------------------------------------------------------------===//
// Resource Synthesis
//===----------------------------------------------------------------------===//

// Creates a new top-level fused bias resource by summing the input and
// recurrent biases elementwise.
static FailureOr<arith::ConstantOp> createFusedBiasResourceConstant(
    OpBuilder &builder, Location loc, MatchedRNNSequence &match,
    StringRef resourceName) {
  FailureOr<SmallVector<float>> inputBiasValues =
      extractFloatElements(match.inputBiasConst);
  FailureOr<SmallVector<float>> recurrentBiasValues =
      extractFloatElements(match.recurrentBiasConst);
  if (failed(inputBiasValues) || failed(recurrentBiasValues))
    return failure();

  SmallVector<float> fusedValues;
  fusedValues.reserve(match.hiddenSize);
  for (int64_t i = 0; i < match.hiddenSize; ++i)
    fusedValues.push_back((*inputBiasValues)[i] + (*recurrentBiasValues)[i]);

  auto fusedTy = RankedTensorType::get({match.hiddenSize}, builder.getF32Type());
  Attribute fusedAttr = DenseF32ResourceElementsAttr::get(
      fusedTy, resourceName,
      HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(fusedValues));
  return builder.create<arith::ConstantOp>(loc, fusedTy,
                                           cast<TypedAttr>(fusedAttr));
}

// Creates a new top-level fused weight resource representing
// `[W_ih | W_hh]`.
static FailureOr<arith::ConstantOp> createFusedWeightResourceConstant(
    OpBuilder &builder, Location loc, MatchedRNNSequence &match,
    StringRef resourceName) {
  FailureOr<SmallVector<float>> inputWeightValues =
      extractFloatElements(match.inputWeightConst);
  FailureOr<SmallVector<float>> recurrentWeightValues =
      extractFloatElements(match.recurrentWeightConst);
  if (failed(inputWeightValues) || failed(recurrentWeightValues))
    return failure();

  SmallVector<float> fusedValues;
  fusedValues.reserve(match.hiddenSize * (match.inputSize + match.hiddenSize));
  for (int64_t row = 0; row < match.hiddenSize; ++row) {
    int64_t inputBase = row * match.inputSize;
    fusedValues.append(inputWeightValues->begin() + inputBase,
                       inputWeightValues->begin() + inputBase + match.inputSize);
    int64_t recurrentBase = row * match.hiddenSize;
    fusedValues.append(recurrentWeightValues->begin() + recurrentBase,
                       recurrentWeightValues->begin() + recurrentBase +
                           match.hiddenSize);
  }

  auto fusedTy = RankedTensorType::get(
      {match.hiddenSize, match.inputSize + match.hiddenSize},
      builder.getF32Type());
  Attribute fusedAttr = DenseF32ResourceElementsAttr::get(
      fusedTy, resourceName,
      HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(fusedValues));
  return builder.create<arith::ConstantOp>(loc, fusedTy,
                                           cast<TypedAttr>(fusedAttr));
}

//===----------------------------------------------------------------------===//
// RNN Sequence Rewrite
//===----------------------------------------------------------------------===//

// Rewrites the isolated unrolled sequence into a single loop over timesteps
// with loop-carried hidden state and fused recurrent parameters.
static LogicalResult rewriteRNNSequence(MatchedRNNSequence &match,
                                        unsigned &fusedBiasCounter,
                                        unsigned &fusedWeightCounter) {
  Location loc = match.executeRegion.getLoc();
  OpBuilder builder(match.executeRegion);

  std::string biasResourceName =
      "analog_rnn_fused_bias_" + std::to_string(fusedBiasCounter++);
  std::string weightResourceName =
      "analog_rnn_fused_weight_" + std::to_string(fusedWeightCounter++);

  FailureOr<arith::ConstantOp> fusedBiasConst =
      createFusedBiasResourceConstant(builder, loc, match, biasResourceName);
  FailureOr<arith::ConstantOp> fusedWeightConst =
      createFusedWeightResourceConstant(builder, loc, match, weightResourceName);
  if (failed(fusedBiasConst) || failed(fusedWeightConst))
    return failure();

  auto hiddenTy =
      RankedTensorType::get({1, match.hiddenSize}, builder.getF32Type());
  auto timestepInputTy =
      RankedTensorType::get({1, match.inputSize}, builder.getF32Type());
  auto transposedWeightTy = RankedTensorType::get(
      {match.inputSize + match.hiddenSize, match.hiddenSize},
      builder.getF32Type());

  auto transposedWeightEmpty = builder.create<tensor::EmptyOp>(
      loc, transposedWeightTy.getShape(), transposedWeightTy.getElementType());
  auto transposedWeight = builder.create<linalg::TransposeOp>(
      loc, (*fusedWeightConst).getResult(), transposedWeightEmpty,
      DenseI64ArrayAttr::get(builder.getContext(), {1, 0}));

  auto initialHiddenCollapsed = builder.create<tensor::CollapseShapeOp>(
      loc, hiddenTy, match.initialHidden,
      SmallVector<ReassociationIndices>{{0, 1}, {2}});

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value upper = builder.create<arith::ConstantIndexOp>(loc, match.sequenceLength);
  auto zeroScalar = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr(0.0f));

  auto loop = builder.create<scf::ForOp>(
      loc, c0, upper, c1, ValueRange(initialHiddenCollapsed.getResult()),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange iterArgs) {
        auto timestepInput3DTy = RankedTensorType::get(
            {1, 1, match.inputSize}, nestedBuilder.getF32Type());
        SmallVector<OpFoldResult> offsets = {nestedBuilder.getIndexAttr(0), iv,
                                             nestedBuilder.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes = {
            nestedBuilder.getIndexAttr(1), nestedBuilder.getIndexAttr(1),
            nestedBuilder.getIndexAttr(match.inputSize)};
        SmallVector<OpFoldResult> strides = {
            nestedBuilder.getIndexAttr(1), nestedBuilder.getIndexAttr(1),
            nestedBuilder.getIndexAttr(1)};
        Value timestepInput3D = nestedBuilder.create<tensor::ExtractSliceOp>(
            nestedLoc, timestepInput3DTy, match.sequenceInput, offsets, sizes,
            strides);
        Value timestepInput = nestedBuilder.create<tensor::CollapseShapeOp>(
            nestedLoc, timestepInputTy, timestepInput3D,
            SmallVector<ReassociationIndices>{{0, 1}, {2}});

        auto concatInput = nestedBuilder.create<tensor::ConcatOp>(
            nestedLoc, /*dim=*/1, ValueRange{timestepInput, iterArgs[0]});

        auto matmulInit = nestedBuilder.create<tensor::EmptyOp>(
            nestedLoc, hiddenTy.getShape(), hiddenTy.getElementType());
        auto filledInit =
            nestedBuilder
                .create<linalg::FillOp>(nestedLoc, ValueRange{zeroScalar},
                                        ValueRange{matmulInit})
                .getResult(0);
        auto matmul = nestedBuilder.create<linalg::MatmulOp>(
            nestedLoc, TypeRange{hiddenTy},
            ValueRange{concatInput, transposedWeight.getResult()[0]},
            ValueRange{filledInit});

        Value biasAdded = buildPointwiseAdd(
            nestedBuilder, nestedLoc, matmul.getResult(0),
            (*fusedBiasConst).getResult(), hiddenTy);
        Value activated =
            buildTanh(nestedBuilder, nestedLoc, biasAdded, hiddenTy);
        nestedBuilder.create<scf::YieldOp>(nestedLoc, ValueRange{activated});
      });

  match.executeRegion.getResult(0).replaceAllUsesWith(loop.getResult(0));
  match.executeRegion.erase();
  eraseIfDead(match.inputBiasConst);
  eraseIfDead(match.recurrentBiasConst);
  eraseIfDead(match.inputWeightConst);
  eraseIfDead(match.recurrentWeightConst);
  return success();
}

} // namespace

llvm::StringRef PrepareRNNForAnalogPass::getArgument() const {
  return "analog-prepare-rnn-for-analog";
}

llvm::StringRef PrepareRNNForAnalogPass::getDescription() const {
  return "Prepare isolated RNN sequences for loop-based analog execution";
}

void PrepareRNNForAnalogPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}

void PrepareRNNForAnalogPass::runOnOperation() {
  func::FuncOp func = getOperation();
  SmallVector<scf::ExecuteRegionOp> candidates;
  unsigned fusedBiasCounter = 0;
  unsigned fusedWeightCounter = 0;

  func.walk([&](scf::ExecuteRegionOp executeRegion) {
    candidates.push_back(executeRegion);
  });

  for (scf::ExecuteRegionOp executeRegion : candidates) {
    FailureOr<MatchedRNNSequence> match =
        matchRNNSequenceBlock(func, executeRegion);
    if (failed(match))
      continue;
    (void)rewriteRNNSequence(*match, fusedBiasCounter, fusedWeightCounter);
  }
}

std::unique_ptr<mlir::Pass> createPrepareRNNForAnalogPass() {
  return std::make_unique<PrepareRNNForAnalogPass>();
}

} // namespace analog
} // namespace mlir
