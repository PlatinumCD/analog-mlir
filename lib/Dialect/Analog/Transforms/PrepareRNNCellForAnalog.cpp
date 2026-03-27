#include "analog-mlir/Dialect/Analog/Transforms/PrepareRNNCellForAnalog.h"

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

static constexpr llvm::StringLiteral kRecurrentPatternAttr =
    "analog.recurrent_pattern";
static constexpr llvm::StringLiteral kRecurrentInputSizeAttr =
    "analog.recurrent_input_size";
static constexpr llvm::StringLiteral kRecurrentHiddenSizeAttr =
    "analog.recurrent_hidden_size";
static constexpr llvm::StringLiteral kPreparedForAnalogAttr =
    "analog.prepared_for_analog";
static constexpr llvm::StringLiteral kRecurrentAffineFusedAttr =
    "analog.recurrent_affine_fused";

struct MatchedRNNCell {
  scf::ExecuteRegionOp executeRegion;
  Value currentInput;
  Value hiddenInput;
  arith::ConstantOp inputWeightConst;
  arith::ConstantOp hiddenWeightConst;
  arith::ConstantOp inputBiasConst;
  arith::ConstantOp hiddenBiasConst;
  RankedTensorType currentInputTy;
  RankedTensorType hiddenInputTy;
  RankedTensorType outputTy;
  RankedTensorType inputWeightTy;
  RankedTensorType hiddenWeightTy;
  RankedTensorType inputBiasTy;
  RankedTensorType hiddenBiasTy;
};

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

static bool isTanhGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;

  Block &body = generic.getRegion().front();
  auto it = body.begin();
  auto tanhOp = dyn_cast<math::TanhOp>(*it++);
  if (!tanhOp)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(*it++);
  if (!yieldOp || it != body.end())
    return false;
  return yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == tanhOp.getResult();
}

static arith::ConstantOp getRank2ConstantThroughTranspose(Value value) {
  auto transpose = value.getDefiningOp<linalg::TransposeOp>();
  if (!transpose)
    return {};
  auto permutation = transpose.getPermutation();
  if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
    return {};
  return transpose.getInput().getDefiningOp<arith::ConstantOp>();
}

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

static FailureOr<MatchedRNNCell> matchRNNCellBlock(scf::ExecuteRegionOp executeRegion) {
  if (!executeRegion->hasAttr(kRecurrentPatternAttr) ||
      executeRegion->getAttrOfType<StringAttr>(kRecurrentPatternAttr).getValue() !=
          "rnn_cell" ||
      executeRegion->hasAttr(kPreparedForAnalogAttr)) {
    return failure();
  }

  if (!executeRegion.getRegion().hasOneBlock())
    return failure();

  Block &block = executeRegion.getRegion().front();
  auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return failure();

  auto activationOp = yieldOp.getOperand(0).getDefiningOp<linalg::GenericOp>();
  if (!isTanhGeneric(activationOp))
    return failure();

  auto outputTy = dyn_cast<RankedTensorType>(activationOp.getResult(0).getType());
  if (!outputTy || !outputTy.hasStaticShape() || outputTy.getRank() != 2 ||
      outputTy.getShape()[0] != 1 || !outputTy.getElementType().isF32())
    return failure();

  auto mergeAdd =
      activationOp.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
  if (!isPointwiseAddGeneric(mergeAdd))
    return failure();

  auto lhsBiasAdd =
      mergeAdd.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
  auto rhsBiasAdd =
      mergeAdd.getDpsInputOperand(1)->get().getDefiningOp<linalg::GenericOp>();
  if (!isPointwiseAddGeneric(lhsBiasAdd) || !isPointwiseAddGeneric(rhsBiasAdd))
    return failure();

  auto lhsMatmul =
      lhsBiasAdd.getDpsInputOperand(0)->get().getDefiningOp<linalg::MatmulOp>();
  auto rhsMatmul =
      rhsBiasAdd.getDpsInputOperand(0)->get().getDefiningOp<linalg::MatmulOp>();
  if (!lhsMatmul || !rhsMatmul)
    return failure();

  auto lhsBiasConst =
      lhsBiasAdd.getDpsInputOperand(1)->get().getDefiningOp<arith::ConstantOp>();
  auto rhsBiasConst =
      rhsBiasAdd.getDpsInputOperand(1)->get().getDefiningOp<arith::ConstantOp>();
  if (!lhsBiasConst || !rhsBiasConst)
    return failure();

  auto lhsWeightConst = getRank2ConstantThroughTranspose(lhsMatmul.getInputs()[1]);
  auto rhsWeightConst = getRank2ConstantThroughTranspose(rhsMatmul.getInputs()[1]);
  if (!lhsWeightConst || !rhsWeightConst)
    return failure();

  Value lhsInput = lhsMatmul.getInputs()[0];
  Value rhsInput = rhsMatmul.getInputs()[0];
  auto lhsInputTy = dyn_cast<RankedTensorType>(lhsInput.getType());
  auto rhsInputTy = dyn_cast<RankedTensorType>(rhsInput.getType());
  auto lhsWeightTy = dyn_cast<RankedTensorType>(lhsWeightConst.getType());
  auto rhsWeightTy = dyn_cast<RankedTensorType>(rhsWeightConst.getType());
  auto lhsBiasTy = dyn_cast<RankedTensorType>(lhsBiasConst.getType());
  auto rhsBiasTy = dyn_cast<RankedTensorType>(rhsBiasConst.getType());
  if (!lhsInputTy || !rhsInputTy || !lhsWeightTy || !rhsWeightTy ||
      !lhsBiasTy || !rhsBiasTy)
    return failure();
  if (!lhsInputTy.hasStaticShape() || !rhsInputTy.hasStaticShape() ||
      !lhsWeightTy.hasStaticShape() || !rhsWeightTy.hasStaticShape() ||
      !lhsBiasTy.hasStaticShape() || !rhsBiasTy.hasStaticShape())
    return failure();

  if (lhsInputTy.getRank() != 2 || rhsInputTy.getRank() != 2 ||
      lhsWeightTy.getRank() != 2 || rhsWeightTy.getRank() != 2 ||
      lhsBiasTy.getRank() != 1 || rhsBiasTy.getRank() != 1)
    return failure();

  int64_t hiddenSize = outputTy.getShape()[1];
  int64_t expectedInputSize =
      executeRegion->getAttrOfType<IntegerAttr>(kRecurrentInputSizeAttr)
          ? executeRegion->getAttrOfType<IntegerAttr>(kRecurrentInputSizeAttr)
                .getInt()
          : -1;
  int64_t expectedHiddenSize =
      executeRegion->getAttrOfType<IntegerAttr>(kRecurrentHiddenSizeAttr)
          ? executeRegion->getAttrOfType<IntegerAttr>(kRecurrentHiddenSizeAttr)
                .getInt()
          : hiddenSize;

  Value hiddenInput = lhsInput;
  Value currentInput = rhsInput;
  RankedTensorType hiddenInputTy = lhsInputTy;
  RankedTensorType currentInputTy = rhsInputTy;
  arith::ConstantOp hiddenWeightConst = lhsWeightConst;
  arith::ConstantOp inputWeightConst = rhsWeightConst;
  RankedTensorType hiddenWeightTy = lhsWeightTy;
  RankedTensorType inputWeightTy = rhsWeightTy;
  arith::ConstantOp hiddenBiasConst = lhsBiasConst;
  arith::ConstantOp inputBiasConst = rhsBiasConst;
  RankedTensorType hiddenBiasTy = lhsBiasTy;
  RankedTensorType inputBiasTy = rhsBiasTy;

  auto branchLooksHidden = [&](RankedTensorType inputTy, RankedTensorType weightTy) {
    return inputTy.getShape()[0] == 1 && inputTy.getShape()[1] == expectedHiddenSize &&
           weightTy.getShape()[0] == hiddenSize &&
           weightTy.getShape()[1] == expectedHiddenSize;
  };

  bool lhsHidden = branchLooksHidden(lhsInputTy, lhsWeightTy);
  bool rhsHidden = branchLooksHidden(rhsInputTy, rhsWeightTy);
  if (!lhsHidden && rhsHidden) {
    hiddenInput = rhsInput;
    currentInput = lhsInput;
    hiddenInputTy = rhsInputTy;
    currentInputTy = lhsInputTy;
    hiddenWeightConst = rhsWeightConst;
    inputWeightConst = lhsWeightConst;
    hiddenWeightTy = rhsWeightTy;
    inputWeightTy = lhsWeightTy;
    hiddenBiasConst = rhsBiasConst;
    inputBiasConst = lhsBiasConst;
    hiddenBiasTy = rhsBiasTy;
    inputBiasTy = lhsBiasTy;
  } else if (!lhsHidden && !rhsHidden) {
    return failure();
  }

  if (hiddenInputTy.getShape()[0] != 1 || currentInputTy.getShape()[0] != 1 ||
      hiddenInputTy.getShape()[1] != expectedHiddenSize)
    return failure();
  if (expectedInputSize >= 0 && currentInputTy.getShape()[1] != expectedInputSize)
    return failure();
  if (hiddenWeightTy.getShape()[0] != hiddenSize ||
      hiddenWeightTy.getShape()[1] != hiddenInputTy.getShape()[1] ||
      inputWeightTy.getShape()[0] != hiddenSize ||
      inputWeightTy.getShape()[1] != currentInputTy.getShape()[1])
    return failure();
  if (hiddenBiasTy.getShape()[0] != hiddenSize || inputBiasTy.getShape()[0] != hiddenSize)
    return failure();

  MatchedRNNCell match;
  match.executeRegion = executeRegion;
  match.currentInput = currentInput;
  match.hiddenInput = hiddenInput;
  match.inputWeightConst = inputWeightConst;
  match.hiddenWeightConst = hiddenWeightConst;
  match.inputBiasConst = inputBiasConst;
  match.hiddenBiasConst = hiddenBiasConst;
  match.currentInputTy = currentInputTy;
  match.hiddenInputTy = hiddenInputTy;
  match.outputTy = outputTy;
  match.inputWeightTy = inputWeightTy;
  match.hiddenWeightTy = hiddenWeightTy;
  match.inputBiasTy = inputBiasTy;
  match.hiddenBiasTy = hiddenBiasTy;
  return match;
}

static FailureOr<arith::ConstantOp> createFusedBiasResourceConstant(
    OpBuilder &builder, Location loc, MatchedRNNCell &match,
    StringRef resourceName) {
  FailureOr<SmallVector<float>> inputBiasValues =
      extractFloatElements(match.inputBiasConst);
  FailureOr<SmallVector<float>> hiddenBiasValues =
      extractFloatElements(match.hiddenBiasConst);
  if (failed(inputBiasValues) || failed(hiddenBiasValues))
    return failure();

  int64_t hiddenSize = match.outputTy.getShape()[1];
  SmallVector<float> fusedValues;
  fusedValues.reserve(hiddenSize);
  for (int64_t i = 0; i < hiddenSize; ++i)
    fusedValues.push_back((*inputBiasValues)[i] + (*hiddenBiasValues)[i]);

  auto fusedTy = RankedTensorType::get({hiddenSize}, builder.getF32Type());
  Attribute fusedAttr = DenseF32ResourceElementsAttr::get(
      fusedTy, resourceName,
      HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(fusedValues));
  return builder.create<arith::ConstantOp>(loc, fusedTy,
                                           cast<TypedAttr>(fusedAttr));
}

static FailureOr<arith::ConstantOp> createFusedWeightResourceConstant(
    OpBuilder &builder, Location loc, MatchedRNNCell &match,
    StringRef resourceName) {
  FailureOr<SmallVector<float>> inputWeightValues =
      extractFloatElements(match.inputWeightConst);
  FailureOr<SmallVector<float>> hiddenWeightValues =
      extractFloatElements(match.hiddenWeightConst);
  if (failed(inputWeightValues) || failed(hiddenWeightValues))
    return failure();

  int64_t hiddenSize = match.outputTy.getShape()[1];
  int64_t inputSize = match.currentInputTy.getShape()[1];
  int64_t recurrentSize = match.hiddenInputTy.getShape()[1];
  SmallVector<float> fusedValues;
  fusedValues.reserve(hiddenSize * (inputSize + recurrentSize));
  for (int64_t row = 0; row < hiddenSize; ++row) {
    int64_t inputBase = row * inputSize;
    fusedValues.append(inputWeightValues->begin() + inputBase,
                       inputWeightValues->begin() + inputBase + inputSize);
    int64_t hiddenBase = row * recurrentSize;
    fusedValues.append(hiddenWeightValues->begin() + hiddenBase,
                       hiddenWeightValues->begin() + hiddenBase + recurrentSize);
  }

  auto fusedTy = RankedTensorType::get({hiddenSize, inputSize + recurrentSize},
                                       builder.getF32Type());
  Attribute fusedAttr = DenseF32ResourceElementsAttr::get(
      fusedTy, resourceName,
      HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(fusedValues));
  return builder.create<arith::ConstantOp>(loc, fusedTy,
                                           cast<TypedAttr>(fusedAttr));
}

static void eraseIfDead(arith::ConstantOp op) {
  if (op && op->use_empty())
    op.erase();
}

static LogicalResult rewriteRNNCell(MatchedRNNCell &match,
                                    unsigned &fusedBiasCounter,
                                    unsigned &fusedWeightCounter) {
  Location loc = match.executeRegion.getLoc();
  OpBuilder builder(match.executeRegion);
  std::string biasResourceName =
      "analog_rnn_cell_fused_bias_" + std::to_string(fusedBiasCounter++);
  std::string weightResourceName =
      "analog_rnn_cell_fused_weight_" + std::to_string(fusedWeightCounter++);
  FailureOr<arith::ConstantOp> fusedBiasConst =
      createFusedBiasResourceConstant(builder, loc, match, biasResourceName);
  FailureOr<arith::ConstantOp> fusedWeightConst =
      createFusedWeightResourceConstant(builder, loc, match, weightResourceName);
  if (failed(fusedBiasConst) || failed(fusedWeightConst))
    return failure();

  auto concatInput = builder.create<tensor::ConcatOp>(
      loc, /*dim=*/1, ValueRange{match.currentInput, match.hiddenInput});

  int64_t hiddenSize = match.outputTy.getShape()[1];
  int64_t fusedInputSize = match.currentInputTy.getShape()[1] +
                           match.hiddenInputTy.getShape()[1];
  auto transposedWeightTy = RankedTensorType::get(
      {fusedInputSize, hiddenSize}, builder.getF32Type());
  auto transposedWeightEmpty = builder.create<tensor::EmptyOp>(
      loc, transposedWeightTy.getShape(), transposedWeightTy.getElementType());
  auto transposedWeight = builder.create<linalg::TransposeOp>(
      loc, (*fusedWeightConst).getResult(), transposedWeightEmpty,
      DenseI64ArrayAttr::get(builder.getContext(), {1, 0}));

  auto zeroValue = builder.create<arith::ConstantOp>(
      loc, builder.getF32FloatAttr(0.0f));
  auto matmulInit = builder.create<tensor::EmptyOp>(
      loc, match.outputTy.getShape(), match.outputTy.getElementType());
  auto filledInit =
      builder.create<linalg::FillOp>(loc, ValueRange{zeroValue},
                                     ValueRange{matmulInit})
          .getResult(0);
  auto matmul = builder.create<linalg::MatmulOp>(
      loc, TypeRange{match.outputTy}, ValueRange{concatInput, transposedWeight.getResult()[0]},
      ValueRange{filledInit});

  Value biasAdded =
      buildPointwiseAdd(builder, loc, matmul.getResult(0),
                        (*fusedBiasConst).getResult(), match.outputTy);
  Value activated = buildTanh(builder, loc, biasAdded, match.outputTy);

  match.executeRegion.getResult(0).replaceAllUsesWith(activated);
  match.executeRegion.erase();
  eraseIfDead(match.inputBiasConst);
  eraseIfDead(match.hiddenBiasConst);
  eraseIfDead(match.inputWeightConst);
  eraseIfDead(match.hiddenWeightConst);
  return success();
}

} // namespace

llvm::StringRef PrepareRNNCellForAnalogPass::getArgument() const {
  return "analog-prepare-rnn-cell-for-analog";
}

llvm::StringRef PrepareRNNCellForAnalogPass::getDescription() const {
  return "Prepare isolated RNN cell blocks for analog execution";
}

void PrepareRNNCellForAnalogPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}

void PrepareRNNCellForAnalogPass::runOnOperation() {
  SmallVector<scf::ExecuteRegionOp> candidates;
  unsigned fusedBiasCounter = 0;
  unsigned fusedWeightCounter = 0;
  getOperation().walk([&](scf::ExecuteRegionOp executeRegion) {
    candidates.push_back(executeRegion);
  });

  for (scf::ExecuteRegionOp executeRegion : candidates) {
    FailureOr<MatchedRNNCell> match = matchRNNCellBlock(executeRegion);
    if (failed(match))
      continue;
    (void)rewriteRNNCell(*match, fusedBiasCounter, fusedWeightCounter);
  }
}

std::unique_ptr<mlir::Pass> createPrepareRNNCellForAnalogPass() {
  return std::make_unique<PrepareRNNCellForAnalogPass>();
}

} // namespace analog
} // namespace mlir
