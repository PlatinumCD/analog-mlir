#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
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

namespace {

using mlir::arith::AddFOp;
using mlir::arith::ConstantOp;
using mlir::func::ReturnOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::math::TanhOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

struct SupportedRNNCellTypes {
  mlir::RankedTensorType hiddenStateTy;
  mlir::RankedTensorType inputStateTy;
  mlir::RankedTensorType outputTy;
};

struct SupportedRNNCellBranch {
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

struct SupportedRNNCellMatch {
  ReturnOp returnOp;
  GenericOp tanhOp;
  GenericOp preActivationAddOp;
  EmptyOp sharedOutputEmpty;
  FillOp sharedFillOp;
  ConstantOp sharedFillConstant;
  SupportedRNNCellBranch recurrentBranch;
  SupportedRNNCellBranch inputBranch;
  SupportedRNNCellTypes types;
  bool hasBias = false;
};

struct RNNCellLoweringState {
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

static bool hasExpectedBodyShape(GenericOp genericOp, unsigned inputCount) {
  if (!genericOp || genericOp.getInputs().size() != inputCount ||
      genericOp.getOutputs().size() != 1 || genericOp.getNumResults() != 1)
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;

  return llvm::isa<mlir::linalg::YieldOp>(body.getOperations().back());
}

static bool isAddfGeneric(GenericOp genericOp) {
  if (!hasExpectedBodyShape(genericOp, /*inputCount=*/2))
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  auto addOp = llvm::dyn_cast<AddFOp>(body.getOperations().front());
  auto yieldOp =
      llvm::dyn_cast<mlir::linalg::YieldOp>(body.getOperations().back());
  return addOp && yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == addOp.getResult();
}

static bool isTanhGeneric(GenericOp genericOp) {
  if (!hasExpectedBodyShape(genericOp, /*inputCount=*/1))
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  auto tanhOp = llvm::dyn_cast<TanhOp>(body.getOperations().front());
  auto yieldOp =
      llvm::dyn_cast<mlir::linalg::YieldOp>(body.getOperations().back());
  return tanhOp && yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == tanhOp.getResult();
}

static mlir::FailureOr<SupportedRNNCellTypes>
getSupportedRNNCellTypes(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 1)
    return mlir::failure();

  auto hiddenStateTy = getSupportedRank2TensorType(func.getArgument(0));
  auto inputStateTy = getSupportedRank2TensorType(func.getArgument(1));
  auto outputTy =
      llvm::dyn_cast<mlir::RankedTensorType>(functionType.getResult(0));
  if (failed(hiddenStateTy) || failed(inputStateTy) || !outputTy ||
      !outputTy.hasStaticShape() || outputTy.getRank() != 2 ||
      !outputTy.getElementType().isF32())
    return mlir::failure();

  if (outputTy.getShape()[0] != 1 || outputTy.getShape()[1] <= 0)
    return mlir::failure();

  if (hiddenStateTy->getShape()[1] != outputTy.getShape()[1])
    return mlir::failure();

  return SupportedRNNCellTypes{*hiddenStateTy, *inputStateTy, outputTy};
}

static mlir::FailureOr<std::tuple<ReturnOp, GenericOp, GenericOp, EmptyOp>>
getSupportedRNNCellRoot(mlir::func::FuncOp func,
                        mlir::RankedTensorType outputTy) {
  if (!func.getBody().hasOneBlock())
    return mlir::failure();

  auto returnOp = llvm::dyn_cast<ReturnOp>(func.front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return mlir::failure();

  auto tanhOp = returnOp.getOperand(0).getDefiningOp<GenericOp>();
  if (!isTanhGeneric(tanhOp))
    return mlir::failure();

  auto tanhResultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(tanhOp.getResult(0).getType());
  auto outputEmpty = tanhOp.getOutputs()[0].getDefiningOp<EmptyOp>();
  if (!tanhResultTy || tanhResultTy != outputTy || !outputEmpty)
    return mlir::failure();

  auto preActivationAddOp = tanhOp.getInputs()[0].getDefiningOp<GenericOp>();
  if (!isAddfGeneric(preActivationAddOp))
    return mlir::failure();

  auto preActivationTy = llvm::dyn_cast<mlir::RankedTensorType>(
      preActivationAddOp.getResult(0).getType());
  if (!preActivationTy || preActivationTy != outputTy)
    return mlir::failure();

  if (preActivationAddOp.getOutputs()[0].getDefiningOp<EmptyOp>() != outputEmpty)
    return mlir::failure();

  return std::make_tuple(returnOp, tanhOp, preActivationAddOp, outputEmpty);
}

static mlir::FailureOr<SupportedRNNCellBranch>
matchSupportedRNNCellBranch(mlir::Value branchValue, EmptyOp sharedOutputEmpty,
                            mlir::RankedTensorType outputTy) {
  SupportedRNNCellBranch branch;
  mlir::Value matmulResult = branchValue;

  auto maybeBiasAdd = branchValue.getDefiningOp<GenericOp>();
  if (maybeBiasAdd && isAddfGeneric(maybeBiasAdd)) {
    if (maybeBiasAdd.getOutputs()[0].getDefiningOp<EmptyOp>() !=
        sharedOutputEmpty)
      return mlir::failure();

    auto firstMatmul = maybeBiasAdd.getInputs()[0].getDefiningOp<MatmulOp>();
    auto secondMatmul = maybeBiasAdd.getInputs()[1].getDefiningOp<MatmulOp>();
    if (static_cast<bool>(firstMatmul) == static_cast<bool>(secondMatmul))
      return mlir::failure();

    auto matmulOp = firstMatmul ? firstMatmul : secondMatmul;
    mlir::Value bias =
        firstMatmul ? maybeBiasAdd.getInputs()[1] : maybeBiasAdd.getInputs()[0];
    auto biasTy = llvm::dyn_cast<mlir::RankedTensorType>(bias.getType());
    auto biasConstant = bias.getDefiningOp<ConstantOp>();
    if (!biasTy || !biasTy.hasStaticShape() || biasTy.getRank() != 1 ||
        !biasTy.getElementType().isF32() ||
        biasTy.getShape()[0] != outputTy.getShape()[1] || !biasConstant)
      return mlir::failure();

    branch.hasBias = true;
    branch.biasAddOp = maybeBiasAdd;
    branch.biasConstant = biasConstant;
    branch.bias = bias;
    branch.biasTy = biasTy;
    matmulResult = matmulOp.getResult(0);
  }

  auto matmulOp = matmulResult.getDefiningOp<MatmulOp>();
  if (!matmulOp || matmulOp.getInputs().size() != 2 ||
      matmulOp.getOutputs().size() != 1)
    return mlir::failure();

  auto sharedFillOp = matmulOp.getOutputs()[0].getDefiningOp<FillOp>();
  auto matmulResultTy =
      llvm::dyn_cast<mlir::RankedTensorType>(matmulOp.getResult(0).getType());
  if (!sharedFillOp || !matmulResultTy || matmulResultTy != outputTy ||
      sharedFillOp.getResult(0) != matmulOp.getOutputs()[0] ||
      sharedFillOp.getInputs().size() != 1 ||
      !isZeroF32Constant(sharedFillOp.getInputs()[0]))
    return mlir::failure();

  auto firstTranspose = matmulOp.getInputs()[0].getDefiningOp<TransposeOp>();
  auto secondTranspose = matmulOp.getInputs()[1].getDefiningOp<TransposeOp>();
  if (static_cast<bool>(firstTranspose) == static_cast<bool>(secondTranspose))
    return mlir::failure();

  auto transposeOp = firstTranspose ? firstTranspose : secondTranspose;
  mlir::Value activation =
      firstTranspose ? matmulOp.getInputs()[1] : matmulOp.getInputs()[0];
  auto activationTy = getSupportedRank2TensorType(activation);
  if (failed(activationTy))
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

  if (weightConstantTy.getShape()[0] != outputTy.getShape()[1] ||
      weightConstantTy.getShape()[1] != activationTy->getShape()[1])
    return mlir::failure();

  if (transposedWeightTy.getShape()[0] != activationTy->getShape()[1] ||
      transposedWeightTy.getShape()[1] != outputTy.getShape()[1])
    return mlir::failure();

  branch.activation = activation;
  branch.activationTy = *activationTy;
  branch.weightConstant = weightConstant;
  branch.weightConstantTy = weightConstantTy;
  branch.transposeInit = transposeInit;
  branch.transposeOp = transposeOp;
  branch.matmulOp = matmulOp;
  branch.sharedFillOp = sharedFillOp;
  branch.sharedFillValue = sharedFillOp.getInputs()[0];
  return branch;
}

static mlir::FailureOr<SupportedRNNCellMatch>
matchSupportedRNNCell(mlir::func::FuncOp func) {
  auto types = getSupportedRNNCellTypes(func);
  if (failed(types))
    return mlir::failure();

  auto root = getSupportedRNNCellRoot(func, types->outputTy);
  if (failed(root))
    return mlir::failure();

  auto firstBranch = matchSupportedRNNCellBranch(
      std::get<2>(*root).getInputs()[0], std::get<3>(*root), types->outputTy);
  auto secondBranch = matchSupportedRNNCellBranch(
      std::get<2>(*root).getInputs()[1], std::get<3>(*root), types->outputTy);
  if (failed(firstBranch) || failed(secondBranch))
    return mlir::failure();

  if (firstBranch->sharedFillOp != secondBranch->sharedFillOp)
    return mlir::failure();

  auto sharedFillConstant =
      firstBranch->sharedFillValue.getDefiningOp<ConstantOp>();
  if (!sharedFillConstant)
    return mlir::failure();

  bool expectBias = converter_utils::hasLayerType(func, "rnn_cell_w_bias");
  bool expectNoBias = converter_utils::hasLayerType(func, "rnn_cell");
  if (expectBias == expectNoBias)
    return mlir::failure();

  if (firstBranch->hasBias != secondBranch->hasBias)
    return mlir::failure();

  if (expectBias && !firstBranch->hasBias)
    return mlir::failure();

  if (expectNoBias && firstBranch->hasBias)
    return mlir::failure();

  if (firstBranch->activation == func.getArgument(0) &&
      secondBranch->activation == func.getArgument(1)) {
    return SupportedRNNCellMatch{std::get<0>(*root),
                                 std::get<1>(*root),
                                 std::get<2>(*root),
                                 std::get<3>(*root),
                                 firstBranch->sharedFillOp,
                                 sharedFillConstant,
                                 *firstBranch,
                                 *secondBranch,
                                 *types,
                                 firstBranch->hasBias};
  }

  if (firstBranch->activation == func.getArgument(1) &&
      secondBranch->activation == func.getArgument(0)) {
    return SupportedRNNCellMatch{std::get<0>(*root),
                                 std::get<1>(*root),
                                 std::get<2>(*root),
                                 std::get<3>(*root),
                                 firstBranch->sharedFillOp,
                                 sharedFillConstant,
                                 *secondBranch,
                                 *firstBranch,
                                 *types,
                                 firstBranch->hasBias};
  }

  return mlir::failure();
}

static mlir::RankedTensorType buildFusedWeightType(SupportedRNNCellMatch &match) {
  return mlir::RankedTensorType::get(
      {match.types.outputTy.getShape()[1],
       match.types.inputStateTy.getShape()[1] +
           match.types.hiddenStateTy.getShape()[1]},
      match.types.outputTy.getElementType());
}

static mlir::RankedTensorType buildFusedInputType(SupportedRNNCellMatch &match) {
  return mlir::RankedTensorType::get(
      {1, match.types.inputStateTy.getShape()[1] +
              match.types.hiddenStateTy.getShape()[1]},
      match.types.outputTy.getElementType());
}

static mlir::RankedTensorType buildFusedBiasType(SupportedRNNCellMatch &match) {
  return mlir::RankedTensorType::get({match.types.outputTy.getShape()[1]},
                                     match.types.outputTy.getElementType());
}

static RNNCellLoweringState buildRNNCellLoweringState(
    mlir::OpBuilder &builder, SupportedRNNCellMatch &match) {
  mlir::Location loc = match.returnOp.getLoc();
  mlir::Type elementType = match.types.outputTy.getElementType();
  mlir::RankedTensorType fusedInputTy = buildFusedInputType(match);
  mlir::RankedTensorType fusedWeightTy = buildFusedWeightType(match);
  mlir::RankedTensorType fusedBiasTy = buildFusedBiasType(match);

  return RNNCellLoweringState{
      loc,
      elementType,
      fusedInputTy,
      fusedWeightTy,
      fusedBiasTy,
      match.types.outputTy,
      match.types.outputTy,
      builder.create<mlir::arith::ConstantFloatOp>(
          loc, llvm::cast<mlir::FloatType>(elementType), llvm::APFloat(0.0f)),
  };
}

static mlir::TypedAttr buildFusedWeightAttr(SupportedRNNCellMatch &match,
                                            mlir::RankedTensorType fusedTy) {
  auto maybeInputWeights = getF32ConstantValues(match.inputBranch.weightConstant);
  auto maybeRecurrentWeights =
      getF32ConstantValues(match.recurrentBranch.weightConstant);
  if (failed(maybeInputWeights) || failed(maybeRecurrentWeights))
    return {};

  llvm::SmallVector<float> inputWeights = *maybeInputWeights;
  llvm::SmallVector<float> recurrentWeights = *maybeRecurrentWeights;
  int64_t outputSize = fusedTy.getShape()[0];
  int64_t inputWidth = match.inputBranch.weightConstantTy.getShape()[1];
  int64_t hiddenWidth = match.recurrentBranch.weightConstantTy.getShape()[1];
  int64_t fusedWidth = fusedTy.getShape()[1];
  llvm::SmallVector<float> fusedWeights(fusedTy.getNumElements(), 0.0f);

  for (int64_t row = 0; row < outputSize; ++row) {
    int64_t inputOffset = row * inputWidth;
    int64_t recurrentOffset = row * hiddenWidth;
    int64_t fusedOffset = row * fusedWidth;
    for (int64_t col = 0; col < inputWidth; ++col)
      fusedWeights[fusedOffset + col] = inputWeights[inputOffset + col];
    for (int64_t col = 0; col < hiddenWidth; ++col)
      fusedWeights[fusedOffset + inputWidth + col] =
          recurrentWeights[recurrentOffset + col];
  }

  bool useResource = isResourceBackedF32Constant(match.inputBranch.weightConstant) ||
                     isResourceBackedF32Constant(
                         match.recurrentBranch.weightConstant);
  if (useResource) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        "analog_rnn_cell_fused_weight_" + std::to_string(nextResourceId++);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(fusedWeights), /*dataIsMutable=*/false);
    return llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
        fusedTy, resourceName, std::move(blob)));
  }

  return llvm::cast<mlir::TypedAttr>(
      mlir::DenseElementsAttr::get(fusedTy, llvm::ArrayRef<float>(fusedWeights)));
}

static mlir::TypedAttr buildFusedBiasAttr(SupportedRNNCellMatch &match,
                                          mlir::RankedTensorType fusedTy) {
  auto maybeInputBias = getF32ConstantValues(match.inputBranch.biasConstant);
  auto maybeRecurrentBias =
      getF32ConstantValues(match.recurrentBranch.biasConstant);
  if (failed(maybeInputBias) || failed(maybeRecurrentBias))
    return {};

  llvm::SmallVector<float> inputBias = *maybeInputBias;
  llvm::SmallVector<float> recurrentBias = *maybeRecurrentBias;
  llvm::SmallVector<float> fusedBias(fusedTy.getNumElements(), 0.0f);

  for (int64_t idx = 0; idx < fusedTy.getShape()[0]; ++idx)
    fusedBias[idx] = inputBias[idx] + recurrentBias[idx];

  bool useResource = isResourceBackedF32Constant(match.inputBranch.biasConstant) ||
                     isResourceBackedF32Constant(match.recurrentBranch.biasConstant);
  if (useResource) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        "analog_rnn_cell_fused_bias_" + std::to_string(nextResourceId++);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(fusedBias), /*dataIsMutable=*/false);
    return llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
        fusedTy, resourceName, std::move(blob)));
  }

  return llvm::cast<mlir::TypedAttr>(
      mlir::DenseElementsAttr::get(fusedTy, llvm::ArrayRef<float>(fusedBias)));
}

static mlir::FailureOr<ConstantOp>
createFusedWeightConstant(SupportedRNNCellMatch &match,
                          mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType fusedTy = buildFusedWeightType(match);
  mlir::TypedAttr fusedAttr = buildFusedWeightAttr(match, fusedTy);
  if (!fusedAttr)
    return mlir::failure();

  mlir::Operation *insertAfter =
      getLaterOp(match.inputBranch.weightConstant.getOperation(),
                 match.recurrentBranch.weightConstant.getOperation());
  rewriter.setInsertionPointAfter(insertAfter);
  return rewriter.create<ConstantOp>(match.returnOp.getLoc(), fusedTy, fusedAttr);
}

static mlir::FailureOr<ConstantOp>
createFusedBiasConstant(SupportedRNNCellMatch &match,
                        mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType fusedTy = buildFusedBiasType(match);
  mlir::TypedAttr fusedAttr = buildFusedBiasAttr(match, fusedTy);
  if (!fusedAttr)
    return mlir::failure();

  mlir::Operation *insertAfter =
      getLaterOp(match.inputBranch.biasConstant.getOperation(),
                 match.recurrentBranch.biasConstant.getOperation());
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
prepareFusedWeight(SupportedRNNCellMatch &match,
                   const RNNCellLoweringState &state,
                   mlir::RewriterBase &rewriter, int64_t arrayRows,
                   int64_t arrayCols) {
  auto fusedWeightConstant = createFusedWeightConstant(match, rewriter);
  if (failed(fusedWeightConstant))
    return mlir::failure();

  auto analogMatrix =
      converter_utils::materializeAnalogMatrix(*fusedWeightConstant, rewriter);
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
      llvm::ArrayRef<int64_t>{state.fusedInputTy.getShape()[1],
                              state.outputTy.getShape()[1]},
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
prepareFusedBias(SupportedRNNCellMatch &match, mlir::RewriterBase &rewriter) {
  PreparedFusedBias preparedBias;
  if (!match.hasBias)
    return preparedBias;

  auto fusedBiasConstant = createFusedBiasConstant(match, rewriter);
  if (failed(fusedBiasConstant))
    return mlir::failure();

  preparedBias.fusedBiasConstant = *fusedBiasConstant;
  preparedBias.bias = fusedBiasConstant->getResult();
  return preparedBias;
}

static mlir::FailureOr<PreparedFusedVector>
prepareFusedVector(SupportedRNNCellMatch &match,
                   PreparedFusedWeight &preparedWeight,
                   const RNNCellLoweringState &state,
                   mlir::RewriterBase &rewriter, int64_t arrayRows,
                   int64_t arrayCols) {
  rewriter.setInsertionPointAfter(preparedWeight.transposeOp.getOperation());
  auto concatOp = rewriter.create<ConcatOp>(
      state.loc, state.fusedInputTy, /*dim=*/1,
      mlir::ValueRange{match.inputBranch.activation,
                       match.recurrentBranch.activation});

  auto analogVector = converter_utils::materializeAnalogVector(
      concatOp.getResult(), preparedWeight.matrixId, rewriter);
  if (failed(analogVector))
    return mlir::failure();

  auto partitionedVector = converter_utils::partitionAnalogVector(
      *analogVector, rewriter, arrayRows, arrayCols);
  if (failed(partitionedVector))
    return mlir::failure();

  auto placementLoop =
      converter_utils::placeAnalogVector(*partitionedVector, rewriter);
  if (failed(placementLoop))
    return mlir::failure();

  return PreparedFusedVector{concatOp.getResult(), *partitionedVector,
                             *placementLoop};
}

static MatmulOp buildFusedMatmulScaffold(
    mlir::OpBuilder &builder, const RNNCellLoweringState &state,
    mlir::Value fusedInput, mlir::Value transposedWeight) {
  mlir::Value matmulInit = buildZeroInitializedTensor(
      builder, state.loc, state.matmulResultTy, state.zeroValue);
  return builder.create<MatmulOp>(state.loc, state.matmulResultTy,
                                  mlir::ValueRange{fusedInput, transposedWeight},
                                  mlir::ValueRange{matmulInit});
}

static mlir::Value applyOptionalFusedBias(
    mlir::OpBuilder &builder, const RNNCellLoweringState &state,
    const PreparedFusedBias &preparedBias, mlir::Value channelResult) {
  if (!preparedBias.bias)
    return channelResult;

  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1}};
  mlir::Value expandedBias = builder.create<ExpandShapeOp>(
      state.loc, state.matmulResultTy, preparedBias.bias, reassociation);
  mlir::Value biasedInit = builder.create<EmptyOp>(
      state.loc, state.matmulResultTy.getShape(), state.elementType);
  return builder
      .create<mlir::linalg::AddOp>(
          state.loc, mlir::ValueRange{channelResult, expandedBias},
          mlir::ValueRange{biasedInit})
      .getResult(0);
}

static GenericOp cloneTanhWithInput(mlir::RewriterBase &rewriter,
                                    SupportedRNNCellMatch &match,
                                    const RNNCellLoweringState &state,
                                    mlir::Value tanhInput) {
  auto tanhInit = rewriter.create<EmptyOp>(state.loc, state.outputTy.getShape(),
                                           state.elementType);
  mlir::IRMapping mapping;
  mapping.map(match.tanhOp.getInputs()[0], tanhInput);
  mapping.map(match.tanhOp.getOutputs()[0], tanhInit.getResult());
  return llvm::cast<GenericOp>(
      rewriter.clone(*match.tanhOp.getOperation(), mapping));
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

static void eraseUnusedOriginalRNNCellOps(SupportedRNNCellMatch &match,
                                          mlir::RewriterBase &rewriter) {
  eraseIfUnused(match.tanhOp.getOperation(), rewriter);
  eraseIfUnused(match.preActivationAddOp.getOperation(), rewriter);

  if (match.hasBias) {
    eraseIfUnused(match.recurrentBranch.biasAddOp.getOperation(), rewriter);
    eraseIfUnused(match.inputBranch.biasAddOp.getOperation(), rewriter);
    eraseIfUnused(match.recurrentBranch.biasConstant.getOperation(), rewriter);
    eraseIfUnused(match.inputBranch.biasConstant.getOperation(), rewriter);
  }

  eraseIfUnused(match.recurrentBranch.matmulOp.getOperation(), rewriter);
  eraseIfUnused(match.inputBranch.matmulOp.getOperation(), rewriter);
  eraseIfUnused(match.sharedFillOp.getOperation(), rewriter);

  eraseIfUnused(match.recurrentBranch.transposeOp.getOperation(), rewriter);
  eraseIfUnused(match.recurrentBranch.transposeInit.getOperation(), rewriter);
  eraseIfUnused(match.inputBranch.transposeOp.getOperation(), rewriter);
  eraseIfUnused(match.inputBranch.transposeInit.getOperation(), rewriter);

  eraseIfUnused(match.recurrentBranch.weightConstant.getOperation(), rewriter);
  eraseIfUnused(match.inputBranch.weightConstant.getOperation(), rewriter);

  eraseIfUnused(match.sharedOutputEmpty.getOperation(), rewriter);
  eraseIfUnused(match.sharedFillConstant.getOperation(), rewriter);
}

// Converts extracted RNN cell bodies into one fused analog matrix-vector path.
class RNNCellConverter : public mlir::analog::LayerConverter {
public:
  mlir::StringRef getName() const override { return "rnn_cell"; }

  void convert(mlir::func::FuncOp func, int64_t arrayRows,
               int64_t arrayCols) const override {
    if (arrayRows <= 0 || arrayCols <= 0)
      return;

    mlir::IRRewriter rewriter(func.getContext());
    auto match = matchSupportedRNNCell(func);
    if (failed(match))
      return;

    rewriter.setInsertionPointAfter(
        getLaterOp(match->inputBranch.weightConstant.getOperation(),
                   match->recurrentBranch.weightConstant.getOperation()));
    RNNCellLoweringState state = buildRNNCellLoweringState(rewriter, *match);

    auto preparedWeight =
        prepareFusedWeight(*match, state, rewriter, arrayRows, arrayCols);
    if (failed(preparedWeight))
      return;

    auto preparedBias = prepareFusedBias(*match, rewriter);
    if (failed(preparedBias))
      return;

    auto preparedVector =
        prepareFusedVector(*match, *preparedWeight, state, rewriter, arrayRows,
                           arrayCols);
    if (failed(preparedVector))
      return;

    auto executionBuffer = converter_utils::insertArrayExecution(
        preparedWeight->partitionedMatrix, preparedVector->partitionedVector,
        preparedWeight->placementLoop, preparedVector->placementLoop, rewriter);
    if (failed(executionBuffer))
      return;

    rewriter.setInsertionPoint(match->returnOp);
    MatmulOp fusedMatmul = buildFusedMatmulScaffold(
        rewriter, state, preparedVector->fusedInput,
        preparedWeight->transposedWeight);
    mlir::Value tanhInput =
        applyOptionalFusedBias(rewriter, state, *preparedBias,
                               fusedMatmul.getResult(0));
    GenericOp fusedTanh = cloneTanhWithInput(rewriter, *match, state, tanhInput);

    auto reducedTensor = converter_utils::insertArrayReduction(
        *executionBuffer, preparedWeight->partitionedMatrix, fusedMatmul,
        rewriter);
    if (failed(reducedTensor))
      return;

    match->tanhOp.getResult(0).replaceAllUsesWith(fusedTanh.getResult(0));
    eraseUnusedFusedMatmulScaffold(fusedMatmul, rewriter);
    eraseUnusedPreparedFusedWeightOps(*preparedWeight, rewriter);
    eraseUnusedOriginalRNNCellOps(*match, rewriter);
    func->setAttr("layer_domain", rewriter.getStringAttr("analog"));
  }
};

} // namespace

namespace mlir {
namespace analog {

void registerRNNCellConverter(LayerConverters &converters,
                              LayerConverterMap &converterMap,
                              MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<RNNCellConverter>();
  const LayerConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["rnn_cell"] = converterPtr;
  converterMap["rnn_cell_w_bias"] = converterPtr;
}

} // namespace analog
} // namespace mlir
