#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/converters/ConverterUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <optional>
#include <utility>

namespace converter_utils = mlir::analog::converter_utils;

namespace {

using mlir::arith::ConstantOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::EmptyOp;

// Carries the partitioned weight grid and the loop that places it on arrays.
struct ConvertedMatrix {
  mlir::Value partitionedMatrix;
  mlir::scf::ForOp placementLoop;
};

// Carries the partitioned input slices and the loop that places them on arrays.
struct ConvertedVector {
  mlir::Value partitionedVector;
  mlir::scf::ForOp placementLoop;
};

// Lowers one candidate weight constant through analog materialization,
// partitioning, and placement.
static std::optional<ConvertedMatrix> convertMatrixConstant(
    ConstantOp constant, mlir::RewriterBase &rewriter, int64_t arrayRows,
    int64_t arrayCols) {
  auto analogMatrix =
      converter_utils::materializeAnalogMatrix(constant, rewriter);
  if (failed(analogMatrix))
    return std::nullopt;

  auto partitionedMatrix = converter_utils::partitionAnalogMatrix(
      *analogMatrix, rewriter, arrayRows, arrayCols);
  if (failed(partitionedMatrix))
    return std::nullopt;

  auto placementLoop =
      converter_utils::placeAnalogMatrix(*partitionedMatrix, rewriter);
  if (failed(placementLoop))
    return std::nullopt;

  return ConvertedMatrix{*partitionedMatrix, *placementLoop};
}

// Finds the first constant in the layer body that can serve as linear weights.
static std::optional<ConvertedMatrix> convertMatrix(
    mlir::func::FuncOp func, mlir::RewriterBase &rewriter, int64_t arrayRows,
    int64_t arrayCols) {
  llvm::SmallVector<ConstantOp> constants;
  func.walk([&](ConstantOp constant) { constants.push_back(constant); });

  for (ConstantOp constant : constants) {
    if (auto converted =
            convertMatrixConstant(constant, rewriter, arrayRows, arrayCols))
      return converted;
  }

  return std::nullopt;
}

// Converts the single layer input into vector slices that match the matrix grid.
static std::optional<ConvertedVector> convertVector(
    mlir::func::FuncOp func, mlir::RewriterBase &rewriter, int64_t arrayRows,
    int64_t arrayCols) {
  if (func.getNumArguments() != 1)
    return std::nullopt;

  auto analogVector =
      converter_utils::materializeAnalogVector(func.getArgument(0), rewriter);
  if (failed(analogVector))
    return std::nullopt;

  auto partitionedVector = converter_utils::partitionAnalogVector(
      *analogVector, rewriter, arrayRows, arrayCols);
  if (failed(partitionedVector))
    return std::nullopt;

  auto placementLoop =
      converter_utils::placeAnalogVector(*partitionedVector, rewriter);
  if (failed(placementLoop))
    return std::nullopt;

  return ConvertedVector{*partitionedVector, *placementLoop};
}

// Locates the digital matmul whose result will be replaced by analog execution.
static MatmulOp findFirstMatmulOp(mlir::func::FuncOp func) {
  MatmulOp matmulOp;
  func.walk([&](MatmulOp op) {
    if (!matmulOp)
      matmulOp = op;
  });
  return matmulOp;
}

// Removes the now-dead matmul and weight-transpose scaffold after rewiring.
static void eraseUnusedLinearOps(MatmulOp matmulOp,
                                 mlir::RewriterBase &rewriter) {
  if (!matmulOp)
    return;

  TransposeOp weightTranspose =
      matmulOp->getOperand(0).getDefiningOp<TransposeOp>();
  if (!weightTranspose)
    weightTranspose = matmulOp->getOperand(1).getDefiningOp<TransposeOp>();

  EmptyOp weightTransposeEmpty;
  if (weightTranspose)
    weightTransposeEmpty =
        weightTranspose->getOperand(1).getDefiningOp<EmptyOp>();

  if (matmulOp->use_empty())
    rewriter.eraseOp(matmulOp);

  if (weightTranspose && weightTranspose->use_empty())
    rewriter.eraseOp(weightTranspose);

  if (weightTransposeEmpty && weightTransposeEmpty->use_empty())
    rewriter.eraseOp(weightTransposeEmpty);
}

// Converts extracted linear layer bodies from tensor matmul to analog arrays.
class LinearConverter : public mlir::analog::LayerConverter {
public:
  // Reports the layer_type key used by the converter dispatch table.
  mlir::StringRef getName() const override { return "linear"; }

  // Replaces a recognized linear body with placed array execution and reduction.
  void convert(mlir::func::FuncOp func, int64_t arrayRows,
               int64_t arrayCols) const override {
    mlir::IRRewriter rewriter(func.getContext());

    // Materialize and place both operands before execution; vector tiling relies
    // on the matrix grid recorded by the utility layer.
    auto matrix = convertMatrix(func, rewriter, arrayRows, arrayCols);
    auto vector = convertVector(func, rewriter, arrayRows, arrayCols);
    if (!matrix || !vector)
      return;

    auto executionBuffer = converter_utils::insertArrayExecution(
        matrix->partitionedMatrix, vector->partitionedVector,
        matrix->placementLoop, vector->placementLoop, rewriter);
    if (failed(executionBuffer))
      return;

    MatmulOp matmulOp = findFirstMatmulOp(func);
    if (!matmulOp)
      return;

    // Preserve the original matmul result contract, then drop dead digital IR.
    (void)converter_utils::insertArrayReduction(*executionBuffer,
                                                matrix->partitionedMatrix,
                                                matmulOp, rewriter);
    eraseUnusedLinearOps(matmulOp, rewriter);
    func->setAttr("layer_domain", rewriter.getStringAttr("analog"));
  }
};

} // namespace

namespace mlir {
namespace analog {

// Registers the linear converter for both biased and bias-free extracted layers.
void registerLinearConverter(LayerConverters &converters,
                             LayerConverterMap &converterMap,
                             MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<LinearConverter>();
  const LayerConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["linear"] = converterPtr;
  converterMap["linear_w_bias"] = converterPtr;
}

} // namespace analog
} // namespace mlir
