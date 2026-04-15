#include "analog-mlir/Dialect/Analog/Transforms/ExtractLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/MatchUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/RewriteUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorImplementationUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <memory>
#include <optional>

namespace extractor_utils = mlir::analog::extractor_utils;
namespace extractor_impl = mlir::analog::extractor_impl;
namespace match_utils = mlir::analog::match_utils;
namespace rewrite_utils = mlir::analog::rewrite_utils;

namespace {

using mlir::arith::ConstantOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::EmptyOp;

// Carries the ops and boundary values that make up one RNN cell slice.
struct RNNCellMatch {
  mlir::Operation *biasOp = nullptr;
  mlir::Operation *preActivationAddOp = nullptr;
  mlir::Operation *rnnCellOp = nullptr;
  mlir::Operation *sharedOutputEmpty = nullptr;
  mlir::Operation *sharedFillOp = nullptr;
  mlir::Operation *sharedFillConstant = nullptr;

  mlir::Operation *firstWeightConstant = nullptr;
  mlir::Operation *firstTransposeEmpty = nullptr;
  mlir::Operation *firstTranspose = nullptr;
  mlir::Operation *firstMatmul = nullptr;
  mlir::Operation *firstBiasConstant = nullptr;
  mlir::Operation *firstBiasAdd = nullptr;

  mlir::Operation *secondWeightConstant = nullptr;
  mlir::Operation *secondTransposeEmpty = nullptr;
  mlir::Operation *secondTranspose = nullptr;
  mlir::Operation *secondMatmul = nullptr;
  mlir::Operation *secondBiasConstant = nullptr;
  mlir::Operation *secondBiasAdd = nullptr;

  mlir::Operation *root = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
};

// Captures one matmul branch that contributes to the RNN pre-activation sum.
struct RNNCellMatmulBranchMatch {
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *transposeEmpty = nullptr;
  mlir::Operation *transpose = nullptr;
  mlir::Operation *matmul = nullptr;
  mlir::Operation *fill = nullptr;
  mlir::Operation *fillConstant = nullptr;
  mlir::Operation *biasAdd = nullptr;
  mlir::Operation *biasConstant = nullptr;
};

// Collects both branches, shared initialization, optional biases, and tanh in
// cloning order.
static void collectMatchedOps(RNNCellMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.firstWeightConstant);
  match_utils::appendUniqueOp(match.ops, match.firstTransposeEmpty);
  match_utils::appendUniqueOp(match.ops, match.firstTranspose);
  match_utils::appendUniqueOp(match.ops, match.secondWeightConstant);
  match_utils::appendUniqueOp(match.ops, match.secondTransposeEmpty);
  match_utils::appendUniqueOp(match.ops, match.secondTranspose);
  match_utils::appendUniqueOp(match.ops, match.sharedOutputEmpty);
  match_utils::appendUniqueOp(match.ops, match.sharedFillConstant);
  match_utils::appendUniqueOp(match.ops, match.sharedFillOp);
  match_utils::appendUniqueOp(match.ops, match.firstMatmul);
  match_utils::appendUniqueOp(match.ops, match.firstBiasConstant);
  match_utils::appendUniqueOp(match.ops, match.firstBiasAdd);
  match_utils::appendUniqueOp(match.ops, match.secondMatmul);
  match_utils::appendUniqueOp(match.ops, match.secondBiasConstant);
  match_utils::appendUniqueOp(match.ops, match.secondBiasAdd);
  match_utils::appendUniqueOp(match.ops, match.biasOp);
  match_utils::appendUniqueOp(match.ops, match.preActivationAddOp);
  match_utils::appendUniqueOp(match.ops, match.rnnCellOp);
}

// Computes the complete match boundary before outlining the cell.
static void finalizeRNNCellMatch(RNNCellMatch &match) {
  collectMatchedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Validates the shared transposed-weight matmul core used by one RNN branch.
static mlir::LogicalResult
matchRNNMatmulBranchCore(MatmulOp matmul, mlir::Operation *sharedOutputEmpty,
                         RNNCellMatmulBranchMatch &branch) {
  // Verifies the matmul shape and requires the shared output initializer.
  if (!extractor_utils::hasOperands(matmul.getOperation(), 3))
    return mlir::failure();

  if (!extractor_utils::hasInputs(matmul.getOperation(), 2))
    return mlir::failure();

  auto outputInit =
      extractor_impl::matchFillOutputInit(matmul.getOperation(), 2);
  if (!outputInit)
    return mlir::failure();

  if (outputInit->outputEmpty != sharedOutputEmpty)
    return mlir::failure();

  // Finds the single transposed weight operand regardless of branch ordering.
  auto firstTranspose =
      extractor_utils::defOpAs<TransposeOp>(matmul.getOperation(), 0);
  auto secondTranspose =
      extractor_utils::defOpAs<TransposeOp>(matmul.getOperation(), 1);
  if (static_cast<bool>(firstTranspose) == static_cast<bool>(secondTranspose))
    return mlir::failure();

  auto transpose = firstTranspose ? firstTranspose : secondTranspose;
  if (!extractor_utils::hasOperands(transpose.getOperation(), 2))
    return mlir::failure();

  if (!extractor_utils::hasInputs(transpose.getOperation(), 1))
    return mlir::failure();

  auto weightConstant =
      extractor_utils::defOpAs<ConstantOp>(transpose.getOperation(), 0);
  if (!weightConstant)
    return mlir::failure();

  auto transposeEmpty =
      extractor_utils::defOpAs<EmptyOp>(transpose.getOperation(), 1);
  if (!transposeEmpty)
    return mlir::failure();

  // Records the validated branch core for the outer RNN matcher.
  branch.weightConstant = weightConstant.getOperation();
  branch.transposeEmpty = transposeEmpty.getOperation();
  branch.transpose = transpose.getOperation();
  branch.matmul = matmul.getOperation();
  branch.fill = outputInit->outputFill;
  branch.fillConstant = outputInit->outputFillConstant;
  return mlir::success();
}

// Recognizes a branch where the matmul result receives its own bias constant.
static mlir::LogicalResult
matchRNNMatmulBranchWithBias(mlir::Operation *branchAdd,
                             mlir::Operation *sharedOutputEmpty,
                             RNNCellMatmulBranchMatch &branch) {
  // Requires the branch add to write into the cell's shared output tensor.
  if (!extractor_utils::isAddfGeneric(branchAdd))
    return mlir::failure();

  if (!extractor_utils::hasOperands(branchAdd, 3))
    return mlir::failure();

  if (!extractor_utils::hasInputs(branchAdd, 2))
    return mlir::failure();

  auto branchOutputEmpty = extractor_utils::defOpAs<EmptyOp>(branchAdd, 2);
  if (!branchOutputEmpty)
    return mlir::failure();

  if (branchOutputEmpty.getOperation() != sharedOutputEmpty)
    return mlir::failure();

  if (!extractor_utils::inputsAreEither<MatmulOp, ConstantOp>(branchAdd))
    return mlir::failure();

  // Splits the add operands before handing the matmul to the shared core.
  auto matmul = extractor_utils::defOpAs<MatmulOp>(branchAdd, 0);
  auto biasConstant = extractor_utils::defOpAs<ConstantOp>(branchAdd, 1);
  if (!matmul || !biasConstant) {
    matmul = extractor_utils::defOpAs<MatmulOp>(branchAdd, 1);
    biasConstant = extractor_utils::defOpAs<ConstantOp>(branchAdd, 0);
  }

  if (!matmul || !biasConstant)
    return mlir::failure();

  branch.biasAdd = branchAdd;
  branch.biasConstant = biasConstant.getOperation();
  if (mlir::failed(matchRNNMatmulBranchCore(matmul, sharedOutputEmpty, branch)))
    return mlir::failure();

  return mlir::success();
}

// Recognizes a bias-free branch represented directly by its matmul.
static mlir::LogicalResult
matchRNNMatmulBranchWithoutBias(mlir::Operation *branchMatmul,
                                mlir::Operation *sharedOutputEmpty,
                                RNNCellMatmulBranchMatch &branch) {
  auto matmul = llvm::dyn_cast<MatmulOp>(branchMatmul);
  if (!matmul)
    return mlir::failure();

  return matchRNNMatmulBranchCore(matmul, sharedOutputEmpty, branch);
}

// Recognizes tanh(add(branch_with_bias, branch_with_bias)) RNN cell bodies.
static std::optional<RNNCellMatch> matchRNNCellWithBias(mlir::Operation *op) {
  // Anchors on tanh and captures the output buffer shared by the full cell.
  if (!extractor_utils::isTanhGeneric(op))
    return std::nullopt;

  if (!extractor_utils::hasOperands(op, 2))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 1))
    return std::nullopt;

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 1);
  if (!outputEmpty)
    return std::nullopt;

  // Finds the pre-activation add that combines input and recurrent branches.
  mlir::Operation *inputAdd = extractor_utils::defOp(op, 0);
  if (!extractor_utils::isAddfGeneric(inputAdd))
    return std::nullopt;

  if (!extractor_utils::hasOperands(inputAdd, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(inputAdd, 2))
    return std::nullopt;

  auto inputAddOutputEmpty = extractor_utils::defOpAs<EmptyOp>(inputAdd, 2);
  if (!inputAddOutputEmpty)
    return std::nullopt;

  if (inputAddOutputEmpty.getOperation() != outputEmpty.getOperation())
    return std::nullopt;

  // Matches both biased branches against the same output initializer.
  RNNCellMatmulBranchMatch firstBranch;
  if (mlir::failed(matchRNNMatmulBranchWithBias(extractor_utils::defOp(inputAdd, 0),
                                                outputEmpty.getOperation(),
                                                firstBranch)))
    return std::nullopt;

  RNNCellMatmulBranchMatch secondBranch;
  if (mlir::failed(matchRNNMatmulBranchWithBias(extractor_utils::defOp(inputAdd, 1),
                                                outputEmpty.getOperation(),
                                                secondBranch)))
    return std::nullopt;

  if (firstBranch.fill != secondBranch.fill)
    return std::nullopt;

  // Materializes the layer match from the validated branch pieces.
  RNNCellMatch match;
  match.biasOp = inputAdd;
  match.preActivationAddOp = inputAdd;
  match.rnnCellOp = op;
  match.sharedOutputEmpty = outputEmpty.getOperation();
  match.sharedFillOp = firstBranch.fill;
  match.sharedFillConstant = firstBranch.fillConstant;
  match.firstWeightConstant = firstBranch.weightConstant;
  match.firstTransposeEmpty = firstBranch.transposeEmpty;
  match.firstTranspose = firstBranch.transpose;
  match.firstMatmul = firstBranch.matmul;
  match.firstBiasConstant = firstBranch.biasConstant;
  match.firstBiasAdd = firstBranch.biasAdd;
  match.secondWeightConstant = secondBranch.weightConstant;
  match.secondTransposeEmpty = secondBranch.transposeEmpty;
  match.secondTranspose = secondBranch.transpose;
  match.secondMatmul = secondBranch.matmul;
  match.secondBiasConstant = secondBranch.biasConstant;
  match.secondBiasAdd = secondBranch.biasAdd;
  match.root = op;
  finalizeRNNCellMatch(match);
  return match;
}

// Recognizes tanh(add(matmul, matmul)) RNN cell bodies without branch biases.
static std::optional<RNNCellMatch> matchRNNCellWithoutBias(mlir::Operation *op) {
  // Shares the same tanh and pre-activation scaffold as the biased matcher.
  if (!extractor_utils::isTanhGeneric(op))
    return std::nullopt;

  if (!extractor_utils::hasOperands(op, 2))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 1))
    return std::nullopt;

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(op, 1);
  if (!outputEmpty)
    return std::nullopt;

  mlir::Operation *inputAdd = extractor_utils::defOp(op, 0);
  if (!extractor_utils::isAddfGeneric(inputAdd))
    return std::nullopt;

  if (!extractor_utils::hasOperands(inputAdd, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(inputAdd, 2))
    return std::nullopt;

  auto inputAddOutputEmpty = extractor_utils::defOpAs<EmptyOp>(inputAdd, 2);
  if (!inputAddOutputEmpty)
    return std::nullopt;

  if (inputAddOutputEmpty.getOperation() != outputEmpty.getOperation())
    return std::nullopt;

  // Requires both add operands to be raw matmul branches.
  RNNCellMatmulBranchMatch firstBranch;
  if (mlir::failed(matchRNNMatmulBranchWithoutBias(extractor_utils::defOp(inputAdd, 0),
                                                   outputEmpty.getOperation(),
                                                   firstBranch)))
    return std::nullopt;

  RNNCellMatmulBranchMatch secondBranch;
  if (mlir::failed(matchRNNMatmulBranchWithoutBias(extractor_utils::defOp(inputAdd, 1),
                                                   outputEmpty.getOperation(),
                                                   secondBranch)))
    return std::nullopt;

  if (firstBranch.fill != secondBranch.fill)
    return std::nullopt;

  // Materializes the layer match from the validated branch pieces.
  RNNCellMatch match;
  match.preActivationAddOp = inputAdd;
  match.rnnCellOp = op;
  match.sharedOutputEmpty = outputEmpty.getOperation();
  match.sharedFillOp = firstBranch.fill;
  match.sharedFillConstant = firstBranch.fillConstant;
  match.firstWeightConstant = firstBranch.weightConstant;
  match.firstTransposeEmpty = firstBranch.transposeEmpty;
  match.firstTranspose = firstBranch.transpose;
  match.firstMatmul = firstBranch.matmul;
  match.secondWeightConstant = secondBranch.weightConstant;
  match.secondTransposeEmpty = secondBranch.transposeEmpty;
  match.secondTranspose = secondBranch.transpose;
  match.secondMatmul = secondBranch.matmul;
  match.root = op;
  finalizeRNNCellMatch(match);
  return match;
}

// Outlines the matched RNN cell into a layer function and replaces the root.
static void rewriteRNNCellExtractor(const RNNCellMatch &match,
                                    mlir::RewriterBase &rewriter) {
  mlir::StringRef layerType =
      match.biasOp ? mlir::StringRef("rnn_cell_w_bias")
                   : mlir::StringRef("rnn_cell");
  rewrite_utils::extractToFunction(match.root, match.ops, match.inputs,
                                   match.outputs, rewriter, layerType);
}

// Finds linalg-based simple RNN cells and outlines each match.
class RNNCellExtractor : public mlir::analog::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though this extractor is
  // stateless.
  explicit RNNCellExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "rnn_cell"; }

  // Extracts biased RNN cells before trying the bias-free fallback.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    extractor_impl::extractAllMatches(func, rewriter, matchRNNCellWithBias,
                                      rewriteRNNCellExtractor);
    extractor_impl::extractAllMatches(func, rewriter, matchRNNCellWithoutBias,
                                      rewriteRNNCellExtractor);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Adds the RNN cell extractor to the layer extraction pipeline.
void registerRNNCellExtractor(LayerExtractors &extractors,
                              MLIRContext *context) {
  extractors.push_back(std::make_unique<RNNCellExtractor>(context));
}

} // namespace analog
} // namespace mlir
