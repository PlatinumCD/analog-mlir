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

// Carries the ops and boundary values that make up one linear layer slice.
struct LinearMatch {
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *weightTransposeEmpty = nullptr;
  mlir::Operation *weightTranspose = nullptr;
  mlir::Operation *outputEmpty = nullptr;
  mlir::Operation *outputFill = nullptr;
  mlir::Operation *outputFillConstant = nullptr;
  mlir::Operation *matmulOp = nullptr;
  mlir::Operation *biasConstant = nullptr;
  mlir::Operation *biasAddOp = nullptr;

  mlir::Operation *root = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
};

// Collects the full matched linear subgraph in cloning order.
static void collectMatchedOps(LinearMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.weightTransposeEmpty);
  match_utils::appendUniqueOp(match.ops, match.weightTranspose);
  match_utils::appendUniqueOp(match.ops, match.outputEmpty);
  match_utils::appendUniqueOp(match.ops, match.outputFillConstant);
  match_utils::appendUniqueOp(match.ops, match.outputFill);
  match_utils::appendUniqueOp(match.ops, match.matmulOp);
  match_utils::appendUniqueOp(match.ops, match.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.biasAddOp);
}

// Computes the complete match boundary before rewriting.
static void finalizeLinearMatch(LinearMatch &match) {
  collectMatchedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Validates the shared matmul core and records its weight and output-init ops.
static mlir::LogicalResult matchLinearCore(LinearMatch &match) {
  mlir::Operation *op = match.matmulOp;

  // Verifies the matmul operand layout.
  if (!extractor_utils::hasOperands(op, 3))
    return mlir::failure();

  if (!extractor_utils::hasInputs(op, 2))
    return mlir::failure();

  // Finds the transposed weight input and the opposite data input.
  auto weightTranspose = extractor_utils::defOpAs<TransposeOp>(op, 0);
  unsigned inputIndex = 0;
  if (!weightTranspose)
    weightTranspose = extractor_utils::defOpAs<TransposeOp>(op, 1);
  else
    inputIndex = 1;

  if (!weightTranspose)
    return mlir::failure();

  // Verifies the transpose shape and its backing weight ops.
  if (!extractor_utils::hasOperands(weightTranspose, 2))
    return mlir::failure();

  if (!extractor_utils::hasInputs(weightTranspose, 1))
    return mlir::failure();

  auto weightConstant =
      extractor_utils::defOpAs<ConstantOp>(weightTranspose.getOperation(), 0);
  if (!weightConstant)
    return mlir::failure();

  auto weightTransposeEmpty =
      extractor_utils::defOpAs<EmptyOp>(weightTranspose.getOperation(), 1);
  if (!weightTransposeEmpty)
    return mlir::failure();

  // Records the non-transpose matmul input as the layer input.
  if (!match.inputs.empty())
    match.inputs.clear();
  match.inputs.push_back(op->getOperand(inputIndex));

  auto outputInit = extractor_impl::matchFillOutputInit(op, 2);
  if (!outputInit)
    return mlir::failure();

  // Stores the shared core ops once the full chain is validated.
  match.weightConstant = weightConstant.getOperation();
  match.weightTransposeEmpty = weightTransposeEmpty.getOperation();
  match.weightTranspose = weightTranspose.getOperation();
  match.outputFill = outputInit->outputFill;
  match.outputFillConstant = outputInit->outputFillConstant;
  match.outputEmpty = outputInit->outputEmpty;
  return mlir::success();
}

// Recognizes linear layers where a bias constant is added to the matmul result.
static std::optional<LinearMatch> matchLinearWithBias(mlir::Operation *op) {
  // Anchors the biased form on a two-input add generic.
  if (!extractor_utils::isAddfGeneric(op))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 2))
    return std::nullopt;

  // Requires the add inputs to be a bias constant and a matmul.
  if (!extractor_utils::inputsAreEither<ConstantOp, MatmulOp>(op))
    return std::nullopt;

  auto biasConstant = extractor_utils::defOpAs<ConstantOp>(op, 0);
  auto matmulOp = extractor_utils::defOpAs<MatmulOp>(op, 1);

  if (!biasConstant || !matmulOp) {
    biasConstant = extractor_utils::defOpAs<ConstantOp>(op, 1);
    matmulOp = extractor_utils::defOpAs<MatmulOp>(op, 0);
  }

  // Seeds the match with the bias-specific anchor ops.
  LinearMatch match;
  match.root = op;
  match.biasAddOp = op;
  match.biasConstant = biasConstant.getOperation();
  match.matmulOp = matmulOp.getOperation();

  // Hands off to the shared matmul core matcher.
  if (mlir::failed(matchLinearCore(match)))
    return std::nullopt;

  finalizeLinearMatch(match);
  return match;
}

// Recognizes linear layers represented directly by the matmul result.
static std::optional<LinearMatch> matchLinearWithoutBias(mlir::Operation *op) {
  // Anchors the bias-free form directly on matmul.
  auto matmulOp = llvm::dyn_cast<MatmulOp>(op);
  if (!matmulOp)
    return std::nullopt;

  // Seeds the match from the shared matmul anchor.
  LinearMatch match;
  match.root = op;
  match.matmulOp = matmulOp.getOperation();

  // Reuses the same core checks as the biased path.
  if (mlir::failed(matchLinearCore(match)))
    return std::nullopt;

  finalizeLinearMatch(match);
  return match;
}

// Outlines the matched linear subgraph into a new function and replaces it
// with a call at the original root.
static void rewriteLinearExtractor(const LinearMatch &match,
                                   mlir::RewriterBase &rewriter) {
  mlir::StringRef layerType =
      match.biasAddOp ? mlir::StringRef("linear_w_bias")
                      : mlir::StringRef("linear");
  rewrite_utils::extractToFunction(match.root, match.ops, match.inputs,
                                   match.outputs, rewriter, layerType);
}

// Finds linalg-based linear layers and outlines each match.
class LinearExtractor : public mlir::analog::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though Linear stores no state.
  explicit LinearExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "linear"; }

  // Repeatedly extracts biased forms before matching the bias-free fallback.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    extractor_impl::extractAllMatches(func, rewriter, matchLinearWithBias,
                                      rewriteLinearExtractor);
    extractor_impl::extractAllMatches(func, rewriter, matchLinearWithoutBias,
                                      rewriteLinearExtractor);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Adds the linear extractor to the layer extraction pipeline.
void registerLinearExtractor(LayerExtractors &extractors, MLIRContext *context) {
  extractors.push_back(std::make_unique<LinearExtractor>(context));
}

} // namespace analog
} // namespace mlir
