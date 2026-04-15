#include "analog-mlir/Dialect/Analog/Transforms/ExtractLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/MatchUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/RewriteUtils.h"
#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorImplementationUtils.h"

#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

namespace extractor_utils = mlir::analog::extractor_utils;
namespace extractor_impl = mlir::analog::extractor_impl;
namespace match_utils = mlir::analog::match_utils;
namespace rewrite_utils = mlir::analog::rewrite_utils;

namespace {

// Carries the ops and boundary values that make up one Conv1D layer slice.
struct Conv1DMatch {
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *biasConstant = nullptr;
  mlir::Operation *outputEmpty = nullptr;
  mlir::Operation *outputFill = nullptr;
  mlir::Operation *outputFillConstant = nullptr;
  mlir::Operation *outputBroadcast = nullptr;
  mlir::Operation *convOp = nullptr;

  mlir::Operation *root = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
};

// Collects the matched Conv1D slice in clone order for outlining.
static void collectMatchedOps(Conv1DMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.outputEmpty);
  match_utils::appendUniqueOp(match.ops, match.outputFillConstant);
  match_utils::appendUniqueOp(match.ops, match.outputFill);
  match_utils::appendUniqueOp(match.ops, match.outputBroadcast);
  match_utils::appendUniqueOp(match.ops, match.convOp);
}

// Computes the external inputs and root outputs after all match ops are known.
static void finalizeConv1DMatch(Conv1DMatch &match) {
  collectMatchedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Recognizes Conv1D layers whose output init broadcasts a constant bias.
static std::optional<Conv1DMatch> matchConv1DWithBias(mlir::Operation *op) {
  // Anchor the match on the linalg Conv1D op and its expected operand shape.
  if (!op || op->getName().getStringRef() != "linalg.conv_1d_ncw_fcw")
    return std::nullopt;

  if (!extractor_utils::hasOperands(op, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 2))
    return std::nullopt;

  // Require the output init to be the bias broadcast used by this pattern.
  auto outputInit = extractor_impl::matchBroadcastOutputInit(op, 2);
  if (!outputInit)
    return std::nullopt;

  // Require exactly one convolution input to be the captured weight constant.
  auto weightInput = extractor_impl::matchSingleConstantInputPair(op);
  if (!weightInput)
    return std::nullopt;

  // Materialize the match only after both bias and weight structure validate.
  Conv1DMatch match;
  match.root = op;
  match.convOp = op;
  match.biasConstant = outputInit->biasConstant;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputBroadcast = outputInit->outputBroadcast;
  match.weightConstant = weightInput->constant;
  finalizeConv1DMatch(match);
  return match;
}

// Recognizes Conv1D layers whose output init is a fill rather than a bias.
static std::optional<Conv1DMatch> matchConv1DWithoutBias(mlir::Operation *op) {
  // Anchor the match on the linalg Conv1D op and its expected operand shape.
  if (!op || op->getName().getStringRef() != "linalg.conv_1d_ncw_fcw")
    return std::nullopt;

  if (!extractor_utils::hasOperands(op, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 2))
    return std::nullopt;

  // Require the output init to be a fill so no bias op is captured.
  auto outputInit = extractor_impl::matchFillOutputInit(op, 2);
  if (!outputInit)
    return std::nullopt;

  // Require exactly one convolution input to be the captured weight constant.
  auto weightInput = extractor_impl::matchSingleConstantInputPair(op);
  if (!weightInput)
    return std::nullopt;

  // Materialize the match only after both init and weight structure validate.
  Conv1DMatch match;
  match.root = op;
  match.convOp = op;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputFill = outputInit->outputFill;
  match.outputFillConstant = outputInit->outputFillConstant;
  match.weightConstant = weightInput->constant;
  finalizeConv1DMatch(match);
  return match;
}

// Outlines the matched Conv1D slice and tags the new layer by bias form.
static void rewriteConv1DExtractor(const Conv1DMatch &match,
                                   mlir::RewriterBase &rewriter) {
  mlir::StringRef layerType =
      match.outputBroadcast ? mlir::StringRef("conv1d_w_bias")
                            : mlir::StringRef("conv1d");
  rewrite_utils::extractToFunction(match.root, match.ops, match.inputs,
                                   match.outputs, rewriter, layerType);
}

// Finds one-dimensional convolution layers and outlines each match.
class Conv1DExtractor : public mlir::analog::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though Conv1D stores no state.
  explicit Conv1DExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "conv1d"; }

  // Repeatedly extracts biased forms before matching the bias-free fallback.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    extractor_impl::extractAllMatches(func, rewriter, matchConv1DWithBias,
                                      rewriteConv1DExtractor);
    extractor_impl::extractAllMatches(func, rewriter, matchConv1DWithoutBias,
                                      rewriteConv1DExtractor);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Adds the Conv1D extractor to the layer extraction pipeline.
void registerConv1DExtractor(LayerExtractors &extractors,
                             MLIRContext *context) {
  extractors.push_back(std::make_unique<Conv1DExtractor>(context));
}

} // namespace analog
} // namespace mlir
