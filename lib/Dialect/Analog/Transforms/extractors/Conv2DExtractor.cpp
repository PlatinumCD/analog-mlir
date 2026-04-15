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

// Carries the ops and boundary values that make up one Conv2D layer slice.
struct Conv2DMatch {
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

// Collects the matched Conv2D slice in clone order for outlining.
static void collectMatchedOps(Conv2DMatch &match) {
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
static void finalizeConv2DMatch(Conv2DMatch &match) {
  collectMatchedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Recognizes Conv2D layers whose output init broadcasts a constant bias.
static std::optional<Conv2DMatch> matchConv2DWithBias(mlir::Operation *op) {
  // Anchor the match on the linalg Conv2D op and its expected operand shape.
  if (!op || op->getName().getStringRef() != "linalg.conv_2d_nchw_fchw")
    return std::nullopt;

  if (!extractor_utils::hasOperands(op, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 2))
    return std::nullopt;

  // Require the output init to be the bias broadcast used by this pattern.
  auto outputInit = extractor_impl::matchBroadcastOutputInit(op, 2);
  if (!outputInit)
    return std::nullopt;

  // Accept the static weight from either convolution input position.
  mlir::Operation *weightConstant =
      extractor_impl::findConstantInput(op, 1, 0);
  if (!weightConstant)
    return std::nullopt;

  // Materialize the match only after both bias and weight structure validate.
  Conv2DMatch match;
  match.root = op;
  match.convOp = op;
  match.biasConstant = outputInit->biasConstant;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputBroadcast = outputInit->outputBroadcast;
  match.weightConstant = weightConstant;
  finalizeConv2DMatch(match);
  return match;
}

// Recognizes Conv2D layers whose output init is a fill rather than a bias.
static std::optional<Conv2DMatch> matchConv2DWithoutBias(mlir::Operation *op) {
  // Anchor the match on the linalg Conv2D op and its expected operand shape.
  if (!op || op->getName().getStringRef() != "linalg.conv_2d_nchw_fchw")
    return std::nullopt;

  if (!extractor_utils::hasOperands(op, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(op, 2))
    return std::nullopt;

  // Require the output init to be a fill so no bias op is captured.
  auto outputInit = extractor_impl::matchFillOutputInit(op, 2);
  if (!outputInit)
    return std::nullopt;

  // Accept the static weight from either convolution input position.
  mlir::Operation *weightConstant =
      extractor_impl::findConstantInput(op, 1, 0);
  if (!weightConstant)
    return std::nullopt;

  // Materialize the match only after both init and weight structure validate.
  Conv2DMatch match;
  match.root = op;
  match.convOp = op;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputFill = outputInit->outputFill;
  match.outputFillConstant = outputInit->outputFillConstant;
  match.weightConstant = weightConstant;
  finalizeConv2DMatch(match);
  return match;
}

// Outlines the matched Conv2D slice and tags the new layer by bias form.
static void rewriteConv2DExtractor(const Conv2DMatch &match,
                                   mlir::RewriterBase &rewriter) {
  mlir::StringRef layerType =
      match.outputBroadcast ? mlir::StringRef("conv2d_w_bias")
                            : mlir::StringRef("conv2d");
  rewrite_utils::extractToFunction(match.root, match.ops, match.inputs,
                                   match.outputs, rewriter, layerType);
}

// Finds two-dimensional convolution layers and outlines each match.
class Conv2DExtractor : public mlir::analog::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though Conv2D stores no state.
  explicit Conv2DExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "conv2d"; }

  // Repeatedly extracts biased forms before matching the bias-free fallback.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    extractor_impl::extractAllMatches(func, rewriter, matchConv2DWithBias,
                                      rewriteConv2DExtractor);
    extractor_impl::extractAllMatches(func, rewriter, matchConv2DWithoutBias,
                                      rewriteConv2DExtractor);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Adds the Conv2D extractor to the layer extraction pipeline.
void registerConv2DExtractor(LayerExtractors &extractors,
                             MLIRContext *context) {
  extractors.push_back(std::make_unique<Conv2DExtractor>(context));
}

} // namespace analog
} // namespace mlir
