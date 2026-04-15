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
using mlir::linalg::FillOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

// Carries the grouped Conv2D subgraph and boundary values to outline.
struct Conv2DGroupedMatch {
  mlir::Operation *inputExpand = nullptr;
  mlir::Operation *groupedConvOp = nullptr;
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *weightExpand = nullptr;
  mlir::Operation *biasConstant = nullptr;
  mlir::Operation *outputEmpty = nullptr;
  mlir::Operation *outputFill = nullptr;
  mlir::Operation *outputFillConstant = nullptr;
  mlir::Operation *outputBroadcast = nullptr;
  mlir::Operation *outputExpand = nullptr;
  mlir::Operation *outputCollapse = nullptr;

  mlir::Operation *root = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
};

// Collects the matched grouped Conv2D slice in clone order for outlining.
static void collectMatchedOps(Conv2DGroupedMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.inputExpand);
  match_utils::appendUniqueOp(match.ops, match.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.weightExpand);
  match_utils::appendUniqueOp(match.ops, match.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.outputEmpty);
  match_utils::appendUniqueOp(match.ops, match.outputFillConstant);
  match_utils::appendUniqueOp(match.ops, match.outputBroadcast);
  match_utils::appendUniqueOp(match.ops, match.outputExpand);
  match_utils::appendUniqueOp(match.ops, match.outputFill);
  match_utils::appendUniqueOp(match.ops, match.groupedConvOp);
  match_utils::appendUniqueOp(match.ops, match.outputCollapse);
}

// Computes external inputs and root outputs after optional ops are ordered.
static void finalizeConv2DGroupedMatch(Conv2DGroupedMatch &match) {
  collectMatchedOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Recognizes grouped Conv2D layers whose output init broadcasts a bias.
static std::optional<Conv2DGroupedMatch>
matchConv2DGroupedWithBias(mlir::Operation *op) {
  // Anchor the match on the result collapse that wraps grouped convolution.
  auto outputCollapse = llvm::dyn_cast<CollapseShapeOp>(op);
  if (!outputCollapse)
    return std::nullopt;

  if (!extractor_utils::hasOperands(outputCollapse.getOperation(), 1))
    return std::nullopt;

  // Require the collapsed value to come from the grouped linalg convolution.
  mlir::Operation *groupedConvOp =
      extractor_utils::defOp(outputCollapse.getSrc());
  if (!groupedConvOp ||
      groupedConvOp->getName().getStringRef() != "linalg.conv_2d_ngchw_gfchw")
    return std::nullopt;

  if (!extractor_utils::hasOperands(groupedConvOp, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(groupedConvOp, 2))
    return std::nullopt;

  // Validate the expanded output init before matching the bias broadcast.
  auto outputExpand = extractor_utils::defOpAs<ExpandShapeOp>(groupedConvOp, 2);
  if (!outputExpand)
    return std::nullopt;

  if (!extractor_utils::hasOperands(outputExpand.getOperation(), 1))
    return std::nullopt;

  auto outputInit =
      extractor_impl::matchBroadcastOutputInit(outputExpand.getSrc());
  if (!outputInit)
    return std::nullopt;

  // Require one expanded dynamic input and one expanded constant weight.
  auto weightMatch = extractor_impl::matchExpandedInputAndWeight(groupedConvOp);
  if (!weightMatch)
    return std::nullopt;

  // Materialize the match only after bias, weight, and shape wrappers validate.
  Conv2DGroupedMatch match;
  match.root = outputCollapse.getOperation();
  match.outputCollapse = outputCollapse.getOperation();
  match.groupedConvOp = groupedConvOp;
  match.inputExpand = weightMatch->inputExpand;
  match.weightConstant = weightMatch->weightConstant;
  match.weightExpand = weightMatch->weightExpand;
  match.biasConstant = outputInit->biasConstant;
  match.outputEmpty = outputInit->outputEmpty;
  match.outputExpand = outputExpand.getOperation();
  match.outputBroadcast = outputInit->outputBroadcast;

  finalizeConv2DGroupedMatch(match);
  return match;
}

// Recognizes grouped Conv2D layers whose output init is a fill, not a bias.
static std::optional<Conv2DGroupedMatch>
matchConv2DGroupedWithoutBias(mlir::Operation *op) {
  // Anchor the match on the result collapse that wraps grouped convolution.
  auto outputCollapse = llvm::dyn_cast<CollapseShapeOp>(op);
  if (!outputCollapse)
    return std::nullopt;

  if (!extractor_utils::hasOperands(outputCollapse.getOperation(), 1))
    return std::nullopt;

  // Require the collapsed value to come from the grouped linalg convolution.
  mlir::Operation *groupedConvOp =
      extractor_utils::defOp(outputCollapse.getSrc());
  if (!groupedConvOp ||
      groupedConvOp->getName().getStringRef() != "linalg.conv_2d_ngchw_gfchw")
    return std::nullopt;

  if (!extractor_utils::hasOperands(groupedConvOp, 3))
    return std::nullopt;

  if (!extractor_utils::hasInputs(groupedConvOp, 2))
    return std::nullopt;

  // Match the bias-free init chain: fill constant into expanded empty tensor.
  auto outputFill = extractor_utils::defOpAs<FillOp>(groupedConvOp, 2);
  if (!outputFill)
    return std::nullopt;

  if (!extractor_utils::hasOperands(outputFill.getOperation(), 2))
    return std::nullopt;

  if (!extractor_utils::hasInputs(outputFill.getOperation(), 1))
    return std::nullopt;

  auto outputFillConstant =
      extractor_utils::defOpAs<ConstantOp>(outputFill.getOperation(), 0);
  if (!outputFillConstant)
    return std::nullopt;

  auto outputExpand =
      extractor_utils::defOpAs<ExpandShapeOp>(outputFill.getOperation(), 1);
  if (!outputExpand)
    return std::nullopt;

  if (!extractor_utils::hasOperands(outputExpand.getOperation(), 1))
    return std::nullopt;

  auto outputEmpty = extractor_utils::defOpAs<EmptyOp>(outputExpand.getSrc());
  if (!outputEmpty)
    return std::nullopt;

  // Require one expanded dynamic input and one expanded constant weight.
  auto weightMatch = extractor_impl::matchExpandedInputAndWeight(groupedConvOp);
  if (!weightMatch)
    return std::nullopt;

  // Materialize the match only after fill, weight, and shape wrappers validate.
  Conv2DGroupedMatch match;
  match.root = outputCollapse.getOperation();
  match.inputExpand = weightMatch->inputExpand;
  match.groupedConvOp = groupedConvOp;
  match.weightConstant = weightMatch->weightConstant;
  match.weightExpand = weightMatch->weightExpand;
  match.outputEmpty = outputEmpty.getOperation();
  match.outputFill = outputFill.getOperation();
  match.outputFillConstant = outputFillConstant.getOperation();
  match.outputExpand = outputExpand.getOperation();
  match.outputCollapse = outputCollapse.getOperation();
  finalizeConv2DGroupedMatch(match);
  return match;
}

// Outlines the grouped Conv2D slice and tags it by whether bias was captured.
static void rewriteConv2DGroupedExtractor(const Conv2DGroupedMatch &match,
                                          mlir::RewriterBase &rewriter) {
  mlir::StringRef layerType =
      match.outputBroadcast ? mlir::StringRef("conv2d_grouped_w_bias")
                            : mlir::StringRef("conv2d_grouped");
  rewrite_utils::extractToFunction(match.root, match.ops, match.inputs,
                                   match.outputs, rewriter, layerType);
}

// Finds grouped two-dimensional convolution layers and outlines each match.
class Conv2DGroupedExtractor : public mlir::analog::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though no state is stored.
  explicit Conv2DGroupedExtractor(mlir::MLIRContext *context) {
    (void)context;
  }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "conv2d_grouped"; }

  // Extracts biased grouped convolutions before the bias-free fallback.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    extractor_impl::extractAllMatches(func, rewriter,
                                      matchConv2DGroupedWithBias,
                                      rewriteConv2DGroupedExtractor);
    extractor_impl::extractAllMatches(func, rewriter,
                                      matchConv2DGroupedWithoutBias,
                                      rewriteConv2DGroupedExtractor);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Adds the grouped Conv2D extractor to the layer extraction pipeline.
void registerConv2DGroupedExtractor(LayerExtractors &extractors,
                                    MLIRContext *context) {
  extractors.push_back(std::make_unique<Conv2DGroupedExtractor>(context));
}

} // namespace analog
} // namespace mlir
