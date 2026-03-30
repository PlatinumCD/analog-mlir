#include "analog-mlir/Dialect/Analog/Transforms/LayerKinds.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformAttrs.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace detail {

namespace {

inline constexpr llvm::StringLiteral kLinearRoutineAttr = "analog-linear-routine";
inline constexpr llvm::StringLiteral kConv1DRoutineAttr = "analog-conv1d-routine";
inline constexpr llvm::StringLiteral kConv2DRoutineAttr = "analog-conv2d-routine";
inline constexpr llvm::StringLiteral kGroupedConv2DRoutineAttr =
    "analog-grouped-conv2d-routine";
inline constexpr llvm::StringLiteral kConv3DRoutineAttr = "analog-conv3d-routine";

inline constexpr llvm::StringLiteral kLinearRoutinePrefix = "analog_linear_routine_";
inline constexpr llvm::StringLiteral kConv1DRoutinePrefix = "analog_conv1d_routine_";
inline constexpr llvm::StringLiteral kConv2DRoutinePrefix = "analog_conv2d_routine_";
inline constexpr llvm::StringLiteral kGroupedConv2DRoutinePrefix =
    "analog_grouped_conv2d_routine_";
inline constexpr llvm::StringLiteral kConv3DRoutinePrefix = "analog_conv3d_routine_";

inline constexpr llvm::StringLiteral kRewrittenConv1DOutputAttr =
    "analog.rewritten_conv1d_output";
inline constexpr llvm::StringLiteral kRewrittenConv2DOutputAttr =
    "analog.rewritten_conv2d_output";
inline constexpr llvm::StringLiteral kRewrittenGroupedConv2DOutputAttr =
    "analog.rewritten_grouped_conv2d_output";
inline constexpr llvm::StringLiteral kRewrittenConv3DOutputAttr =
    "analog.rewritten_conv3d_output";

constexpr LayerKindInfo kLayerKindInfos[] = {
    {LayerKind::linear, "", kLinearRoutineAttr, kLinearRoutinePrefix},
    {LayerKind::conv1d, kRewrittenConv1DOutputAttr, kConv1DRoutineAttr,
     kConv1DRoutinePrefix},
    {LayerKind::conv2d, kRewrittenConv2DOutputAttr, kConv2DRoutineAttr,
     kConv2DRoutinePrefix},
    {LayerKind::groupedConv2d, kRewrittenGroupedConv2DOutputAttr,
     kGroupedConv2DRoutineAttr, kGroupedConv2DRoutinePrefix},
    {LayerKind::conv3d, kRewrittenConv3DOutputAttr, kConv3DRoutineAttr,
     kConv3DRoutinePrefix},
};

} // namespace

llvm::ArrayRef<LayerKindInfo> getLayerKindInfos() { return kLayerKindInfos; }

const LayerKindInfo *findLayerKindByRewrittenOutputAttr(llvm::StringRef attrName) {
  for (const LayerKindInfo &info : kLayerKindInfos) {
    if (!info.rewrittenOutputAttr.empty() && info.rewrittenOutputAttr == attrName)
      return &info;
  }
  return nullptr;
}

const LayerKindInfo *findLayerKindByRoutineAttr(llvm::StringRef attrName) {
  for (const LayerKindInfo &info : kLayerKindInfos) {
    if (info.routineAttr == attrName)
      return &info;
  }
  return nullptr;
}

const LayerKindInfo *findLayerKindByRoutinePrefix(llvm::StringRef symbolName) {
  for (const LayerKindInfo &info : kLayerKindInfos) {
    if (symbolName.starts_with(info.routinePrefix))
      return &info;
  }
  return nullptr;
}

} // namespace detail
} // namespace analog
} // namespace mlir
