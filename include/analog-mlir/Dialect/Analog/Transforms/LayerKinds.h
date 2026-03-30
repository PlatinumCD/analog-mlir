#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_LAYER_KINDS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_LAYER_KINDS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace analog {
namespace detail {

enum class LayerKind {
  linear,
  conv1d,
  conv2d,
  groupedConv2d,
  conv3d,
};

struct LayerKindInfo {
  LayerKind kind;
  llvm::StringRef rewrittenOutputAttr;
  llvm::StringRef routineAttr;
  llvm::StringRef routinePrefix;
};

llvm::ArrayRef<LayerKindInfo> getLayerKindInfos();
const LayerKindInfo *findLayerKindByRewrittenOutputAttr(llvm::StringRef attrName);
const LayerKindInfo *findLayerKindByRoutineAttr(llvm::StringRef attrName);
const LayerKindInfo *findLayerKindByRoutinePrefix(llvm::StringRef symbolName);

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_LAYER_KINDS_H
