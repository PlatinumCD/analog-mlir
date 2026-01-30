#ifndef ANALOG_MLIR_TRANSFORMS_NN_TAGNNLAYERINDEX_H
#define ANALOG_MLIR_TRANSFORMS_NN_TAGNNLAYERINDEX_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nn {

struct TagNNLayerIndexPass
    : public mlir::PassWrapper<TagNNLayerIndexPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TagNNLayerIndexPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void runOnOperation() override;
};

/// Factory
std::unique_ptr<mlir::Pass> createTagNNLayerIndexPass();

} // namespace nn
} // namespace mlir

#endif

