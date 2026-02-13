#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PARTITION_LAYERS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PARTITION_LAYERS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace analog {

struct PartitionLayersPass
    : public mlir::PassWrapper<PartitionLayersPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionLayersPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createPartitionLayersPass();

} // namespace analog
} // namespace mlir

#endif
