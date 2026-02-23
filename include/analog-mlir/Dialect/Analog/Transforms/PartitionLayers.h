#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PARTITION_LAYERS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PARTITION_LAYERS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <llvm/Support/CommandLine.h>
#include <memory>

namespace mlir {
namespace analog {

struct PartitionLayersPass
    : public mlir::PassWrapper<PartitionLayersPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionLayersPass)

  Option<int64_t> num_cores{
      *this, "num-cores",
      llvm::cl::desc("Number of cores to map layer groups onto"),
      llvm::cl::init(2)};

  PartitionLayersPass() = default;
  PartitionLayersPass(const PartitionLayersPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createPartitionLayersPass();

} // namespace analog
} // namespace mlir

#endif
