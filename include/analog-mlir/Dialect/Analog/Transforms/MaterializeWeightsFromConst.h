#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_MATERIALIZE_WEIGHTS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_MATERIALIZE_WEIGHTS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace analog {

struct MaterializeWeightsFromConstPass
    : public mlir::PassWrapper<MaterializeWeightsFromConstPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeWeightsFromConstPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};


std::unique_ptr<mlir::Pass> createMaterializeWeightsFromConstPass();

} // namespace analog
} // namespace mlir

#endif

