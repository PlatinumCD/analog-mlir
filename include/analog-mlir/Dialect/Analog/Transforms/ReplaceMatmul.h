#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REPLACE_MATMUL_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REPLACE_MATMUL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <llvm/Support/CommandLine.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace analog {

struct ReplaceMatmulPass
    : public mlir::PassWrapper<ReplaceMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceMatmulPass)

  // ---- REQUIRED ----
  ReplaceMatmulPass() = default;
  ReplaceMatmulPass(
      const ReplaceMatmulPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};


std::unique_ptr<mlir::Pass> createReplaceMatmulPass();

} // namespace analog
} // namespace mlir

#endif

