#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PARTITION_MATRIX_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PARTITION_MATRIX_H

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

struct PartitionMatrixPass
    : public mlir::PassWrapper<PartitionMatrixPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionMatrixPass)

  // ---- REQUIRED ----
  PartitionMatrixPass() = default;
  PartitionMatrixPass(
      const PartitionMatrixPass &other)
      : PassWrapper(other) {}

  Option<int64_t> array_rows {*this, "array-rows",
      llvm::cl::desc("Number of rows per analog array"),
      llvm::cl::init(16)
  };

  Option<int64_t> array_cols {*this, "array-cols",
      llvm::cl::desc("Number of cols per analog array"),
      llvm::cl::init(16)
  };

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};


std::unique_ptr<mlir::Pass> createPartitionMatrixPass();
std::unique_ptr<mlir::Pass> createPartitionMatrixPass(int64_t arrayRows, int64_t arrayCols);


} // namespace analog
} // namespace mlir

#endif

