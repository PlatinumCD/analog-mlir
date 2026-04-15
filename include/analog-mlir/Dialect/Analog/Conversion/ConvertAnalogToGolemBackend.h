#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERTANALOGTOGOLEMBACKEND_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERTANALOGTOGOLEMBACKEND_H

#include "analog-mlir/Dialect/Analog/IR/AnalogDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace analog {

// Drives the module-level conversion from backend-ready Analog IR into
// Golem-compatible MLIR.
struct ConvertAnalogToGolemBackendPass
    : public mlir::PassWrapper<ConvertAnalogToGolemBackendPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAnalogToGolemBackendPass)

  // Returns the stable pipeline flag used by MLIR pass registration.
  mlir::StringRef getArgument() const final {
    return "analog-lower-to-golem";
  }

  // Summarizes the lowering pass in MLIR pass-help output.
  mlir::StringRef getDescription() const final {
    return "Lower Analog MLIR to Golem";
  }

  // Declares the dialects that conversion patterns and legality checks may
  // touch.
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::analog::AnalogDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect>();
  }

  // Applies the partial conversion and signals pass failure when illegal IR
  // remains.
  void runOnOperation() override;
};

// Makes the backend lowering pass available to textual pipelines and pass
// managers.
void registerConvertAnalogToGolemBackendPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERTANALOGTOGOLEMBACKEND_H
