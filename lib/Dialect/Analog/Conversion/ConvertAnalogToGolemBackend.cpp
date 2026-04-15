#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"
#include "analog-mlir/Dialect/Analog/Conversion/golem/GolemUtils.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace analog {

namespace {

// Collects backend operation lowerings and function boundary type rewrites.
void populateConversionPatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx) {
  golem::populateLowerMatrixPatterns(patterns, typeConverter, ctx);
  golem::populateLowerVectorPatterns(patterns, typeConverter, ctx);
  golem::populateLowerExecutePatterns(patterns, typeConverter, ctx);
  golem::populateLowerStorePatterns(patterns, typeConverter, ctx);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

// Preserves task-graph shell ops while requiring lowered types everywhere else.
void configureConversionTarget(ConversionTarget &target,
                               TypeConverter &typeConverter) {
  target.addIllegalDialect<analog::AnalogDialect>();
  target.addLegalOp<analog::TaskGraphCreateOp, analog::TaskGraphInputOp,
                    analog::TaskGraphOutputOp, analog::TaskGraphTemporaryOp,
                    analog::TaskGraphPersistentOp, analog::TaskCreateOp>();
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<arith::ArithDialect, bufferization::BufferizationDialect,
                         LLVM::LLVMDialect, memref::MemRefDialect,
                         scf::SCFDialect, tensor::TensorDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return typeConverter.isSignatureLegal(op.getCalleeType()) &&
           typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return typeConverter.isLegal(op->getOperandTypes());
  });
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return typeConverter.isLegal(op); });
}

} // namespace

// Applies the Golem type and operation conversion to the whole module.
void ConvertAnalogToGolemBackendPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp module = getOperation();

  TypeConverter typeConverter;
  golem::populateAnalogTypeConversions(typeConverter);

  RewritePatternSet patterns(ctx);
  populateConversionPatterns(patterns, typeConverter, ctx);

  ConversionTarget target(*ctx);
  configureConversionTarget(target, typeConverter);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

// Exposes the pass to MLIR textual pipelines and pass managers.
void registerConvertAnalogToGolemBackendPass() {
  PassRegistration<ConvertAnalogToGolemBackendPass>();
}

} // namespace analog
} // namespace mlir
