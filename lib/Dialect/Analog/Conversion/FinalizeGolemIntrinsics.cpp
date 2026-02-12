#include "analog-mlir/Dialect/Analog/Conversion/FinalizeGolemIntrinsics.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

static LLVM::LLVMFuncOp getOrCreateLLVMFunc(ModuleOp module, StringRef name,
                                            LLVM::LLVMFunctionType type) {

  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return fn;
  }

  OpBuilder b(module.getBodyRegion());
  auto fn = b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

} // namespace

llvm::StringRef FinalizeGolemIntrinsicsPass::getArgument() const {
  return "analog-finalize-golem-intrinsics";
}

llvm::StringRef FinalizeGolemIntrinsicsPass::getDescription() const {
  return "Rewrite golem wrapper calls into final LLVM RISC-V golem intrinsic calls";
}

void FinalizeGolemIntrinsicsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void FinalizeGolemIntrinsicsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);

  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm.set",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrTy, i32Ty},
                                  false));
  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm.load",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrTy, i32Ty},
                                  false));
  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm.store",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {ptrTy, i32Ty},
                                  false));
  getOrCreateLLVMFunc(
      module, "llvm.riscv.golem.analog.mvm",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {i32Ty}, false));

  module.walk([&](LLVM::CallOp call) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr) {
      return;
    }

    StringRef callee = calleeAttr.getValue();

    if (callee == "golem_analog_mvm_set" || 
        callee == "golem_analog_mvm_load" ||
        callee == "golem_analog_mvm_store") {

      if (call.getNumOperands() < 2) {
        return;
      }

      Value ptr = call.getOperand(0);
      Value tileId = call.getOperand(call.getNumOperands() - 1);
      StringRef dst = callee == "golem_analog_mvm_set"
                          ? "llvm.riscv.golem.analog.mvm.set"
                          : (callee == "golem_analog_mvm_load"
                                 ? "llvm.riscv.golem.analog.mvm.load"
                                 : "llvm.riscv.golem.analog.mvm.store");

      OpBuilder b(call);
      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, dst),
          SmallVector<Value>{ptr, tileId}
      );

      call.erase();
      return;
    }

    if (callee == "golem_analog_mvm_compute") {

      if (call.getNumOperands() < 1) {
        return;
      }

      Value tileId = call.getOperand(call.getNumOperands() - 1);
      OpBuilder b(call);

      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, "llvm.riscv.golem.analog.mvm"),
          SmallVector<Value>{tileId}
      );

      call.erase();
      return;
    }
  });

  for (StringRef oldName : {"golem_analog_mvm_set",
                            "golem_analog_mvm_load",
                            "golem_analog_mvm_store",
                            "golem_analog_mvm_compute"}) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(oldName)) {
      if (fn.use_empty()) {
        fn.erase();
      }
    }
  }
}

std::unique_ptr<mlir::Pass> createFinalizeGolemIntrinsicsPass() {
  return std::make_unique<FinalizeGolemIntrinsicsPass>();
}

} // namespace analog
} // namespace mlir
