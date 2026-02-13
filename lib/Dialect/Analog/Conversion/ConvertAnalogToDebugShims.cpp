#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToDebugShims.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

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

static Value getDataPtrOperand(LLVM::CallOp call) {
  auto buildPtrWithOffset = [&](Value basePtr, Value offset) -> Value {
    if (!basePtr || !offset)
      return basePtr;
    if (!llvm::isa<IntegerType>(offset.getType()))
      return basePtr;
    OpBuilder b(call);
    auto elemTy = Float32Type::get(call.getContext());
    return b.create<LLVM::GEPOp>(call.getLoc(), basePtr.getType(), elemTy,
                                 basePtr, ValueRange{offset});
  };

  // For exploded memref calls, operands are typically:
  //   [allocated_ptr, aligned_ptr, offset, sizes..., strides..., tile_id]
  // and the first logical element base pointer is the aligned pointer.
  if (call.getNumOperands() >= 3 &&
      llvm::isa<LLVM::LLVMPointerType>(call.getOperand(1).getType()))
    return buildPtrWithOffset(call.getOperand(1), call.getOperand(2));

  Value ptr = call.getOperand(0);
  if (auto extract = ptr.getDefiningOp<LLVM::ExtractValueOp>()) {
    if (extract.getPosition().size() == 1 &&
        (extract.getPosition()[0] == 0 || extract.getPosition()[0] == 1)) {
      auto structTy =
          llvm::dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
      if (structTy && structTy.getBody().size() >= 3 &&
          llvm::isa<LLVM::LLVMPointerType>(structTy.getBody()[1])) {
        OpBuilder b(call);
        Value aligned = extract.getPosition()[0] == 1
                            ? ptr
                            : b.create<LLVM::ExtractValueOp>(
                                  call.getLoc(), extract.getContainer(),
                                  ArrayRef<int64_t>{1});
        Value offset = b.create<LLVM::ExtractValueOp>(
            call.getLoc(), extract.getContainer(), ArrayRef<int64_t>{2});
        return buildPtrWithOffset(aligned, offset);
      }
    }
  }

  return ptr;
}

} // namespace

llvm::StringRef ConvertAnalogToDebugShimsPass::getArgument() const {
  return "analog-convert-to-debug-shims";
}

llvm::StringRef ConvertAnalogToDebugShimsPass::getDescription() const {
  return "Rewrite analog backend calls to debug shim call targets for simulation/instrumentation";
}

void ConvertAnalogToDebugShimsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void ConvertAnalogToDebugShimsPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  getOrCreateLLVMFunc(
      module, "golem_debug_mvm_set",
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, "golem_debug_mvm_load",
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, "golem_debug_mvm_store",
      LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i32Ty}, false));
  getOrCreateLLVMFunc(
      module, "golem_debug_mvm_compute",
      LLVM::LLVMFunctionType::get(voidTy, {i32Ty}, false));

  module.walk([&](LLVM::CallOp call) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr) {
      return;
    }

    StringRef callee = calleeAttr.getValue();

    if (callee == "golem_analog_mvm_set" ||
        callee == "llvm.riscv.golem.analog.mvm.set") {
      if (call.getNumOperands() < 2) {
        return;
      }

      Value ptr = getDataPtrOperand(call);
      Value tileId = call.getOperand(call.getNumOperands() - 1);

      OpBuilder b(call);
      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, "golem_debug_mvm_set"),
          SmallVector<Value>{ptr, tileId});
      call.erase();
      return;
    }

    if (callee == "golem_analog_mvm_load" ||
        callee == "llvm.riscv.golem.analog.mvm.load") {
      if (call.getNumOperands() < 2) {
        return;
      }

      Value ptr = getDataPtrOperand(call);
      Value tileId = call.getOperand(call.getNumOperands() - 1);

      OpBuilder b(call);
      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, "golem_debug_mvm_load"),
          SmallVector<Value>{ptr, tileId});
      call.erase();
      return;
    }

    if (callee == "golem_analog_mvm_store" ||
        callee == "llvm.riscv.golem.analog.mvm.store") {
      if (call.getNumOperands() < 2) {
        return;
      }

      Value ptr = getDataPtrOperand(call);
      Value tileId = call.getOperand(call.getNumOperands() - 1);

      OpBuilder b(call);
      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, "golem_debug_mvm_store"),
          SmallVector<Value>{ptr, tileId});
      call.erase();
      return;
    }

    if (callee == "golem_analog_mvm_compute" ||
        callee == "llvm.riscv.golem.analog.mvm") {
      if (call.getNumOperands() < 1) {
        return;
      }

      Value tileId = call.getOperand(call.getNumOperands() - 1);

      OpBuilder b(call);
      b.create<LLVM::CallOp>(
          call.getLoc(), TypeRange{},
          SymbolRefAttr::get(ctx, "golem_debug_mvm_compute"),
          SmallVector<Value>{tileId});
      call.erase();
      return;
    }
  });

  for (StringRef oldName : {
           "golem_analog_mvm_set",
           "golem_analog_mvm_load",
           "golem_analog_mvm_store",
           "golem_analog_mvm_compute",
           "llvm.riscv.golem.analog.mvm.set",
           "llvm.riscv.golem.analog.mvm.load",
           "llvm.riscv.golem.analog.mvm.store",
           "llvm.riscv.golem.analog.mvm",
       }) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(oldName)) {
      if (fn.use_empty()) {
        fn.erase();
      }
    }
  }
}

std::unique_ptr<mlir::Pass> createConvertAnalogToDebugShimsPass() {
  return std::make_unique<ConvertAnalogToDebugShimsPass>();
}

} // namespace analog
} // namespace mlir
